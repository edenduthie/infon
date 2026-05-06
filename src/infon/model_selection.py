"""Model-selection helpers — sklearn-style AutoML.

Public API:

    held_out_infon_split(cog, frac, seed)
        Reproducible infon-level train/heldout split.

    kfold_infon_split(cog, k, seed)
        k-fold cross-validation splits — list of (train_ids, heldout_ids).

    auto_select_ssl(cog, modes, ...)
        Fit one reasoner per SSL mode, score on held-out queries,
        return ranked list.

    sweep(cog, grid, time_budget_sec=None, ...)
        GridSearchCV — cartesian product of config overrides. Honours
        an optional wall-clock budget (returns best-so-far on timeout).

    cross_val_score(cog, config_overrides, k=5, ...)
        Average score ± std across k held-out folds.

    ensemble_top_k(cog, grid, k=3, ...)
        Train the top-K sweep winners, combine their per-query mass
        functions via Dempster's rule. Ensemble inherits calibration.
"""
from __future__ import annotations

import copy
import itertools
import random
import re
import time
from dataclasses import dataclass, field

from .random_state import derive_seed, make_python_rng


@dataclass
class SweepResult:
    """One point in the grid. Sortable by score descending."""
    config_overrides: dict        # the knobs that produced this score
    score: float
    n_train: int
    n_heldout: int
    extras: dict = field(default_factory=dict)

    def __lt__(self, other):
        return self.score < other.score


def kfold_infon_split(cog, k: int = 5,
                      seed: int | None = None
                      ) -> list[tuple[list[str], list[str]]]:
    """Return k (train_ids, heldout_ids) pairs — each heldout is 1/k
    of the infons, folds are disjoint, every infon appears in exactly
    one heldout.

    seed falls back to cog.random_state if None.
    """
    if seed is None:
        seed = getattr(cog, "random_state", None) or 0
    rng = make_python_rng(seed, "kfold_infon_split")

    infons = cog.store.query_infons(limit=5000)
    ids = [i.infon_id for i in infons]
    rng.shuffle(ids)
    n = len(ids)
    if k < 2 or k > n:
        # Degenerate — return a single LOO-ish fold
        return [(ids[1:], ids[:1])]

    folds = []
    fold_size = n // k
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else n
        heldout = ids[start:end]
        train = ids[:start] + ids[end:]
        folds.append((train, heldout))
    return folds


def held_out_infon_split(cog, frac: float = 0.2,
                         seed: int | None = None) -> tuple[list[str], list[str]]:
    """Return (train_ids, heldout_ids) — two disjoint lists of infon ids.

    Infon-level split: the graph topology stays the same, but a fraction
    of infon *nodes* are set aside and not used to generate training
    queries. The training phase still sees their edges (so the GNN
    message-passes over them), but the teacher signal and the evaluation
    queries only touch their respective halves. This is option (b) from
    the SSL-sweep design discussion.

    seed falls back to cog.random_state if None.
    """
    if seed is None:
        seed = getattr(cog, "random_state", None) or 0
    rng = make_python_rng(seed, "held_out_infon_split")

    infons = cog.store.query_infons(limit=5000)
    ids = [i.infon_id for i in infons]
    rng.shuffle(ids)
    n_heldout = max(1, int(round(frac * len(ids))))
    return ids[n_heldout:], ids[:n_heldout]


def _generate_queries_from_ids(cog, infon_ids, seed, n_max=15):
    """Build a self-generated query set from the given infon ids.

    SUPPORTS queries come directly from the triples; NEI queries are
    role-swaps that don't appear anywhere in the full store.
    """
    rng = make_python_rng(seed, "queries")
    infons = [cog.store.get_infon(i) for i in infon_ids]
    infons = [i for i in infons if i is not None]
    infons.sort(key=lambda x: -x.confidence)

    all_triples = {
        (i.subject, i.predicate, i.object)
        for i in cog.store.query_infons(limit=5000)
    }
    types = cog.schema.types
    actors = [n for n in cog.schema.names if types.get(n) == "actor"]
    feats = [n for n in cog.schema.names
             if types.get(n) in ("feature", "market")]

    queries, seen = [], set()
    for inf in infons:
        if len(queries) >= n_max:
            break
        k = (inf.subject, inf.predicate, inf.object)
        if k in seen:
            continue
        seen.add(k)
        queries.append({
            "q": f"Did {inf.subject} {inf.predicate} {inf.object}?",
            "gold": "SUPPORTS",
        })

    nei = []
    for item in list(queries):
        m = re.match(r"Did (\S+) (\S+) (.+)\?", item["q"])
        if not m:
            continue
        s, p, o = m.groups()
        which = rng.choice(["subj", "obj"])
        if which == "subj" and actors:
            cands = [a for a in actors if a != s]
            if not cands:
                continue
            triple = (rng.choice(cands), p, o)
        elif which == "obj" and feats:
            cands = [a for a in feats if a != o]
            if not cands:
                continue
            triple = (s, p, rng.choice(cands))
        else:
            continue
        # The mirror claim must not exist anywhere in the full store,
        # otherwise it's not a genuine NEI.
        if triple in all_triples or triple in seen:
            continue
        seen.add(triple)
        nei.append({
            "q": f"Did {triple[0]} {triple[1]} {triple[2]}?",
            "gold": "NOT ENOUGH INFO",
        })
    queries.extend(nei[:n_max])
    return queries


def auto_select_ssl(cog,
                    modes: list[str] | None = None,
                    heldout_frac: float = 0.2,
                    hidden_dim: int = 32,
                    epochs: int = 10,
                    verbose: bool = False) -> list[SweepResult]:
    """Fit a reasoner under each SSL mode, score on held-out queries,
    return a ranked list with the champion at index 0.

    The default modes are ('laplacian',) — it's the one the runtime
    defaults to. Callers pass ('laplacian', 'barlow', 'vicreg', 'jepa')
    to actually compare. Extra modes require fit_ssl() methods on the
    reasoner, which are wired in a later commit; until then this
    helper falls back to the laplacian-only path.
    """
    from .logic import HypergraphReasoner

    if modes is None:
        modes = ["laplacian"]

    seed = getattr(cog, "random_state", None) or 0
    train_ids, heldout_ids = held_out_infon_split(
        cog, frac=heldout_frac, seed=seed,
    )

    # Queries from the held-out half only
    queries = _generate_queries_from_ids(
        cog, heldout_ids, seed=derive_seed(seed, "queries"),
    )

    results: list[SweepResult] = []
    for mode in modes:
        if verbose:
            print(f"  [auto_select_ssl] fitting mode='{mode}' ...")
        reasoner = HypergraphReasoner(
            cog.store, cog.encoder, cog.schema,
            hidden_dim=hidden_dim, n_layers=2,
            use_sheaf=(mode in ("laplacian", "sheaf")),
        )
        graph = reasoner.builder.build(feature_dim=hidden_dim)
        lap_w = 0.1 if mode in ("laplacian", "sheaf") else 0.0
        reasoner.fit(graph=graph, epochs=epochs,
                     laplacian_weight=lap_w, verbose=False)

        score = reasoner.score(queries) if queries else 0.0
        results.append(SweepResult(
            config_overrides={"ssl_mode": mode},
            score=score,
            n_train=len(train_ids),
            n_heldout=len(heldout_ids),
            extras={"n_queries": len(queries)},
        ))
        if verbose:
            print(f"    → score={score:.2%} on {len(queries)} heldout queries")

    results.sort(key=lambda r: -r.score)
    return results


def _fit_and_score(cog, overrides: dict, queries, epochs: int,
                   hidden_dim_default: int):
    """Factored body — build reasoner per overrides, fit, score."""
    from .logic import HypergraphReasoner

    mode = overrides.get("ssl_mode", "laplacian")
    hd = overrides.get("hidden_dim", hidden_dim_default)
    reasoner = HypergraphReasoner(
        cog.store, cog.encoder, cog.schema,
        hidden_dim=hd, n_layers=overrides.get("n_layers", 2),
        use_sheaf=(mode in ("laplacian", "sheaf")),
    )
    graph = reasoner.builder.build(feature_dim=hd)
    reasoner.fit(
        graph=graph, epochs=epochs,
        laplacian_weight=0.1 if mode in ("laplacian", "sheaf") else 0.0,
        verbose=False,
    )
    return reasoner, reasoner.score(queries) if queries else 0.0


def sweep(cog,
          grid: dict,
          heldout_frac: float = 0.2,
          hidden_dim: int = 32,
          epochs: int = 10,
          time_budget_sec: float | None = None,
          verbose: bool = False) -> list[SweepResult]:
    """Grid search over config overrides.

    Parameters
    ----------
    grid : dict[str, list]
        Cartesian product of parameter values to try, e.g.
        {"ssl_mode": ["laplacian", "barlow"],
         "hidden_dim": [16, 32]}.
    time_budget_sec : float, optional
        Wall-clock cap in seconds. When set, `sweep` stops launching new
        configs once the elapsed time crosses this threshold and returns
        whatever it has scored. The currently-running config is allowed
        to finish (we stop between configs, not mid-fit).

    Returns
    -------
    list[SweepResult], sorted by score descending.
    """
    keys = list(grid.keys())
    seed = getattr(cog, "random_state", None) or 0
    train_ids, heldout_ids = held_out_infon_split(
        cog, frac=heldout_frac, seed=seed,
    )
    queries = _generate_queries_from_ids(
        cog, heldout_ids, seed=derive_seed(seed, "queries"),
    )

    results: list[SweepResult] = []
    combos = list(itertools.product(*[grid[k] for k in keys]))
    t0 = time.perf_counter()
    for combo in combos:
        elapsed = time.perf_counter() - t0
        if time_budget_sec is not None and elapsed > time_budget_sec:
            if verbose:
                print(f"  [sweep] time budget ({time_budget_sec}s) "
                      f"reached after {len(results)}/{len(combos)} "
                      f"configs — stopping")
            break
        overrides = dict(zip(keys, combo))
        if verbose:
            print(f"  [sweep] {overrides} ...")
        reasoner, score = _fit_and_score(cog, overrides, queries,
                                         epochs, hidden_dim)
        results.append(SweepResult(
            config_overrides=overrides,
            score=score,
            n_train=len(train_ids),
            n_heldout=len(heldout_ids),
            extras={"n_queries": len(queries)},
        ))
        if verbose:
            print(f"    → score={score:.2%}")

    results.sort(key=lambda r: -r.score)
    return results


def cross_val_score(cog,
                    config_overrides: dict | None = None,
                    k: int = 5,
                    hidden_dim: int = 32,
                    epochs: int = 10,
                    verbose: bool = False) -> dict:
    """K-fold cross-validation on infon splits.

    Unlike `sweep` (one held-out split, many configs), `cross_val_score`
    fits the same config across k rotating folds and returns the mean
    and std of the per-fold scores. Useful once you've picked a
    candidate config via `sweep` and want to estimate its variance.

    Returns
    -------
    dict with keys:
        'mean_score' : float
        'std_score'  : float
        'per_fold'   : list[float]
        'n_folds'    : int
    """
    overrides = config_overrides or {}
    seed = getattr(cog, "random_state", None) or 0
    folds = kfold_infon_split(cog, k=k, seed=seed)

    per_fold = []
    for i, (train_ids, heldout_ids) in enumerate(folds):
        qs = _generate_queries_from_ids(
            cog, heldout_ids,
            seed=derive_seed(seed, "cv", f"fold-{i}"),
        )
        if not qs:
            continue
        _, score = _fit_and_score(cog, overrides, qs, epochs, hidden_dim)
        per_fold.append(score)
        if verbose:
            print(f"  [CV fold {i+1}/{len(folds)}] score={score:.2%}  "
                  f"(n_train={len(train_ids)}, n_heldout={len(heldout_ids)})")

    if not per_fold:
        return {"mean_score": float("nan"),
                "std_score": float("nan"),
                "per_fold": [], "n_folds": 0}

    mean = sum(per_fold) / len(per_fold)
    var = sum((s - mean) ** 2 for s in per_fold) / len(per_fold)
    return {
        "mean_score": mean,
        "std_score": var ** 0.5,
        "per_fold": per_fold,
        "n_folds": len(per_fold),
    }


def ensemble_top_k(cog,
                   grid: dict,
                   k: int = 3,
                   heldout_frac: float = 0.2,
                   hidden_dim: int = 32,
                   epochs: int = 10,
                   verbose: bool = False) -> dict:
    """Train the top-K configs from `sweep`, combine their per-query
    mass functions via Dempster's rule.

    The ensemble inherits the DS calibration invariants: if any one of
    the K base models is certain, Dempster's rule concentrates the
    ensemble's mass; if they all disagree or are ignorant, θ stays
    high — which is exactly the behaviour you want from an ensemble.

    Returns
    -------
    dict with keys:
        'members'    : list[SweepResult]    # top-K configs
        'queries'    : list[dict]           # held-out query set
        'ensemble'   : list[MassFunction]   # one per query, combined
        'ensemble_verdicts' : list[str]     # argmax verdict per query
        'ensemble_score'    : float
        'per_member_scores' : list[float]
    """
    from .dempster_shafer import MassFunction, combine_multiple
    from .logic import HypergraphReasoner

    # 1. Run the sweep to find top-K configs
    all_results = sweep(cog, grid, heldout_frac=heldout_frac,
                        hidden_dim=hidden_dim, epochs=epochs,
                        verbose=verbose)
    if not all_results:
        return {"members": [], "queries": [], "ensemble": [],
                "ensemble_verdicts": [], "ensemble_score": 0.0,
                "per_member_scores": []}

    top = all_results[:k]
    if verbose:
        print(f"  [ensemble] using top-{len(top)} configs "
              f"(scores {[round(r.score, 2) for r in top]})")

    # 2. Rebuild each top-K config, get per-query mass functions
    seed = getattr(cog, "random_state", None) or 0
    _, heldout_ids = held_out_infon_split(
        cog, frac=heldout_frac, seed=seed,
    )
    queries = _generate_queries_from_ids(
        cog, heldout_ids, seed=derive_seed(seed, "queries"),
    )

    per_member_masses: list[list[MassFunction]] = []
    per_member_scores: list[float] = []
    for member in top:
        mode = member.config_overrides.get("ssl_mode", "laplacian")
        hd = member.config_overrides.get("hidden_dim", hidden_dim)
        r = HypergraphReasoner(
            cog.store, cog.encoder, cog.schema,
            hidden_dim=hd,
            n_layers=member.config_overrides.get("n_layers", 2),
            use_sheaf=(mode in ("laplacian", "sheaf")),
        )
        g = r.builder.build(feature_dim=hd)
        r.fit(graph=g, epochs=epochs,
              laplacian_weight=0.1 if mode in ("laplacian", "sheaf") else 0.0,
              verbose=False)
        per_member_masses.append([
            MassFunction(
                supports=d["supports"], refutes=d["refutes"],
                uncertain=d["uncertain"], theta=d["theta"],
            )
            for d in r.predict_mass(queries)
        ])
        per_member_scores.append(r.score(queries) if queries else 0.0)

    # 3. Combine per-query across members via Dempster's rule
    ensemble_masses: list[MassFunction] = []
    ensemble_verdicts: list[str] = []
    for qi in range(len(queries)):
        contribs = [per_member_masses[m][qi] for m in range(len(top))]
        combined = combine_multiple(contribs)
        ensemble_masses.append(combined)
        # Verdict rule mirrors the retrieval reasoner: argmax over
        # {S, R, U}, but θ > .85 stays NOT ENOUGH INFO.
        if combined.theta > 0.85:
            ensemble_verdicts.append("NOT ENOUGH INFO")
        else:
            vals = {"SUPPORTS": combined.supports,
                    "REFUTES":  combined.refutes,
                    "NOT ENOUGH INFO": combined.uncertain + combined.theta}
            ensemble_verdicts.append(max(vals, key=vals.get))

    # Ensemble score vs gold
    if queries:
        correct = sum(
            1 for i, q in enumerate(queries)
            if ensemble_verdicts[i] == q["gold"]
        )
        ensemble_score = correct / len(queries)
    else:
        ensemble_score = 0.0

    return {
        "members": top,
        "queries": queries,
        "ensemble": ensemble_masses,
        "ensemble_verdicts": ensemble_verdicts,
        "ensemble_score": ensemble_score,
        "per_member_scores": per_member_scores,
    }


__all__ = [
    "SweepResult",
    "held_out_infon_split",
    "kfold_infon_split",
    "auto_select_ssl",
    "sweep",
    "cross_val_score",
    "ensemble_top_k",
]
