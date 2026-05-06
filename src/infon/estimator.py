"""Estimator contract — sklearn-style fit / predict / score, wrapped
around the existing reason() / query() / expand() methods.

Goal: let every reasoning-time object answer three questions uniformly:
    - .fit(X, y=None)    train on a graph (X) with optional supervision
    - .predict(X)         return a verdict or label for each query
    - .score(X, y)        return a scalar accuracy-style metric

The existing rich return types (ReasoningResult, QueryResult) stay as-is
— predict() is just a thin wrapper that returns the bare verdict label.
Users who want the full mass function keep calling reason().

This file only defines the protocol + utility helpers. Wrapper methods
are attached to HypergraphReasoner via patch_estimator_api().
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Estimator(Protocol):
    """Minimal sklearn-style contract for Infon reasoning objects.

    Any object with these three methods can be dropped into the
    model-selection helpers (sweep, cross_val_score, ...) without
    knowing its underlying type.
    """

    def fit(self, X: Any = None, y: Any = None) -> "Estimator":
        """Train on X (a HyperGraph or an Infon instance).

        y is optional — when present it's used as supervised signal;
        when None the estimator falls back to self-supervised defaults.
        """

    def predict(self, X: Any) -> Any:
        """Return a prediction per query.

        For a reasoner: a list of verdict strings.
        For a head (next-anchor, etc.): the head's raw output.
        """

    def score(self, X: Any, y: Any) -> float:
        """Scalar accuracy-style metric. Higher is better."""


# ═════════════════════════════════════════════════════════════════════
# HypergraphReasoner estimator methods
# ═════════════════════════════════════════════════════════════════════

def reasoner_predict(self, queries) -> list[str]:
    """sklearn-style .predict(): return verdict strings.

    Parameters
    ----------
    queries : list of str | list of dict
        - ["Did Toyota invest in batteries?", ...]
        - [{"q": "...", "gold": "SUPPORTS"}, ...]  (gold ignored)

    Returns
    -------
    list of str
        One of "SUPPORTS" / "REFUTES" / "NOT ENOUGH INFO" /
        "CONTRADICTED" per query. Note the space in "NOT ENOUGH INFO"
        is deliberate — it's the shipping verdict spelling from
        dempster_shafer.py. If your gold labels use an underscore,
        normalize them before comparing.
    """
    out = []
    for q in queries:
        text = q if isinstance(q, str) else q.get("q") or q.get("query")
        r = self.reason(text)
        out.append(r.verdict)
    return out


def reasoner_predict_mass(self, queries) -> list[dict]:
    """Like predict() but returns the full mass function per query.

    Returns
    -------
    list of dict with keys {verdict, supports, refutes, uncertain, theta}
    """
    out = []
    for q in queries:
        text = q if isinstance(q, str) else q.get("q") or q.get("query")
        r = self.reason(text)
        out.append({
            "verdict":   r.verdict,
            "supports":  r.mass.supports,
            "refutes":   r.mass.refutes,
            "uncertain": r.mass.uncertain,
            "theta":     r.mass.theta,
        })
    return out


def reasoner_score(self, queries, y=None) -> float:
    """Accuracy on a list of (query, gold) pairs.

    Accepts either:
      - queries=list of str, y=list of str   (parallel arrays)
      - queries=list of dict with 'gold' key, y=None  (self-contained)
    """
    if y is None:
        if not queries or not isinstance(queries[0], dict):
            raise ValueError("score() needs y or dict queries with 'gold'")
        gold = [q["gold"] for q in queries]
        qs = queries
    else:
        if len(queries) != len(y):
            raise ValueError("queries and y have different lengths")
        gold = list(y)
        qs = list(queries)
    preds = self.predict(qs)
    correct = sum(1 for p, g in zip(preds, gold) if p == g)
    return correct / max(len(gold), 1)


def reasoner_transform(self, queries) -> list[dict]:
    """Like .transform() in sklearn — returns activations + retrieved
    infons for each query. Equivalent to running `cog.query(q)` on a
    reasoner instance."""
    out = []
    cog_like = getattr(self, "store", None)
    for q in queries:
        text = q if isinstance(q, str) else q.get("q") or q.get("query")
        activations = self.builder.encoder.encode([text])[0]
        # Pick top-k anchor activations
        anchors = {
            name: float(activations[i])
            for i, name in enumerate(self.builder.encoder.anchor_names)
        }
        anchors_sorted = sorted(anchors.items(), key=lambda x: -x[1])[:10]
        out.append({"query": text, "anchors": dict(anchors_sorted)})
    return out


def patch_estimator_api(cls):
    """Attach predict / predict_mass / score / transform to the given
    class. Called once at import time on HypergraphReasoner."""
    cls.predict = reasoner_predict
    cls.predict_mass = reasoner_predict_mass
    cls.score = reasoner_score
    cls.transform = reasoner_transform
    return cls
