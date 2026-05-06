"""Multi-task sentence embedder trained on the synthetic corpus.

Takes a SPLADE sparse vector (30,522-d) for a sentence and produces:
    - 64-d node embedding (for use as GNN input feature)
    - per-anchor activation logits (which schema anchors are present)
    - DS mass (4-d)  — sentence-level belief
    - evidentiality (3-class)
    - modality (3-class)

Trained jointly on all signals; the shared trunk produces a unified
representation that downstream heads read.

SPLADE itself stays frozen. Only this module's parameters train.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceEmbedder(nn.Module):
    """Shared-trunk multi-task embedder over frozen SPLADE output."""

    # Four role slots for the anchor-role head: O (no role), NSUBJ, VERB, OBJ.
    # Collapses POBJ+DOBJ and DOBJ into OBJ for a cleaner sentence-level target.
    N_ROLE_SLOTS = 4
    ROLE_O, ROLE_NSUBJ, ROLE_VERB, ROLE_OBJ = range(4)
    ROLE_SLOT_LABELS = ["O", "NSUBJ", "VERB", "OBJ"]

    def __init__(self,
                 sparse_dim: int = 30522,
                 n_anchors: int = 22,
                 node_dim: int = 64,
                 trunk_dim: int = 256,
                 n_evid: int = 3,
                 n_modl: int = 3):
        super().__init__()
        self.sparse_dim = sparse_dim
        self.n_anchors = n_anchors
        self.node_dim = node_dim
        # Shared non-linear trunk: sparse → dense
        self.trunk = nn.Sequential(
            nn.Linear(sparse_dim, trunk_dim),
            nn.ReLU(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.ReLU(),
        )
        # Heads
        self.node_head = nn.Linear(trunk_dim, node_dim)
        self.anchor_head = nn.Linear(trunk_dim, n_anchors)
        # Role-aware anchor head: for each anchor, predict its role slot.
        # Output shape: (batch, n_anchors * N_ROLE_SLOTS). Reshape at
        # forward time to (batch, n_anchors, N_ROLE_SLOTS).
        self.anchor_role_head = nn.Linear(
            trunk_dim, n_anchors * self.N_ROLE_SLOTS,
        )
        self.mass_head = nn.Linear(trunk_dim, 4)
        self.evid_head = nn.Linear(trunk_dim, n_evid)
        self.modl_head = nn.Linear(trunk_dim, n_modl)

    def forward(self, sparse: torch.Tensor) -> dict:
        """sparse: (batch, sparse_dim)  →  dict with node/anchor/mass/..."""
        h = self.trunk(sparse)
        node = F.normalize(self.node_head(h), dim=-1)
        role_raw = self.anchor_role_head(h)
        role_logits = role_raw.view(
            -1, self.n_anchors, self.N_ROLE_SLOTS,
        )
        return {
            "trunk": h,
            "node": node,
            "anchor_logits": self.anchor_head(h),
            "role_logits": role_logits,              # (B, n_anchors, 4)
            "mass": F.softmax(self.mass_head(h), dim=-1),
            "evid_logits": self.evid_head(h),
            "modl_logits": self.modl_head(h),
        }

    def node_embedding(self, sparse: torch.Tensor) -> torch.Tensor:
        """Inference-only path for the GNN builder: just the 64-d node feature."""
        with torch.no_grad():
            h = self.trunk(sparse)
            return F.normalize(self.node_head(h), dim=-1)


# ── Training utilities ────────────────────────────────────────────

EVID_LABELS = ["assertion", "hedge", "attribution"]
MODL_LABELS = ["certain", "possible", "obligation"]


def _anchor_multihot(example, anchor_names: list[str]) -> torch.Tensor:
    """Multi-hot target over schema anchors: which anchors are
    activated by this sentence's gold labels."""
    idx_by_name = {n: i for i, n in enumerate(anchor_names)}
    vec = torch.zeros(len(anchor_names))
    for anchor in example.token_anchors:
        if anchor and anchor in idx_by_name:
            vec[idx_by_name[anchor]] = 1.0
    return vec


# Collapse the generator's 6-class token roles to the embedder's
# 4 sentence-level role slots.
from .synth import (
    ROLE_O as _SYNTH_O,
    ROLE_NSUBJ as _SYNTH_NSUBJ,
    ROLE_VERB as _SYNTH_VERB,
    ROLE_DOBJ as _SYNTH_DOBJ,
    ROLE_POBJ as _SYNTH_POBJ,
    ROLE_MARKER as _SYNTH_MARKER,
)
_TOKEN_ROLE_TO_SLOT = {
    _SYNTH_O: SentenceEmbedder.ROLE_O,
    _SYNTH_NSUBJ: SentenceEmbedder.ROLE_NSUBJ,
    _SYNTH_VERB: SentenceEmbedder.ROLE_VERB,
    _SYNTH_DOBJ: SentenceEmbedder.ROLE_OBJ,
    _SYNTH_POBJ: SentenceEmbedder.ROLE_OBJ,
    _SYNTH_MARKER: SentenceEmbedder.ROLE_O,
}


def _role_targets(example, anchor_names: list[str]) -> torch.Tensor:
    """Per-anchor role-slot target for one example: shape (n_anchors,).

    For each anchor in the schema, the target role-slot is:
        - O   if the anchor doesn't appear in this sentence
        - NSUBJ/VERB/OBJ if it does, per the gold token-role label
    When an anchor appears in multiple positions (rare), we keep the
    highest-priority slot with order NSUBJ > VERB > OBJ > O.
    """
    idx_by_name = {n: i for i, n in enumerate(anchor_names)}
    slots = torch.full((len(anchor_names),),
                       SentenceEmbedder.ROLE_O, dtype=torch.long)
    priority = {
        SentenceEmbedder.ROLE_NSUBJ: 3,
        SentenceEmbedder.ROLE_VERB: 2,
        SentenceEmbedder.ROLE_OBJ: 1,
        SentenceEmbedder.ROLE_O: 0,
    }
    current_priority = [0] * len(anchor_names)
    for role, anchor in zip(example.token_roles, example.token_anchors):
        if not anchor or anchor not in idx_by_name:
            continue
        slot = _TOKEN_ROLE_TO_SLOT.get(role, SentenceEmbedder.ROLE_O)
        ai = idx_by_name[anchor]
        if priority[slot] > current_priority[ai]:
            slots[ai] = slot
            current_priority[ai] = priority[slot]
    return slots


def train_embedder(
    examples: list,
    splade_encoder,
    anchor_names: list[str],
    node_dim: int = 64,
    trunk_dim: int = 256,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    sparse_dim: int = 30522,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Train the multi-task embedder on a synthetic corpus.

    Encodes each sentence through frozen SPLADE once at the start
    (cached), then iterates the embedder against the per-example
    gold labels.

    Returns the trained embedder + stats dict.
    """
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)

    embedder = SentenceEmbedder(
        sparse_dim=sparse_dim,
        n_anchors=len(anchor_names),
        node_dim=node_dim,
        trunk_dim=trunk_dim,
    )
    optimizer = torch.optim.Adam(embedder.parameters(), lr=lr)

    # Encode all sentences through SPLADE once; cache as a dense tensor.
    # For a 5k corpus at 30522 dims this is 5k × 30522 × 4 bytes ≈ 600 MB.
    # Use float16 to halve it.
    if verbose:
        print(f"  [embedder-train] encoding {len(examples)} sentences "
              f"through SPLADE...")
    sentences = [ex.sentence for ex in examples]
    sparse_np = splade_encoder.encode_sparse(sentences, batch_size=32)
    sparse = torch.from_numpy(sparse_np).float()

    # Gold targets
    anchor_targets = torch.stack([
        _anchor_multihot(ex, anchor_names) for ex in examples
    ])
    role_targets = torch.stack([
        _role_targets(ex, anchor_names) for ex in examples
    ])   # (n_examples, n_anchors) long
    mass_targets = torch.tensor(
        [ex.ds_mass for ex in examples], dtype=torch.float32,
    )
    evid_targets = torch.tensor(
        [EVID_LABELS.index(ex.evidentiality) for ex in examples],
        dtype=torch.long,
    )
    modl_targets = torch.tensor(
        [MODL_LABELS.index(ex.modality) for ex in examples],
        dtype=torch.long,
    )
    template_ids = [ex.template_id for ex in examples]

    # Pair up contrastive positives/negatives: same-template → positive
    rng = torch.Generator().manual_seed(seed)

    losses = []
    embedder.train()
    n = len(examples)
    indices = torch.arange(n)
    for epoch in range(epochs):
        perm = indices[torch.randperm(n, generator=rng)]
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            batch_idx = perm[i:i + batch_size]
            if len(batch_idx) < 2:
                continue
            batch_sparse = sparse[batch_idx]
            out = embedder(batch_sparse)

            # (a) anchor multi-label BCE
            L_anchor = F.binary_cross_entropy_with_logits(
                out["anchor_logits"], anchor_targets[batch_idx],
            )
            # (a') role-slot CE per anchor (most anchors will be O)
            #     role_logits: (B, n_anchors, 4)
            #     role_targets[batch_idx]: (B, n_anchors)
            role_logits = out["role_logits"]
            B, A, _ = role_logits.shape
            L_role = F.cross_entropy(
                role_logits.reshape(B * A, -1),
                role_targets[batch_idx].reshape(B * A),
            )
            # (b) mass KL
            pred_mass = out["mass"].clamp(min=1e-8)
            tgt_mass = mass_targets[batch_idx].clamp(min=1e-8)
            tgt_mass = tgt_mass / tgt_mass.sum(-1, keepdim=True)
            L_mass = F.kl_div(
                pred_mass.log(), tgt_mass,
                reduction="batchmean", log_target=False,
            )
            # (c) evidentiality CE
            L_evid = F.cross_entropy(
                out["evid_logits"], evid_targets[batch_idx],
            )
            # (d) modality CE
            L_modl = F.cross_entropy(
                out["modl_logits"], modl_targets[batch_idx],
            )
            # (e) contrastive node embedding: within the batch, same
            # template = positive, different = negative. Use InfoNCE:
            # sim(a, p) - logsumexp(sim(a, all_other)) for each anchor.
            node_vecs = out["node"]
            sim = node_vecs @ node_vecs.T / 0.1    # temperature 0.1
            # Build label matrix: 1 where templates match, 0 otherwise
            batch_tids = [template_ids[int(j)] for j in batch_idx]
            # Same-template mask (excluding self on diagonal)
            L_contrast = torch.zeros(1, device=node_vecs.device).squeeze()
            for idx in range(len(batch_idx)):
                positives = [k for k in range(len(batch_idx))
                             if k != idx and batch_tids[k] == batch_tids[idx]]
                if not positives:
                    continue
                # Positive logits are sim[idx, positives]; negatives are all others
                pos_sim = sim[idx, positives]
                all_sim = sim[idx].clone()
                all_sim[idx] = -1e4  # mask self
                log_denom = torch.logsumexp(all_sim, dim=0)
                # Average over positives
                contrib = (pos_sim - log_denom).mean()
                L_contrast = L_contrast + (-contrib)
            L_contrast = L_contrast / max(len(batch_idx), 1)

            loss = (L_anchor + L_role + L_mass + 0.5 * L_evid +
                    0.5 * L_modl + 0.2 * L_contrast)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  [embedder-train] epoch {epoch+1}/{epochs}: "
                  f"loss={avg_loss:.4f}")

    embedder.eval()

    # Evaluate training-set accuracy for diagnostics
    with torch.no_grad():
        out = embedder(sparse)
        anchor_preds = (out["anchor_logits"] > 0).float()
        anchor_acc = (anchor_preds == anchor_targets).float().mean().item()
        role_preds = out["role_logits"].argmax(-1)
        # Full-example role accuracy (including 'O' which dominates)
        role_acc = (role_preds == role_targets).float().mean().item()
        # Non-O role accuracy — the hard cases
        non_O_mask = role_targets != SentenceEmbedder.ROLE_O
        if non_O_mask.any():
            role_non_O_acc = (
                (role_preds[non_O_mask] == role_targets[non_O_mask])
                .float().mean().item()
            )
        else:
            role_non_O_acc = 0.0
        evid_preds = out["evid_logits"].argmax(-1)
        evid_acc = (evid_preds == evid_targets).float().mean().item()
        modl_preds = out["modl_logits"].argmax(-1)
        modl_acc = (modl_preds == modl_targets).float().mean().item()
        mass_kl = F.kl_div(
            out["mass"].clamp(min=1e-8).log(),
            mass_targets.clamp(min=1e-8) / mass_targets.sum(-1, keepdim=True),
            reduction="batchmean", log_target=False,
        ).item()

    return {
        "embedder": embedder,
        "losses": losses,
        "final_loss": losses[-1] if losses else float("nan"),
        "n_examples": len(examples),
        "anchor_accuracy": float(anchor_acc),
        "role_accuracy": float(role_acc),
        "role_non_O_accuracy": float(role_non_O_acc),
        "evidentiality_accuracy": float(evid_acc),
        "modality_accuracy": float(modl_acc),
        "mass_kl": float(mass_kl),
    }


def schema_fingerprint(anchor_names: list[str],
                       types: dict[str, str]) -> str:
    """Stable hash of the (sorted anchor names + their types).

    Any change in the anchor set or their types produces a different
    fingerprint. Same schema under any dict ordering produces the same
    fingerprint. Used to decide whether a cached embedder is still valid.
    """
    import hashlib
    parts = []
    for name in sorted(anchor_names):
        parts.append(f"{name}:{types.get(name, 'unknown')}")
    blob = "|".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def save_embedder(embedder: SentenceEmbedder, path: str | Path,
                  fingerprint: str = "",
                  anchor_names: list[str] | None = None,
                  trunk_dim: int = 256) -> None:
    """Save the embedder's state dict + config + schema fingerprint."""
    state = {
        "state_dict": embedder.state_dict(),
        "config": {
            "sparse_dim": embedder.sparse_dim,
            "n_anchors": embedder.n_anchors,
            "node_dim": embedder.node_dim,
            "trunk_dim": trunk_dim,
            "fingerprint": fingerprint,
            "anchor_names": anchor_names or [],
        },
    }
    torch.save(state, path)


def load_embedder(path: str | Path,
                  trunk_dim: int | None = None,
                  n_evid: int = 3,
                  n_modl: int = 3) -> SentenceEmbedder:
    """Load a saved embedder. trunk_dim is inferred from the save
    when available, or defaults to 256 for older files."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    effective_trunk = trunk_dim or cfg.get("trunk_dim", 256)
    emb = SentenceEmbedder(
        sparse_dim=cfg["sparse_dim"],
        n_anchors=cfg["n_anchors"],
        node_dim=cfg["node_dim"],
        trunk_dim=effective_trunk,
        n_evid=n_evid,
        n_modl=n_modl,
    )
    emb.load_state_dict(state["state_dict"])
    emb.eval()
    return emb


def load_embedder_metadata(path: str | Path) -> dict:
    """Read a saved embedder's config dict without instantiating the
    module. Used to check fingerprint before deciding to load/retrain."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state.get("config", {})


def extract_triples_via_head(
    sentence: str,
    embedder: SentenceEmbedder,
    splade_encoder,
    schema,
    anchor_activation_threshold: float = 0.5,
    role_confidence_threshold: float = 0.5,
) -> list[dict]:
    """Use the trained SentenceEmbedder to extract triples from a sentence.

    Returns a list of triples (usually 0 or 1 per sentence) with:
        subject, predicate, object   — schema anchor names
        subject_confidence, predicate_confidence, object_confidence
        mass (S, R, U, theta)
        evidentiality, modality

    The head tells us:
      - which anchors this sentence activates (anchor_logits)
      - what role each active anchor plays (role_logits)
      - the belief mass of the sentence (mass)

    We combine these into a single triple when the head confidently
    assigns anchors to NSUBJ + VERB + OBJ. If any role is unfilled
    we return an empty list — the caller can fall back.
    """
    import numpy as np
    import torch

    anchor_names = list(schema.names)
    sparse_np = splade_encoder.encode_sparse([sentence], batch_size=1)
    sparse_t = torch.from_numpy(sparse_np).float()

    with torch.no_grad():
        out = embedder(sparse_t)

    # (batch=1, n_anchors) anchor probabilities
    anchor_probs = torch.sigmoid(out["anchor_logits"])[0].cpu().numpy()
    # (batch=1, n_anchors, n_role_slots) role probabilities
    role_probs = torch.softmax(out["role_logits"], dim=-1)[0].cpu().numpy()
    mass = out["mass"][0].cpu().numpy().tolist()
    evid_idx = int(out["evid_logits"][0].argmax().item())
    modl_idx = int(out["modl_logits"][0].argmax().item())

    # For each role slot, pick the most confident anchor that is
    # (a) activated (anchor_prob > anchor_activation_threshold)
    # (b) predicted into that role with high probability
    # Note: anchor_names and embedder.n_anchors must align. The
    # embedder's n_anchors was set when it was trained; the schema
    # may have grown since. Use the smaller of the two.
    effective_n = min(embedder.n_anchors, len(anchor_names))

    def pick_role_anchor(role_idx: int,
                          exclude: set[int] | None = None) -> tuple:
        """Return (anchor_idx, combined_score) or (None, 0)."""
        exclude = exclude or set()
        best_idx = -1
        best_score = 0.0
        for ai in range(effective_n):
            if ai in exclude:
                continue
            ap = float(anchor_probs[ai])
            rp = float(role_probs[ai, role_idx])
            combined = ap * rp
            if ap > anchor_activation_threshold and \
                    rp > role_confidence_threshold and \
                    combined > best_score:
                best_score = combined
                best_idx = ai
        return best_idx, best_score

    subj_idx, subj_conf = pick_role_anchor(SentenceEmbedder.ROLE_NSUBJ)
    verb_idx, verb_conf = pick_role_anchor(
        SentenceEmbedder.ROLE_VERB,
        exclude={subj_idx} if subj_idx >= 0 else set(),
    )

    # Type-aware OBJ selection: for certain predicates (partnership-like
    # relations) an actor-object is preferred over a market-object. For
    # production/investment-like predicates, a feature is preferred over
    # a market.
    #
    # Resolution strategy: collect ALL role-OBJ candidates above
    # threshold, then pick the highest-scoring one whose type is
    # preferred given the predicate's semantics.
    types = schema.types
    exclude = {i for i in (subj_idx, verb_idx) if i >= 0}

    def collect_obj_candidates():
        """Return list of (ai, combined_score) above thresholds."""
        out = []
        for ai in range(effective_n):
            if ai in exclude:
                continue
            ap = float(anchor_probs[ai])
            rp = float(role_probs[ai, SentenceEmbedder.ROLE_OBJ])
            combined = ap * rp
            if ap > anchor_activation_threshold and \
                    rp > role_confidence_threshold:
                out.append((ai, combined))
        return sorted(out, key=lambda x: -x[1])

    # Preference order per type
    PARTNERSHIP_LIKE = {"partners", "acquires", "merges", "competes"}
    verb_name_raw = anchor_names[verb_idx] if verb_idx >= 0 else ""
    if verb_name_raw in PARTNERSHIP_LIKE:
        type_preference = ["actor", "feature", "market"]
    else:
        type_preference = ["feature", "actor", "market"]

    obj_idx = -1
    obj_conf = 0.0
    for preferred_type in type_preference:
        for ai, score in collect_obj_candidates():
            if types.get(anchor_names[ai]) == preferred_type:
                obj_idx = ai
                obj_conf = score
                break
        if obj_idx >= 0:
            break

    if subj_idx < 0 or verb_idx < 0 or obj_idx < 0:
        return []

    # Respect schema type constraints: subject should be actor; verb
    # should be relation; object can be actor/feature/market.
    subj_name = anchor_names[subj_idx]
    verb_name = anchor_names[verb_idx]
    obj_name = anchor_names[obj_idx]

    # Only emit if types line up with the expected triple shape
    if types.get(subj_name) != "actor":
        return []
    if types.get(verb_name) != "relation":
        return []
    if types.get(obj_name) in (None, "relation"):
        return []

    return [{
        "subject": subj_name,
        "predicate": verb_name,
        "object": obj_name,
        "subject_confidence": subj_conf,
        "predicate_confidence": verb_conf,
        "object_confidence": obj_conf,
        "mass": {"supports": mass[0], "refutes": mass[1],
                 "uncertain": mass[2], "theta": mass[3]},
        "evidentiality": EVID_LABELS[evid_idx],
        "modality": MODL_LABELS[modl_idx],
    }]


def get_or_train_embedder(
    schema,
    splade_encoder,
    model_dir: str | Path,
    n_synth_examples: int = 1500,
    trunk_dim: int = 256,
    node_dim: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    verbose: bool = False,
) -> SentenceEmbedder:
    """Return a trained SentenceEmbedder for this schema.

    Checks `model_dir/embedder.pt` for cached weights whose schema
    fingerprint matches the current schema. If cached weights exist and
    match, loads them. Otherwise: generates a synthetic corpus from the
    schema, trains from scratch, and saves. Either way returns an eval-
    mode embedder ready for inference.

    `schema` can be either a cognition.schema.AnchorSchema or a
    cognition.synth.Schema.
    """
    from pathlib import Path as _Path
    from .synth import Schema as SynthSchema, generate_corpus

    # Normalize schema → synth-Schema
    if isinstance(schema, SynthSchema):
        synth_schema = schema
        # Need anchor_names + types for the fingerprint
        anchor_names = synth_schema.all_anchor_names()
        types = synth_schema.types()
    else:
        # Assume it's an AnchorSchema-like object with .names and .types
        anchor_names = list(schema.names)
        types = dict(schema.types)
        synth_schema = SynthSchema.from_anchor_schema(schema)

    fp = schema_fingerprint(anchor_names, types)

    path = _Path(model_dir) / "embedder.pt"
    if path.exists():
        try:
            meta = load_embedder_metadata(path)
            if meta.get("fingerprint") == fp:
                if verbose:
                    print(f"  [embedder-cache] loading {path} "
                          f"(fingerprint {fp})")
                return load_embedder(path)
        except Exception as e:
            if verbose:
                print(f"  [embedder-cache] cache at {path} unreadable: {e}")

    # Cache miss → train fresh
    if verbose:
        print(f"  [embedder-cache] no valid cache at {path}; "
              f"training fresh embedder on {n_synth_examples} synth "
              f"examples (fingerprint {fp})")

    examples = generate_corpus(schema=synth_schema, n=n_synth_examples,
                                seed=42)
    stats = train_embedder(
        examples=examples,
        splade_encoder=splade_encoder,
        anchor_names=anchor_names,
        node_dim=node_dim,
        trunk_dim=trunk_dim,
        epochs=epochs,
        lr=lr,
        verbose=verbose,
    )
    embedder = stats["embedder"]

    _Path(model_dir).mkdir(parents=True, exist_ok=True)
    save_embedder(embedder, path, fingerprint=fp,
                   anchor_names=anchor_names, trunk_dim=trunk_dim)
    if verbose:
        print(f"  [embedder-cache] saved trained embedder to {path}")
    return embedder
