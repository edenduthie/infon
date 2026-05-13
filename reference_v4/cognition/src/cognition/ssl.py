"""Self-supervised learning modes for HypergraphReasoner.

Four pretext losses, all drop-in replacements for each other:

    'laplacian'   sheaf-Laplacian edge coherence (the current default)
    'barlow'      Barlow Twins on two corrupted views of the graph
    'vicreg'      VICReg — variance + invariance + covariance, no BN
    'jepa'        Temporal-JEPA with EMA teacher + relation-conditioned
                  predictor; targets live in the EMA teacher's latent
                  space

Each helper is a plain nn.Module or a stateless function — the
HypergraphReasoner picks one via its `fit_ssl(mode=...)` method.

The default is 'laplacian' because it's the one that's held up cleanest
across every experiment and doesn't require an auxiliary network. The
others become available when users ask for them explicitly, and
`HypergraphReasoner.auto_select_ssl()` will fit each and return the
one that scores best on a held-out infon split.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# Heads + predictors
# ═══════════════════════════════════════════════════════════════════════

class BarlowHead(nn.Module):
    """Shared 2-layer projection head with output batchnorm.

    Output is BN'd along the feature axis so the cross-correlation
    matrix for the BT loss is well-defined without extra normalization.

    References
    ----------
    Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021).
        "Barlow Twins: Self-Supervised Learning via Redundancy Reduction."
        ICML 2021. arXiv:2103.03230
    """

    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.bn = nn.BatchNorm1d(proj_dim, affine=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.bn(self.net(h))


class VICRegHead(nn.Module):
    """Projection head for VICReg. Identical shape to BarlowHead minus
    the output BN — VICReg's variance term does the anti-collapse work
    without needing BN.

    References
    ----------
    Bardes, A., Ponce, J., & LeCun, Y. (2022). "VICReg: Variance-
        Invariance-Covariance Regularization for Self-Supervised
        Learning." ICLR 2022. arXiv:2105.04906
    """

    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class RelationPredictor(nn.Module):
    """Per-relation 2-layer MLP predictor for JEPA-style training.

    Given a source-node embedding h_s and a relation id r, predicts the
    target-node embedding in the EMA teacher's latent space.

    References
    ----------
    Assran, M. et al. (2023). "Self-Supervised Learning from Images with
        a Joint-Embedding Predictive Architecture." CVPR 2023 (I-JEPA).
        arXiv:2301.08243
    Bardes, A. et al. (2024). "V-JEPA: Revisiting Feature Prediction for
        Learning Visual Representations from Video." arXiv:2404.08471
    Assran, M. et al. (2025). "V-JEPA 2: Self-Supervised Video Models
        Enable Understanding, Prediction, and Planning."
        (Action-conditioned predictor architecture this module adapts
        for graph-relation conditioning.)
    LeCun, Y. (2022). "A Path Towards Autonomous Machine Intelligence."
        OpenReview. (H-JEPA framework; relation-conditioning here is the
        discrete analogue of V-JEPA 2's action-conditioning.)
    """

    def __init__(self, hidden_dim: int, n_relations: int,
                 proj_dim: int = 128):
        super().__init__()
        self.rel_embed = nn.Embedding(n_relations, hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(2 * hidden_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, hidden_dim),
        )

    def forward(self, h_src: torch.Tensor,
                rel: torch.Tensor) -> torch.Tensor:
        return self.trunk(
            torch.cat([h_src, self.rel_embed(rel)], dim=-1)
        )


# ═══════════════════════════════════════════════════════════════════════
# Losses
# ═══════════════════════════════════════════════════════════════════════

def barlow_twins_loss(z_a: torch.Tensor, z_b: torch.Tensor,
                      lambda_off: float = 5e-3) -> torch.Tensor:
    """Standard BT — invariance on the diagonal + redundancy off-diagonal.

    z_a, z_b are assumed already batch-normalized along feature dim.
    """
    n = z_a.shape[0]
    c = (z_a.T @ z_b) / n
    on_diag = torch.diagonal(c)
    off_diag = c - torch.diag(on_diag)
    return ((1.0 - on_diag) ** 2).sum() + lambda_off * (off_diag ** 2).sum()


def vicreg_loss(z_a: torch.Tensor, z_b: torch.Tensor,
                lambda_inv: float = 25.0,
                lambda_var: float = 25.0,
                lambda_cov: float = 1.0,
                gamma: float = 1.0,
                eps: float = 1e-4) -> torch.Tensor:
    """VICReg — Bardes et al. 2022. Three terms, no negatives, no BN."""
    inv = F.mse_loss(z_a, z_b)

    def _var(z):
        std = torch.sqrt(z.var(dim=0) + eps)
        return F.relu(gamma - std).mean()

    var = 0.5 * (_var(z_a) + _var(z_b))

    def _cov(z):
        n, d = z.shape
        z = z - z.mean(dim=0, keepdim=True)
        c = (z.T @ z) / max(n - 1, 1)
        off = c - torch.diag(torch.diagonal(c))
        return off.pow(2).sum() / d

    cov = 0.5 * (_cov(z_a) + _cov(z_b))
    return lambda_inv * inv + lambda_var * var + lambda_cov * cov


def jepa_loss(pred: torch.Tensor, target: torch.Tensor,
              variance_reg: torch.Tensor | None = None,
              lambda_var: float = 1.0) -> torch.Tensor:
    """JEPA — smooth L2 in teacher embedding space + optional
    VICReg variance term on the student's own features to prevent
    collapse."""
    loss = F.mse_loss(pred, target)
    if variance_reg is not None:
        std = torch.sqrt(variance_reg.var(dim=0) + 1e-4)
        loss = loss + lambda_var * F.relu(1.0 - std).mean()
    return loss


# ═══════════════════════════════════════════════════════════════════════
# Utilities — EMA updates, view corruptions, rank diagnostics
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, tau: float):
    """Polyak average: teacher ← tau·teacher + (1−tau)·student."""
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(tau).add_(ps.data, alpha=1.0 - tau)


def effective_rank(z: torch.Tensor) -> float:
    """Participation ratio of squared singular values — a diagnostic
    for representational collapse. Ranges from 1 (total collapse) to D
    (uniform spread across feature axes)."""
    with torch.no_grad():
        s = torch.linalg.svdvals(z - z.mean(dim=0, keepdim=True))
        s2 = s.pow(2)
        s2 = s2 / (s2.sum() + 1e-12)
        return math.exp(-(s2 * (s2 + 1e-12).log()).sum().item())


def corrupt_edge_drop(edge_index: torch.Tensor,
                      edge_types: torch.Tensor,
                      edge_weights: torch.Tensor,
                      frac: float,
                      rng: random.Random
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly drop a fraction of edges."""
    n = edge_index.shape[1]
    keep = torch.tensor(
        [rng.random() >= frac for _ in range(n)], dtype=torch.bool,
    )
    return edge_index[:, keep], edge_types[keep], edge_weights[keep]


def corrupt_role_mask(edge_index: torch.Tensor,
                      edge_types: torch.Tensor,
                      edge_weights: torch.Tensor,
                      infon_node_indices: set[int],
                      spoke_rel_ids: set[int],
                      rng: random.Random
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each infon, randomly drop one of its three spoke edges
    (INITIATES / ASSERTS / TARGETS)."""
    drop_for = {i: rng.choice(list(spoke_rel_ids))
                for i in infon_node_indices}
    n = edge_index.shape[1]
    keep = torch.ones(n, dtype=torch.bool)
    for e in range(n):
        r = int(edge_types[e])
        if r not in spoke_rel_ids:
            continue
        s = int(edge_index[0, e]); t = int(edge_index[1, e])
        inf = t if t in infon_node_indices else (
            s if s in infon_node_indices else None
        )
        if inf is None:
            continue
        if drop_for.get(inf) == r:
            keep[e] = False
    return edge_index[:, keep], edge_types[keep], edge_weights[keep]


# ═══════════════════════════════════════════════════════════════════════
# Dataclass for per-mode results
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SSLResult:
    """Summary of an SSL pretraining run."""
    mode: str                 # 'laplacian' | 'barlow' | 'vicreg' | 'jepa'
    epochs: int
    final_loss: float
    eff_rank_before: float
    eff_rank_after: float
    heldout_score: float | None = None   # populated by auto_select_ssl
    extras: dict | None = None           # per-mode metrics

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "epochs": self.epochs,
            "final_loss": self.final_loss,
            "eff_rank_before": self.eff_rank_before,
            "eff_rank_after": self.eff_rank_after,
            "heldout_score": self.heldout_score,
            "extras": self.extras or {},
        }


SSL_MODES = ("laplacian", "barlow", "vicreg", "jepa")
DEFAULT_SSL_MODE = "laplacian"
