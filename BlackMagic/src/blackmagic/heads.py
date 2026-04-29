"""Multi-task classification heads on the shared SPLADE backbone.

The bundled splade-tiny (2-layer BERT, hidden=128) already encodes every
sentence during extraction. We add lightweight linear heads on the [CLS]
pooled representation for tasks that are currently handled by heuristics:

Heads:
  1. NLI head (3-class): entailment / neutral / contradiction
     - Replaces hand-crafted polarity + alignment heuristics in DS
     - Input: [CLS](premise) ; [CLS](hypothesis) → 256-dim → 3 logits
     - Directly produces Dempster-Shafer mass functions

  2. Relevance head (2-class): relevant / irrelevant
     - Filters infons that don't pertain to the claim
     - Input: [CLS](claim) ; [CLS](infon_sentence) → 256-dim → 2 logits
     - Fixes evidence precision (currently 77%)

  3. Polarity head (3-class): affirmed / negated / uncertain
     - Replaces regex-only negation detection
     - Input: [CLS](sentence) → 128-dim → 3 logits
     - Single sentence classification (no pair needed)

  4. Relation type head (5-class): causal / temporal / spatial / attributive / none
     - Improves triple extraction quality for the predicate role
     - Input: [CLS](sentence) → 128-dim → 5 logits
     - Single sentence classification

All heads share the same frozen backbone. Training requires only a small
synthetic dataset (generated once, used for all heads). Total added params:
  NLI:       512*128 + 128*64 + 64*3 = 73,987
  Relevance: 512*64 + 64*2 = 32,898
  Polarity:  128*32 + 32*3 = 4,192
  RelType:   128*32 + 32*5 = 4,256
  TOTAL:     ~115K params (460KB)

The backbone stays frozen (17MB). Total model: ~17.5MB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .dempster_shafer import MassFunction

# ═══════════════════════════════════════════════════════════════════════
# HEAD ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════

class NLIHead(nn.Module):
    """Sentence-pair NLI: (premise_cls, hypothesis_cls) → {entail, neutral, contradict}.

    Input: [CLS] embeddings from premise and hypothesis.
    Features: [p; h; |p-h|; p*h] — classic InferSent decomposition (4*hidden_dim).
    Output: 3-class logits → softmax → DS mass function.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        # InferSent features: concat + diff + product = 4*hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
        )

    def forward(self, cls_premise: torch.Tensor, cls_hypothesis: torch.Tensor) -> torch.Tensor:
        """Returns logits (batch, 3): [entailment, neutral, contradiction]."""
        diff = torch.abs(cls_premise - cls_hypothesis)
        prod = cls_premise * cls_hypothesis
        combined = torch.cat([cls_premise, cls_hypothesis, diff, prod], dim=-1)
        return self.net(combined)

    def predict_mass(self, cls_premise: torch.Tensor, cls_hypothesis: torch.Tensor) -> list[MassFunction]:
        """Predict DS mass functions from CLS embeddings.

        Maps softmax probabilities to DS masses with a calibration step:
        - High-confidence predictions (max prob > 0.6) get low theta
        - Low-confidence predictions get more mass redistributed to theta (ignorance)
        This prevents overconfident neutral predictions from washing out signal.
        """
        logits = self.forward(cls_premise, cls_hypothesis)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        masses = []
        for p in probs:
            max_prob = float(max(p))
            # Calibration: reserve theta proportional to uncertainty
            # If max_prob is low (uncertain), give more to theta
            theta = max(0.0, 0.5 * (1.0 - max_prob))
            scale = 1.0 - theta
            masses.append(MassFunction(
                supports=float(p[0]) * scale,
                uncertain=float(p[1]) * scale,
                refutes=float(p[2]) * scale,
                theta=theta,
            ))
        return masses


class RelevanceHead(nn.Module):
    """Sentence-pair relevance: (claim_cls, evidence_cls) → {relevant, irrelevant}.

    Filters infons that don't pertain to the claim before DS scoring.
    Features: [claim; evidence; |claim-evidence|; claim*evidence] (4*hidden_dim).
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, cls_claim: torch.Tensor, cls_evidence: torch.Tensor) -> torch.Tensor:
        """Returns logits (batch, 2): [relevant, irrelevant]."""
        diff = torch.abs(cls_claim - cls_evidence)
        prod = cls_claim * cls_evidence
        combined = torch.cat([cls_claim, cls_evidence, diff, prod], dim=-1)
        return self.net(combined)

    def predict_relevant(self, cls_claim: torch.Tensor, cls_evidence: torch.Tensor,
                         threshold: float = 0.5) -> np.ndarray:
        """Returns boolean mask of relevant evidence."""
        logits = self.forward(cls_claim, cls_evidence)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        return probs[:, 0] > threshold  # column 0 = relevant


class PolarityHead(nn.Module):
    """Single-sentence polarity: sentence_cls → {affirmed, negated, uncertain}.

    Replaces regex-based negation detection.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 3),
        )

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        """Returns logits (batch, 3): [affirmed, negated, uncertain]."""
        return self.net(cls)

    def predict_polarity(self, cls: torch.Tensor) -> list[int]:
        """Returns polarity: 1=affirmed, 0=negated, -1=uncertain."""
        logits = self.forward(cls)
        preds = logits.argmax(dim=-1).cpu().tolist()
        # Map: 0→1(affirmed), 1→0(negated), 2→-1(uncertain)
        mapping = {0: 1, 1: 0, 2: -1}
        return [mapping[p] for p in preds]


class RelationTypeHead(nn.Module):
    """Single-sentence relation type: sentence_cls → {causal, temporal, spatial, attributive, none}.

    Improves predicate role classification in triple extraction.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 5),
        )
        self.labels = ["causal", "temporal", "spatial", "attributive", "none"]

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        """Returns logits (batch, 5)."""
        return self.net(cls)

    def predict_type(self, cls: torch.Tensor) -> list[str]:
        """Returns relation type labels."""
        logits = self.forward(cls)
        preds = logits.argmax(dim=-1).cpu().tolist()
        return [self.labels[p] for p in preds]


# ═══════════════════════════════════════════════════════════════════════
# MULTI-HEAD WRAPPER
# ═══════════════════════════════════════════════════════════════════════

class CognitionHeads(nn.Module):
    """All classification heads, sharing the SPLADE backbone.

    Usage:
        from cognition.encoder import SpladeEncoder
        encoder = SpladeEncoder()  # loads backbone
        heads = CognitionHeads.load("path/to/heads/")

        # Get CLS from backbone
        cls = heads.encode_cls(encoder, ["sentence 1", "sentence 2"])

        # Use individual heads
        masses = heads.nli.predict_mass(cls_premise, cls_hypothesis)
        relevant = heads.relevance.predict_relevant(cls_claim, cls_evidence)
        polarities = heads.polarity.predict_polarity(cls)
        rel_types = heads.relation_type.predict_type(cls)
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nli = NLIHead(hidden_dim)
        self.relevance = RelevanceHead(hidden_dim)
        self.polarity = PolarityHead(hidden_dim)
        self.relation_type = RelationTypeHead(hidden_dim)

    @torch.no_grad()
    def encode_cls(self, encoder, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        """Get [CLS] pooled embeddings from the SPLADE backbone.

        Reuses the encoder's model (no extra forward pass needed if we
        hook into the hidden states during SPLADE encoding).
        """
        all_cls = []
        device = encoder.device

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = encoder.tokenizer(
                batch, max_length=encoder.max_length, padding=True,
                truncation=True, return_tensors="pt",
                return_token_type_ids=False,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # base_model works for any AutoModelForMaskedLM backbone
            # (BERT, DistilBERT, XLM-R).
            outputs = encoder.model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # [CLS] is position 0
            cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_dim)
            all_cls.append(cls_emb.cpu())

        return torch.cat(all_cls, dim=0)

    def save(self, path: str | Path):
        """Save all head weights (just the heads, not the backbone)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "nli": self.nli.state_dict(),
            "relevance": self.relevance.state_dict(),
            "polarity": self.polarity.state_dict(),
            "relation_type": self.relation_type.state_dict(),
            "hidden_dim": self.hidden_dim,
        }, path / "heads.pt")

    @classmethod
    def load(cls, path: str | Path) -> CognitionHeads:
        """Load head weights from disk."""
        path = Path(path)
        checkpoint = torch.load(path / "heads.pt", map_location="cpu", weights_only=True)
        hidden_dim = checkpoint.get("hidden_dim", 128)
        heads = cls(hidden_dim=hidden_dim)
        heads.nli.load_state_dict(checkpoint["nli"])
        heads.relevance.load_state_dict(checkpoint["relevance"])
        heads.polarity.load_state_dict(checkpoint["polarity"])
        heads.relation_type.load_state_dict(checkpoint["relation_type"])
        heads.eval()
        return heads

    def param_count(self) -> dict:
        """Count parameters per head."""
        return {
            "nli": sum(p.numel() for p in self.nli.parameters()),
            "relevance": sum(p.numel() for p in self.relevance.parameters()),
            "polarity": sum(p.numel() for p in self.polarity.parameters()),
            "relation_type": sum(p.numel() for p in self.relation_type.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


# ═══════════════════════════════════════════════════════════════════════
# TRAINING DATA SCHEMA
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrainingSample:
    """A single multi-task training sample.

    One sentence (or sentence pair) annotated for all applicable heads.
    Generated synthetically from templates + FEVER/HoVer gold data.
    """
    # Text
    premise: str = ""          # evidence sentence
    hypothesis: str = ""       # claim (empty for single-sentence tasks)

    # Labels (None = not applicable for this sample)
    nli_label: int | None = None        # 0=entail, 1=neutral, 2=contradict
    relevance_label: int | None = None  # 0=relevant, 1=irrelevant
    polarity_label: int | None = None   # 0=affirmed, 1=negated, 2=uncertain
    relation_type_label: int | None = None  # 0-4 per RelationTypeHead.labels


def generate_training_data(fever_claims: list[dict], n_samples: int = 5000) -> list[TrainingSample]:
    """Generate multi-task training data from FEVER gold evidence.

    Strategy: each FEVER claim + evidence pair gives us labels for ALL heads:
      - NLI: SUPPORTS→entail, REFUTES→contradict, NEI→neutral
      - Relevance: gold evidence→relevant, random other evidence→irrelevant
      - Polarity: from the evidence sentence itself (negation cues)
      - Relation type: from sentence structure (heuristic seed, cleaned by NLI)

    This produces ONE dataset that trains ALL heads simultaneously.
    """
    import random
    random.seed(42)
    samples = []

    # Collect all evidence sentences for negative sampling
    all_evidence = []
    for claim in fever_claims:
        for ev in claim.get("evidence", []):
            if len(ev) >= 3:
                all_evidence.append(ev[2])

    for claim in fever_claims:
        claim_text = claim["claim"]
        label = claim.get("label", "")
        evidence_list = claim.get("evidence", [])

        # NLI label
        nli_map = {"SUPPORTS": 0, "REFUTES": 2, "NOT ENOUGH INFO": 1}
        nli_label = nli_map.get(label)

        for ev in evidence_list:
            if len(ev) < 3:
                continue
            ev_text = ev[2]

            # Positive sample: claim + its gold evidence
            polarity = _detect_polarity_label(ev_text)
            rel_type = _detect_relation_type(ev_text)

            samples.append(TrainingSample(
                premise=ev_text,
                hypothesis=claim_text,
                nli_label=nli_label,
                relevance_label=0,  # gold evidence is relevant
                polarity_label=polarity,
                relation_type_label=rel_type,
            ))

            # Negative sample: claim + random irrelevant evidence
            if all_evidence:
                neg_ev = random.choice(all_evidence)
                samples.append(TrainingSample(
                    premise=neg_ev,
                    hypothesis=claim_text,
                    nli_label=1,  # random evidence is neutral
                    relevance_label=1,  # irrelevant
                    polarity_label=_detect_polarity_label(neg_ev),
                    relation_type_label=_detect_relation_type(neg_ev),
                ))

        if len(samples) >= n_samples:
            break

    return samples[:n_samples]


def _detect_polarity_label(text: str) -> int:
    """Heuristic polarity label for training data seed."""
    import re
    neg_pattern = re.compile(
        r'\b(not|no|never|neither|nor|cannot|can\'t|won\'t|doesn\'t|didn\'t|'
        r'hasn\'t|haven\'t|isn\'t|aren\'t|wasn\'t|weren\'t|hardly|barely|'
        r'seldom|rarely|unlikely|impossible|failed|denied|refused)\b',
        re.IGNORECASE,
    )
    if neg_pattern.search(text):
        return 1  # negated
    return 0  # affirmed


def _detect_relation_type(text: str) -> int:
    """Heuristic relation type for training data seed."""
    text_lower = text.lower()

    # Causal
    if any(w in text_lower for w in ["because", "caused", "led to", "resulted", "therefore", "due to"]):
        return 0
    # Temporal
    if any(w in text_lower for w in ["before", "after", "during", "when", "while", "until", "since"]):
        return 1
    # Spatial
    if any(w in text_lower for w in ["located", "in", "at", "near", "between", "from", "to"]):
        return 2
    # Attributive (is/was/has properties)
    if any(w in text_lower for w in [" is ", " was ", " are ", " were ", " has ", " had "]):
        return 3
    # None
    return 4
