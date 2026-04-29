"""Encoder: SPLADE sparse expansion + AnchorProjector (English / WordPiece only).

Uses splade-tiny (4.4M params, 17MB) bundled with the package to produce
30,522-dim sparse vocabulary vectors via log(1 + ReLU(MLM_logits)) +
max-pooling. AnchorProjector maps vocab activations to typed anchor
scores via token-ID max-pooling from the schema.

    sentence → SPLADE → 30k sparse → AnchorProjector → N anchor activations

BlackMagic assumes English input and a BERT WordPiece tokenizer. Multilingual
variants (XLM-R / SentencePiece) live in the `cognition` package.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Bundled splade-tiny model (17MB, shipped with the package)
_BUNDLED_MODEL = str(Path(__file__).parent / "model")
_DEFAULT_MODEL = _BUNDLED_MODEL


class SpladeEncoder:
    """SPLADE sparse encoder: text → 30,522-dim sparse vocabulary vector.

    Uses BertForMaskedLM with log(1 + ReLU(logits)) activation and
    max-pooling across token positions.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL,
                 max_length: int = 256, device: str | None = None):
        self.model_name = model_name or _DEFAULT_MODEL
        self.max_length = max_length
        self.device = device or (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)
        self.vocab_size = self.model.config.vocab_size

    @torch.no_grad()
    def encode_sparse(self, texts: list[str],
                      batch_size: int = 32) -> np.ndarray:
        """Encode texts → (n_texts, vocab_size) sparse activation matrix."""
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch, max_length=self.max_length, padding=True,
                truncation=True, return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits  # (batch, seq_len, vocab_size)

            # SPLADE activation: log(1 + ReLU(logits))
            activated = torch.log1p(torch.relu(logits))

            # Max-pool across token positions, masked
            activated = activated * attention_mask.unsqueeze(-1)
            sparse_vecs = activated.max(dim=1).values  # (batch, vocab_size)

            all_vecs.append(sparse_vecs.cpu().numpy())

        return np.concatenate(all_vecs, axis=0)


class AnchorProjector:
    """Project SPLADE sparse vectors onto typed anchor activations.

    For each anchor in the schema, maps its token strings to BERT WordPiece
    vocab IDs. Projection is max-pooling: an anchor's score = max(sparse_vec[token_ids]).
    """

    def __init__(self, schema, tokenizer):
        self.schema = schema
        self.tokenizer = tokenizer
        self.anchor_token_ids: dict[str, list[int]] = {}
        missing = []

        for name, info in schema.anchors.items():
            ids = set()
            for token in info.get("tokens", []):
                # Direct token → ID
                tid = tokenizer.convert_tokens_to_ids(token)
                if tid != tokenizer.unk_token_id:
                    ids.add(tid)
                # WordPiece continuation subword (e.g. "##battery")
                sub_tid = tokenizer.convert_tokens_to_ids(f"##{token}")
                if sub_tid != tokenizer.unk_token_id:
                    ids.add(sub_tid)
                # Fallback: tokenize multi-word anchor and collect word pieces
                if not ids:
                    sub_ids = tokenizer.encode(token, add_special_tokens=False)
                    for sid in sub_ids:
                        if sid != tokenizer.unk_token_id:
                            ids.add(sid)

            self.anchor_token_ids[name] = list(ids)
            if not ids:
                missing.append(name)

        if missing and len(missing) <= 10:
            print(f"  AnchorProjector: no vocab tokens for {missing}")
        elif missing:
            print(f"  AnchorProjector: no vocab tokens for {len(missing)} anchors")

    def project(self, sparse_vec: np.ndarray) -> dict[str, float]:
        """Project one SPLADE vector → {anchor_name: score}."""
        activations = {}
        for name, token_ids in self.anchor_token_ids.items():
            if token_ids:
                score = max(sparse_vec[tid] for tid in token_ids)
                if score > 0:
                    activations[name] = float(score)
        return activations

    def project_batch(self, sparse_matrix: np.ndarray) -> list[dict[str, float]]:
        """Project a batch of SPLADE vectors."""
        return [self.project(sparse_matrix[i])
                for i in range(sparse_matrix.shape[0])]

    def project_to_matrix(self, sparse_matrix: np.ndarray,
                          anchor_names: list[str]) -> np.ndarray:
        """Project SPLADE matrix → (n_texts, n_anchors) dense matrix."""
        n = sparse_matrix.shape[0]
        out = np.zeros((n, len(anchor_names)), dtype=np.float32)

        for i in range(n):
            vec = sparse_matrix[i]
            for j, name in enumerate(anchor_names):
                token_ids = self.anchor_token_ids.get(name, [])
                if token_ids:
                    out[i, j] = max(vec[tid] for tid in token_ids)
        return out


class Encoder:
    """Encode English sentences into anchor activation vectors.

    Uses SPLADE for broad vocabulary coverage, then projects onto a typed
    anchor schema. No training required — just define your schema tokens.

    Quick start:
        encoder = Encoder(schema=my_schema)
        activations = encoder.encode(["Toyota invests in batteries."])
    """

    def __init__(self, schema=None,
                 model_name: str = _DEFAULT_MODEL,
                 anchor_names: list[str] | None = None,
                 anchor_types: dict[str, str] | None = None,
                 max_length: int = 256,
                 device: str | None = None):
        self.model_name = model_name or _DEFAULT_MODEL
        self.max_length = max_length

        # SPLADE encoder
        self.splade = SpladeEncoder(self.model_name, max_length, device)
        self.device = self.splade.device

        # Schema and projector
        self.anchor_names = anchor_names or (list(schema.names) if schema else [])
        self.anchor_types = anchor_types or (schema.types if schema else {})
        self.n_anchors = len(self.anchor_names)

        self._projector = None
        if schema:
            self._projector = AnchorProjector(schema, self.splade.tokenizer)

    @property
    def tokenizer(self):
        return self.splade.tokenizer

    @property
    def model(self):
        return self.splade.model

    def encode(self, texts: list[str],
               batch_size: int = 32) -> np.ndarray:
        """Encode sentences → (n_texts, n_anchors) anchor activation matrix."""
        sparse = self.splade.encode_sparse(texts, batch_size=batch_size)

        if self._projector is None:
            raise RuntimeError(
                "No schema/projector configured. Pass schema= to __init__."
            )

        return self._projector.project_to_matrix(sparse, self.anchor_names)

    def encode_sparse(self, texts: list[str],
                      batch_size: int = 32) -> np.ndarray:
        """Encode sentences → raw (n_texts, vocab_size) SPLADE vectors."""
        return self.splade.encode_sparse(texts, batch_size=batch_size)

    def encode_single(self, text: str) -> dict[str, float]:
        """Encode one sentence → {anchor_name: score}."""
        sparse = self.splade.encode_sparse([text])
        return self._projector.project(sparse[0])

    def find_spans(self, text: str, anchor_name: str,
                   anchor_defs: dict) -> list[dict]:
        """Find character spans where an anchor's tokens appear in text.

        English / ASCII only — uses \\b word boundaries. If you need CJK
        support, use the cognition package.
        """
        info = anchor_defs.get(anchor_name, {})
        tokens = info.get("tokens", [])
        if not tokens:
            return []

        sorted_tokens = sorted(tokens, key=len, reverse=True)
        pattern = re.compile(
            r'\b(' + '|'.join(re.escape(t) for t in sorted_tokens) + r')\b',
            re.IGNORECASE,
        )
        return [
            {"text": m.group(), "start": m.start(), "end": m.end()}
            for m in pattern.finditer(text)
        ]
