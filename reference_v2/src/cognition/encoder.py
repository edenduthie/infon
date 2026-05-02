"""Encoder: SPLADE sparse expansion + AnchorProjector.

Uses splade-tiny (4.4M params, 17MB) bundled with the package to produce
30,522-dim sparse vocabulary vectors via log(1 + ReLU(MLM_logits)) +
max-pooling. The AnchorProjector maps vocab activations to typed anchor
scores via token ID max-pooling from the schema.

sentence → SPLADE → 30k sparse → AnchorProjector → N anchor activations
"""

from __future__ import annotations

import json
import re
import warnings
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForMaskedLM


# Bundled splade-tiny model (17MB, shipped with the package)
_BUNDLED_MODEL = str(Path(__file__).parent / "model")
_DEFAULT_MODEL = _BUNDLED_MODEL

# Named model aliases for convenience.
# The English anchor schema stays fixed; multilingual input is projected via
# XLM-R's shared sparse space back onto the English anchors (no schema changes).
MODEL_ALIASES = {
    "splade-tiny": _BUNDLED_MODEL,
    "tiny": _BUNDLED_MODEL,
    "en": _BUNDLED_MODEL,
    "multilingual": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    "xlmr": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
}


def _resolve_model(name: str) -> str:
    if not name:
        return _DEFAULT_MODEL
    return MODEL_ALIASES.get(name, name)


class SpladeEncoder:
    """SPLADE sparse encoder: text → 30,522-dim sparse vocabulary vector.

    Uses BertForMaskedLM with log(1 + ReLU(logits)) activation and
    max-pooling across token positions.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL,
                 max_length: int = 256, device: str | None = None):
        self.model_name = _resolve_model(model_name)
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

        # Detect tokenizer family so AnchorProjector picks the right subword
        # convention. XLM-R/mBERT-sp use SentencePiece (▁ word-initial prefix);
        # BERT-family models use WordPiece (## continuation prefix).
        toks = self.tokenizer.get_vocab() if hasattr(self.tokenizer, "get_vocab") else {}
        self.uses_sentencepiece = any(t.startswith("▁") for t in list(toks)[:2000])

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
                return_token_type_ids=False,  # XLM-R has no segment IDs
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

    For each anchor in the schema, maps its token strings to BERT vocab IDs.
    Projection is max-pooling: the anchor's score = max(sparse_vec[token_ids]).
    """

    def __init__(self, schema, tokenizer):
        self.schema = schema
        self.tokenizer = tokenizer
        self.anchor_token_ids: dict[str, list[int]] = {}
        missing = []

        # Detect tokenizer convention once — SentencePiece (▁) vs WordPiece (##)
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        uses_sp = any(t.startswith("▁") for t in list(vocab)[:2000])
        unk = tokenizer.unk_token_id

        # Common subword noise: single-char pieces, stray affixes, and the
        # SentencePiece/WordPiece space marker itself fire on almost any CJK
        # text. Without this filter, short acronym anchors like "ccg"/"png"
        # collapse onto the same garbage tokens and produce false activations.
        def _is_noisy_subword(tok_str: str) -> bool:
            if not tok_str:
                return True
            # Strip the SentencePiece boundary marker; then assess the payload
            bare = tok_str.lstrip("▁").lstrip("#")
            # Single-character ASCII letter/digit is almost always noise.
            if len(bare) <= 1 and bare.isascii():
                return True
            # Common junk continuation pieces in XLM-R vocab.
            if bare in {"s", "er", "ed", "ing", "ly", "tion"}:
                return True
            return False

        for name, info in schema.anchors.items():
            ids = set()
            for token in info.get("tokens", []):
                # 1) bare token — matches single-piece vocab entries
                tid = tokenizer.convert_tokens_to_ids(token)
                if tid is not None and tid != unk:
                    ids.add(tid)

                # 2) word-initial variant — SentencePiece uses ▁, WordPiece
                # has no prefix on word-initial tokens (## is continuation)
                if uses_sp:
                    sp_tid = tokenizer.convert_tokens_to_ids(f"▁{token}")
                    if sp_tid is not None and sp_tid != unk:
                        ids.add(sp_tid)
                else:
                    sub_tid = tokenizer.convert_tokens_to_ids(f"##{token}")
                    if sub_tid is not None and sub_tid != unk:
                        ids.add(sub_tid)

                # 3) Full tokenization — always run so multi-word anchors
                # (e.g. "china coast guard") contribute their content pieces,
                # but drop noisy single-character subwords. This was the main
                # source of cross-lingual false activations: tiny acronyms
                # like "LNG" tokenize to ['▁L','N','G'] and 'G' collides with
                # hundreds of CJK-adjacent vocab IDs.
                sub_ids = tokenizer.encode(token, add_special_tokens=False)
                for sid in sub_ids:
                    if sid is None or sid == unk:
                        continue
                    piece = tokenizer.convert_ids_to_tokens(sid)
                    if _is_noisy_subword(piece):
                        continue
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
        """Project SPLADE matrix → (n_texts, n_anchors) dense matrix.

        anchor_names defines the column ordering.
        """
        n = sparse_matrix.shape[0]
        out = np.zeros((n, len(anchor_names)), dtype=np.float32)
        name_to_col = {name: j for j, name in enumerate(anchor_names)}

        for i in range(n):
            vec = sparse_matrix[i]
            for name, token_ids in self.anchor_token_ids.items():
                j = name_to_col.get(name)
                if j is not None and token_ids:
                    out[i, j] = max(vec[tid] for tid in token_ids)
        return out


class Encoder:
    """Encode sentences into anchor activation vectors.

    Uses SPLADE for broad vocabulary coverage across any domain, then
    projects onto a typed anchor schema. No training required — just
    define your schema tokens and go.

    Quick start:
        encoder = Encoder(schema=my_schema)
        activations = encoder.encode(["Toyota invests in batteries."])

    From a saved config directory:
        encoder = Encoder.from_dir("models/my-domain/")
    """

    def __init__(self, schema=None,
                 model_name: str = _DEFAULT_MODEL,
                 anchor_names: list[str] | None = None,
                 anchor_types: dict[str, str] | None = None,
                 max_length: int = 256,
                 device: str | None = None):
        self.model_name = model_name
        self.max_length = max_length

        # SPLADE encoder
        self.splade = SpladeEncoder(model_name, max_length, device)
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

    @classmethod
    def from_dir(cls, path: str | Path, device: str | None = None) -> Encoder:
        """Load encoder config from a directory (schema + settings)."""
        from .schema import AnchorSchema

        p = Path(path)
        with open(p / "config.json") as f:
            config = json.load(f)

        model_name = config.get("model_name", _DEFAULT_MODEL)
        max_length = config.get("max_length", 256)

        # Load schema
        schema_path = p / "schema.json"
        if schema_path.exists():
            schema = AnchorSchema.from_file(schema_path)
        else:
            # Build from config
            anchor_defs = config.get("anchor_defs", {})
            if anchor_defs:
                schema = AnchorSchema(anchor_defs)
            else:
                names = config.get("anchor_names", [])
                types = config.get("anchor_types", {})
                schema = AnchorSchema({
                    name: {"type": types.get(name, "feature"), "tokens": [name]}
                    for name in names
                })

        return cls(
            schema=schema,
            model_name=model_name,
            max_length=max_length,
            device=device,
        )

    def encode(self, texts: list[str],
               batch_size: int = 32) -> np.ndarray:
        """Encode sentences → (n_texts, n_anchors) anchor activation matrix.

        SPLADE scores are in range [0, ~5]. Higher = stronger activation.
        """
        sparse = self.splade.encode_sparse(texts, batch_size=batch_size)

        if self._projector is None:
            raise RuntimeError("No schema/projector configured. Pass schema= to __init__.")

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
        """Find character spans where an anchor's tokens appear in text."""
        info = anchor_defs.get(anchor_name, {})
        tokens = info.get("tokens", [])
        if not tokens:
            return []

        sorted_tokens = sorted(tokens, key=len, reverse=True)
        # ASCII tokens get \b boundaries; CJK/non-ASCII match as substrings
        # since those scripts have no word-boundary concept.
        alts = []
        for t in sorted_tokens:
            esc = re.escape(t)
            alts.append(rf"\b{esc}\b" if t.isascii() else esc)
        pattern = re.compile("|".join(alts), re.IGNORECASE)
        return [
            {"text": m.group(), "start": m.start(), "end": m.end()}
            for m in pattern.finditer(text)
        ]
