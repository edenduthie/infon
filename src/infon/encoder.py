"""SPLADE encoder and anchor projection for change of basis.

The encoder implements a two-stage pipeline:
1. SpladeEncoder: Maps text to sparse 30,522-dimensional BERT token space
2. AnchorProjector: Maps SPLADE token space to domain anchor space

Model weights are downloaded on first use from the HuggingFace Hub and cached
locally by the `transformers` library (default: ~/.cache/huggingface). The model
ID can be overridden via the INFON_SPLADE_MODEL environment variable, or by
passing `model_name` to SpladeEncoder. A local filesystem path also works.
"""

import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from infon.schema import AnchorSchema

DEFAULT_SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"


class SpladeEncoder:
    """SPLADE encoder for sparse text representation.

    Encodes text into a sparse 30,522-dimensional vector via
    log(1 + ReLU(MLM_logits)) max-pooled across tokens.

    The model is fetched from the HuggingFace Hub on first use and cached by
    the transformers library. Override with the INFON_SPLADE_MODEL env var or
    by passing `model_name` (which may be a Hub ID or a local path).
    """

    def __init__(self, model_name: str | Path | None = None):
        """Initialize the SPLADE encoder.

        Args:
            model_name: HF Hub model ID or local filesystem path. Defaults to
                the INFON_SPLADE_MODEL env var, then DEFAULT_SPLADE_MODEL.
        """
        self.model_name = str(
            model_name
            or os.environ.get("INFON_SPLADE_MODEL")
            or DEFAULT_SPLADE_MODEL
        )

        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def model(self) -> Any:
        """Lazy-load the SPLADE model."""
        if self._model is None:
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.eval()
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Lazy-load the tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def encode_sparse(self, text: str) -> dict[int, float]:
        """Encode text to sparse SPLADE vector (single-text convenience).

        For ingest pipelines that encode many texts (schema discovery,
        docstring extraction), prefer ``encode_sparse_batch`` — batching
        through transformers gives a 5-10x speedup on CPU because the
        per-call Python and tokenizer overhead dominates the actual
        forward pass for short inputs.
        """
        if not text.strip():
            return {}
        return self.encode_sparse_batch([text])[0]

    def encode_sparse_batch(
        self, texts: list[str], batch_size: int = 32, max_length: int = 128
    ) -> list[dict[int, float]]:
        """Encode many texts to sparse SPLADE vectors in one forward pass per
        batch. Empty / whitespace-only inputs map to ``{}``.

        Args:
            texts: List of input strings.
            batch_size: Inputs per forward pass. 32 is a good default on CPU
                with 512-token cap; larger batches help on GPU but blow up CPU
                memory because the dense intermediate is ``batch × seq × vocab``.

        Returns:
            One sparse-dict per input, in the same order as ``texts``. Empty
            inputs produce empty dicts (no model call wasted on them).
        """
        if not texts:
            return []

        results: list[dict[int, float]] = [{}] * len(texts)
        # Build the work queue of (original-index, text) pairs, skipping
        # empties so we don't waste forward passes on them.
        work: list[tuple[int, str]] = [
            (i, t) for i, t in enumerate(texts) if t.strip()
        ]
        if not work:
            return results

        for chunk_start in range(0, len(work), batch_size):
            chunk = work[chunk_start : chunk_start + batch_size]
            chunk_texts = [t for _, t in chunk]

            inputs = self.tokenizer(
                chunk_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                # logits: (batch, seq_len, vocab_size)
                logits = outputs.logits

            relu_logits = torch.relu(logits)
            log_relu = torch.log1p(relu_logits)
            # Mask padding positions so they don't contribute to the max-pool.
            attention_mask = inputs["attention_mask"].unsqueeze(-1).bool()
            log_relu = log_relu.masked_fill(~attention_mask, 0.0)
            # max-pool across token positions → (batch, vocab_size)
            max_activations = torch.max(log_relu, dim=1).values

            # Sparse-ify each row using nonzero indices for efficiency on
            # CPU — iterating every vocab id was the dominant cost in the
            # original per-text path.
            for row_idx, (original_idx, _) in enumerate(chunk):
                row = max_activations[row_idx]
                nonzero_idx = torch.nonzero(row, as_tuple=False).squeeze(-1)
                vals = row[nonzero_idx]
                results[original_idx] = {
                    int(tid.item()): float(v.item())
                    for tid, v in zip(nonzero_idx, vals)
                }

        return results


class AnchorProjector:
    """Projects SPLADE token space to anchor concept space.

    Maps a sparse SPLADE vector (30,522-dim token space) to a dense anchor-space
    vector by max-pooling SPLADE activations over each anchor's token list.
    """

    def __init__(self, tokenizer: Any):
        """Initialize the anchor projector.

        Args:
            tokenizer: The SPLADE tokenizer (for token -> id mapping)
        """
        self.tokenizer = tokenizer

    def project(
        self,
        sparse_vector: dict[int, float],
        schema: AnchorSchema,
    ) -> dict[str, float]:
        """Project SPLADE vector to anchor space.

        For each anchor in the schema, max-pool the SPLADE activations over the
        anchor's token list.

        Args:
            sparse_vector: Sparse SPLADE vector (token_id -> activation)
            schema: AnchorSchema defining the target anchor space

        Returns:
            Dict mapping anchor_key -> activation (non-zero only)
        """
        anchor_vector: dict[str, float] = {}

        for anchor_key, anchor in schema.anchors.items():
            max_activation = 0.0

            # Max-pool over anchor's tokens
            for token_str in anchor.tokens:
                # Get token IDs for this string (may be multiple subword tokens)
                token_ids = self.tokenizer.encode(
                    token_str,
                    add_special_tokens=False,
                )

                # Max-pool over all token IDs for this token string
                for token_id in token_ids:
                    if token_id in sparse_vector:
                        activation = sparse_vector[token_id]
                        max_activation = max(max_activation, activation)

            # Only include non-zero activations
            if max_activation > 0.0:
                anchor_vector[anchor_key] = max_activation

        return anchor_vector


# Module-level singleton encoder (lazy-initialized)
_encoder_instance: SpladeEncoder | None = None


def _get_encoder() -> SpladeEncoder:
    """Get the singleton SPLADE encoder instance."""
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = SpladeEncoder()
    return _encoder_instance


def encode(text: str, schema: AnchorSchema) -> dict[str, float]:
    """Encode text to anchor space (end-to-end).

    Combines SPLADE encoding and anchor projection in a single call.

    Args:
        text: Input text to encode
        schema: AnchorSchema defining the target anchor space

    Returns:
        Dict mapping anchor_key -> activation (non-zero only)
    """
    encoder = _get_encoder()
    projector = AnchorProjector(encoder.tokenizer)

    # Stage 1: Encode to SPLADE space
    sparse_vector = encoder.encode_sparse(text)

    # Stage 2: Project to anchor space
    anchor_vector = projector.project(sparse_vector, schema)

    return anchor_vector
