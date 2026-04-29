"""SPLADE encoder and anchor projection for change of basis.

The encoder implements a two-stage pipeline:
1. SpladeEncoder: Maps text to sparse 30,522-dimensional BERT token space
2. AnchorProjector: Maps SPLADE token space to domain anchor space

All model weights are loaded from the bundled path (no network access required).
Encoding is deterministic and stateless.
"""

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from infon.schema import AnchorSchema

# Bundled model path (relative to this file)
BUNDLED_MODEL_PATH = Path(__file__).parent / "models" / "splade-tiny"


class SpladeEncoder:
    """SPLADE encoder for sparse text representation.

    Uses the bundled splade-tiny model (4.4M params) to encode text into a sparse
    30,522-dimensional vector via log(1 + ReLU(MLM_logits)) max-pooled across tokens.

    The model is loaded from the bundled path on first initialization and cached
    in memory. No network access is required.
    """

    def __init__(self, model_path: Path | None = None):
        """Initialize the SPLADE encoder.

        Args:
            model_path: Optional override for model path (defaults to bundled path)
        """
        self.model_path = model_path or BUNDLED_MODEL_PATH
        
        # Lazy-init model and tokenizer
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def model(self) -> Any:
        """Lazy-load the SPLADE model."""
        if self._model is None:
            # Load from bundled path, offline mode
            self._model = AutoModelForMaskedLM.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
            self._model.eval()  # Set to evaluation mode
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Lazy-load the tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
        return self._tokenizer

    def encode_sparse(self, text: str) -> dict[int, float]:
        """Encode text to sparse SPLADE vector.

        Uses log(1 + ReLU(MLM_logits)) max-pooled across token positions.

        Args:
            text: Input text to encode

        Returns:
            Dict mapping token_id -> activation (non-zero only)
        """
        if not text.strip():
            return {}

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Forward pass (no gradient needed)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape: (batch=1, seq_len, vocab_size=30522)

        # Apply SPLADE formula: log(1 + ReLU(logits))
        # Max-pool across token positions (dim=1)
        relu_logits = torch.relu(logits)
        log_relu = torch.log1p(relu_logits)  # log(1 + x)
        max_activations = torch.max(log_relu, dim=1).values  # shape: (1, vocab_size)

        # Convert to sparse dict (only non-zero activations)
        activations = max_activations.squeeze(0)  # shape: (vocab_size,)
        sparse_vector = {}

        for token_id in range(activations.shape[0]):
            activation = activations[token_id].item()
            if activation > 0.0:
                sparse_vector[token_id] = activation

        return sparse_vector


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
