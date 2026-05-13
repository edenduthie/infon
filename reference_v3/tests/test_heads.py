"""Tests for multi-task classification heads."""

import pytest
import torch
from cognition.heads import (
    NLIHead, RelevanceHead, PolarityHead, RelationTypeHead,
    CognitionHeads, TrainingSample,
)
from cognition.dempster_shafer import MassFunction


class TestNLIHead:
    def test_forward_shape(self):
        head = NLIHead(hidden_dim=128)
        p = torch.randn(4, 128)
        h = torch.randn(4, 128)
        logits = head(p, h)
        assert logits.shape == (4, 3)

    def test_predict_mass(self):
        head = NLIHead(hidden_dim=128)
        p = torch.randn(2, 128)
        h = torch.randn(2, 128)
        masses = head.predict_mass(p, h)
        assert len(masses) == 2
        for m in masses:
            assert isinstance(m, MassFunction)
            total = m.supports + m.refutes + m.uncertain + m.theta
            assert abs(total - 1.0) < 1e-5


class TestRelevanceHead:
    def test_forward_shape(self):
        head = RelevanceHead(hidden_dim=128)
        claim = torch.randn(4, 128)
        ev = torch.randn(4, 128)
        logits = head(claim, ev)
        assert logits.shape == (4, 2)

    def test_predict_relevant(self):
        head = RelevanceHead(hidden_dim=128)
        claim = torch.randn(3, 128)
        ev = torch.randn(3, 128)
        mask = head.predict_relevant(claim, ev, threshold=0.5)
        assert mask.shape == (3,)
        assert mask.dtype == bool


class TestPolarityHead:
    def test_forward_shape(self):
        head = PolarityHead(hidden_dim=128)
        cls = torch.randn(4, 128)
        logits = head(cls)
        assert logits.shape == (4, 3)

    def test_predict_polarity(self):
        head = PolarityHead(hidden_dim=128)
        cls = torch.randn(3, 128)
        pols = head.predict_polarity(cls)
        assert len(pols) == 3
        for p in pols:
            assert p in {1, 0, -1}


class TestRelationTypeHead:
    def test_forward_shape(self):
        head = RelationTypeHead(hidden_dim=128)
        cls = torch.randn(4, 128)
        logits = head(cls)
        assert logits.shape == (4, 5)

    def test_predict_type(self):
        head = RelationTypeHead(hidden_dim=128)
        cls = torch.randn(3, 128)
        types = head.predict_type(cls)
        assert len(types) == 3
        valid = {"causal", "temporal", "spatial", "attributive", "none"}
        for t in types:
            assert t in valid


class TestCognitionHeads:
    def test_save_load(self, tmp_path):
        heads = CognitionHeads(hidden_dim=128)
        heads.save(tmp_path)
        loaded = CognitionHeads.load(tmp_path)
        # Check param counts match
        assert heads.param_count() == loaded.param_count()

    def test_param_count(self):
        heads = CognitionHeads(hidden_dim=128)
        counts = heads.param_count()
        assert counts["total"] > 0
        assert counts["nli"] > 0
        assert counts["relevance"] > 0
        assert counts["polarity"] > 0
        assert counts["relation_type"] > 0

    def test_load_trained(self):
        """Test loading the trained heads from the model directory."""
        from pathlib import Path
        heads_path = Path(__file__).parent.parent / "src" / "cognition" / "model"
        if (heads_path / "heads.pt").exists():
            heads = CognitionHeads.load(heads_path)
            assert heads.param_count()["total"] > 100000  # ~115K with InferSent
