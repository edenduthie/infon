"""infon.logical_tool — JSON expression DSL + agent tool."""
from __future__ import annotations

import json
import os
import tempfile

import pytest


SCHEMA = {
    "toyota":   {"type": "actor",    "tokens": ["toyota"]},
    "honda":    {"type": "actor",    "tokens": ["honda"]},
    "tesla":    {"type": "actor",    "tokens": ["tesla"]},
    "catl":     {"type": "actor",    "tokens": ["catl"]},
    "invests":  {"type": "relation", "tokens": ["invest", "invests"]},
    "partners": {"type": "relation", "tokens": ["partner", "partners"]},
    "acquires": {"type": "relation", "tokens": ["acquire", "acquires"]},
    "battery":  {"type": "feature",  "tokens": ["battery", "batteries"]},
}


DOCS = [
    {"id": "d1", "text": "Toyota invests in battery technology."},
    {"id": "d2", "text": "Honda partners with CATL on battery supply."},
    {"id": "d3", "text": "Tesla invests in battery factories."},
]


def _make_cog(tmp):
    from infon import InfonEngine, InfonConfig
    sp = os.path.join(tmp, "schema.json")
    with open(sp, "w") as f:
        json.dump(SCHEMA, f)
    cog = InfonEngine(InfonConfig(
        schema_path=sp,
        db_path=os.path.join(tmp, "cog.db"),
        random_state=42,
    ))
    for d in DOCS:
        cog.ingest([d])
    cog.consolidate()
    return cog


def test_validate_rejects_bad_exprs():
    from infon.logical_tool import validate, LogicalExprError

    bad_cases = [
        42,
        "not a dict",
        {"op": "wat"},
        {"op": "triple", "s": "a"},                  # missing p, o
        {"op": "and"},                                 # no args
        {"op": "and", "args": [{"op": "triple", "s": "a", "p": "b", "o": "c"}]},  # 1 arg
        {"op": "not", "args": []},
        {"op": "if", "args": [{"op": "triple", "s": "a", "p": "b", "o": "c"}]},  # only 1 arg
        {"op": "exists", "type": "actor"},             # no body
    ]
    for expr in bad_cases:
        with pytest.raises(LogicalExprError):
            validate(expr)


def test_validate_accepts_good_exprs():
    from infon.logical_tool import validate

    good = [
        {"op": "triple", "s": "a", "p": "b", "o": "c"},
        {"op": "and", "args": [
            {"op": "triple", "s": "a", "p": "b", "o": "c"},
            {"op": "triple", "s": "d", "p": "e", "o": "f"},
        ]},
        {"op": "not", "args": [{"op": "triple", "s": "a", "p": "b", "o": "c"}]},
        {"op": "exists", "type": "actor", "body":
            {"op": "triple", "s": "?x", "p": "invests", "o": "battery"}},
    ]
    for expr in good:
        validate(expr)  # should not raise


def test_triple_evaluates_like_reason():
    from infon.logical_tool import evaluate

    with tempfile.TemporaryDirectory() as tmp:
        cog = _make_cog(tmp)
        result = evaluate(cog, {
            "op": "triple",
            "s": "toyota", "p": "invests", "o": "battery",
        })
        assert result["verdict"] == "SUPPORTS"
        assert result["mass"]["supports"] > 0.5
        assert result["mass"]["theta"] < 0.5
        cog.close()


def test_not_swaps_supports_refutes():
    from infon.logical_tool import evaluate

    with tempfile.TemporaryDirectory() as tmp:
        cog = _make_cog(tmp)
        pos = evaluate(cog, {
            "op": "triple", "s": "toyota", "p": "invests", "o": "battery",
        })
        neg = evaluate(cog, {
            "op": "not",
            "args": [{"op": "triple", "s": "toyota",
                      "p": "invests", "o": "battery"}],
        })
        assert pos["mass"]["supports"] == neg["mass"]["refutes"]
        assert pos["mass"]["refutes"] == neg["mass"]["supports"]
        cog.close()


def test_and_combines_sub_expressions():
    from infon.logical_tool import evaluate

    with tempfile.TemporaryDirectory() as tmp:
        cog = _make_cog(tmp)
        result = evaluate(cog, {"op": "and", "args": [
            {"op": "triple", "s": "toyota", "p": "invests", "o": "battery"},
            {"op": "triple", "s": "honda",  "p": "partners", "o": "catl"},
        ]})
        # Both sub-expressions should have been evaluated and traced
        triple_traces = [t for t in result["trace"] if t["op"] == "triple"]
        assert len(triple_traces) == 2
        and_traces = [t for t in result["trace"] if t["op"] == "and"]
        assert len(and_traces) == 1
        # Verdict should be SUPPORTS since both claims are in corpus
        assert result["verdict"] == "SUPPORTS"
        cog.close()


def test_exists_enumerates_actors():
    from infon.logical_tool import evaluate

    with tempfile.TemporaryDirectory() as tmp:
        cog = _make_cog(tmp)
        result = evaluate(cog, {
            "op": "exists", "type": "actor",
            "body": {"op": "triple", "s": "?x",
                     "p": "invests", "o": "battery"},
        })
        # Should have evaluated one triple per actor (4 of them)
        triple_count = sum(1 for t in result["trace"] if t["op"] == "triple")
        assert triple_count == 4
        # At least one actor (Toyota, Tesla) invests in battery → SUPPORTS
        assert result["verdict"] == "SUPPORTS"
        cog.close()


def test_unknown_claim_returns_high_theta():
    from infon.logical_tool import evaluate

    with tempfile.TemporaryDirectory() as tmp:
        cog = _make_cog(tmp)
        result = evaluate(cog, {
            "op": "triple", "s": "tesla", "p": "acquires", "o": "catl",
        })
        assert result["verdict"] == "NOT ENOUGH INFO"
        assert result["mass"]["theta"] > 0.5
        cog.close()


def test_evaluate_logic_agent_tool():
    """The @tool wrapper works against an Infon instance set via
    create_tools(...)."""
    from infon.agent_tools import create_tools, evaluate_logic

    with tempfile.TemporaryDirectory() as tmp:
        cog = _make_cog(tmp)
        _, _ = create_tools(cog)
        out = evaluate_logic(json.dumps({
            "op": "triple", "s": "toyota",
            "p": "invests", "o": "battery",
        }))
        assert "verdict:" in out
        assert "SUPPORTS" in out
        assert "trace" in out

        # Bad JSON → error message (not a crash)
        err = evaluate_logic("not json at all")
        assert err.startswith("ERROR")

        # Bad grammar → error message
        err = evaluate_logic(json.dumps({"op": "triple", "s": "a"}))
        assert err.startswith("ERROR")
        cog.close()
