"""Agent-facing logical-query DSL.

LLMs are bad at writing SQL / Cypher / SPARQL but good at emitting
structured JSON. This module accepts a JSON expression tree, evaluates
every sub-expression against a fitted HypergraphReasoner, and returns a
Dempster-Shafer mass function for the whole thing.

Grammar
-------
Every expression is a JSON object with an "op" field. Allowed ops:

    {"op": "triple", "s": "<subject>", "p": "<predicate>", "o": "<object>"}
        Leaf. Returns the mass function of the claim
        "<s> <p> <o>" as if it had been passed to reason(...).

    {"op": "and", "args": [expr, expr, ...]}
        Conjunction. Dempster-combines the sub-masses; intersecting
        with {S} for two SUPPORTS and similarly for R.

    {"op": "or", "args": [expr, expr, ...]}
        Disjunction. Duals the and-case — any sub-expression
        supporting is enough to support the whole.

    {"op": "not", "args": [expr]}
        Negation. Swaps supports ↔ refutes in the sub-mass.

    {"op": "if", "args": [premise, conclusion]}
        Material conditional. Equivalent to (or (not premise) conclusion).

    {"op": "exists", "type": "<actor|relation|feature|market>",
     "body": expr_template}
        Existential over typed anchors. Replaces "?x" in the body with
        each anchor of the given type; returns the OR-combination.

    {"op": "forall", "type": "<type>", "body": expr_template}
        Universal over typed anchors. AND-combination instead of OR.

Usage
-----
    from infon.logical_tool import evaluate

    expr = {"op": "and", "args": [
        {"op": "triple", "s": "toyota", "p": "invests", "o": "battery"},
        {"op": "not", "args": [
            {"op": "triple", "s": "toyota", "p": "acquires", "o": "catl"},
        ]},
    ]}

    result = evaluate(cog, expr)
    print(result["verdict"], result["mass"])

Design notes
------------
The evaluator works *entirely in mass-function space*, not in the GNN's
hidden space. That's a deliberate choice: it means every sub-expression
has a verdict + θ the agent can inspect, and compound queries preserve
the Dempster-Shafer calibration that the rest of the library built its
reputation on. The GNN's `compound_query()` method is an alternative
that composes in the hidden space — it's less agent-friendly because
the intermediate values are dense vectors, not interpretable verdicts.

A subtle DS property worth knowing about: a mass with `θ = 1.0`
(total ignorance) is a *neutral element* for Dempster combination.
So AND(A_supported, B_ignorant) = A_supported — not "uncertain AND."
That is mathematically correct under Dempster-Shafer theory: knowing
nothing about B should not undo what you knew about A. If your agent
needs "every argument must have positive support," check the returned
`trace` and reject the query if any sub-expression has θ > threshold.
"""
from __future__ import annotations

from typing import Any

from .dempster_shafer import MassFunction, combine_multiple, combine_dempster


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate(cog, expr: dict, *,
             reasoner=None,
             verdict_theta_floor: float = 0.85) -> dict:
    """Entry point. Parse + evaluate a JSON expression.

    Parameters
    ----------
    cog : InfonEngine
        Fitted instance with an ingested corpus.
    expr : dict
        A JSON-serialisable expression. See module docstring for grammar.
    reasoner : HypergraphReasoner, optional
        Pass an already-fitted reasoner to avoid re-fitting. Default:
        cog.reasoner().
    verdict_theta_floor : float
        θ threshold above which we return NOT ENOUGH INFO rather than
        argmax.

    Returns
    -------
    dict with keys:
        'verdict'   : str — one of SUPPORTS / REFUTES / NOT ENOUGH INFO / CONTRADICTED
        'mass'      : dict — {supports, refutes, uncertain, theta}
        'trace'     : list — per-sub-expression (op, verdict, mass)
        'expr'      : the input expression echoed back for audit
    """
    validate(expr)
    if reasoner is None:
        reasoner = cog.reasoner()

    trace: list[dict] = []
    mf = _eval_node(cog, reasoner, expr, trace)

    if mf.theta > verdict_theta_floor:
        verdict = "NOT ENOUGH INFO"
    else:
        scores = {
            "SUPPORTS":       mf.supports,
            "REFUTES":        mf.refutes,
            "NOT ENOUGH INFO": mf.uncertain + mf.theta,
        }
        verdict = max(scores, key=scores.get)

    return {
        "verdict": verdict,
        "mass": {
            "supports":  mf.supports,
            "refutes":   mf.refutes,
            "uncertain": mf.uncertain,
            "theta":     mf.theta,
        },
        "trace": trace,
        "expr": expr,
    }


# ══════════════════════════════════════════════════════════════════════
# Validation — catches malformed expressions before the LLM gets an
# unhelpful stack trace.
# ══════════════════════════════════════════════════════════════════════

VALID_OPS = {"triple", "and", "or", "not", "if", "exists", "forall"}


class LogicalExprError(ValueError):
    """Raised when a JSON expression violates the grammar."""


def validate(expr: Any) -> None:
    """Raise LogicalExprError if the expression is not well-formed.

    Doesn't evaluate — just structure-checks. Useful for early failure
    when an LLM emits garbage JSON.
    """
    if not isinstance(expr, dict):
        raise LogicalExprError(f"expression must be a dict, got {type(expr).__name__}")
    op = expr.get("op")
    if op not in VALID_OPS:
        raise LogicalExprError(
            f"unknown op {op!r}; valid: {sorted(VALID_OPS)}"
        )
    if op == "triple":
        for f in ("s", "p", "o"):
            if f not in expr or not isinstance(expr[f], str):
                raise LogicalExprError(
                    f"triple expression must have string fields s, p, o — missing {f}"
                )
    elif op in ("and", "or"):
        if "args" not in expr or not isinstance(expr["args"], list):
            raise LogicalExprError(f"{op!r} requires 'args' list")
        if len(expr["args"]) < 2:
            raise LogicalExprError(f"{op!r} requires at least 2 args")
        for a in expr["args"]:
            validate(a)
    elif op == "not":
        if "args" not in expr or not isinstance(expr["args"], list) \
                or len(expr["args"]) != 1:
            raise LogicalExprError("not requires exactly 1 arg in 'args' list")
        validate(expr["args"][0])
    elif op == "if":
        if "args" not in expr or len(expr.get("args", [])) != 2:
            raise LogicalExprError("if requires exactly 2 args [premise, conclusion]")
        for a in expr["args"]:
            validate(a)
    elif op in ("exists", "forall"):
        if "type" not in expr or "body" not in expr:
            raise LogicalExprError(f"{op!r} requires 'type' and 'body'")
        validate(expr["body"])


# ══════════════════════════════════════════════════════════════════════
# Recursive evaluator — each branch returns a MassFunction
# ══════════════════════════════════════════════════════════════════════

def _eval_node(cog, reasoner, expr: dict, trace: list) -> MassFunction:
    op = expr["op"]

    if op == "triple":
        question = f"Did {expr['s']} {expr['p']} {expr['o']}?"
        r = reasoner.reason(question)
        mf = r.mass
        trace.append({"op": op, "query": question,
                      "verdict": r.verdict, "mass": _mf_dict(mf)})
        return mf

    if op == "not":
        inner = _eval_node(cog, reasoner, expr["args"][0], trace)
        mf = MassFunction(
            supports=inner.refutes,
            refutes=inner.supports,
            uncertain=inner.uncertain,
            theta=inner.theta,
        )
        trace.append({"op": op, "verdict": _verdict(mf),
                      "mass": _mf_dict(mf)})
        return mf

    if op == "and":
        parts = [_eval_node(cog, reasoner, a, trace) for a in expr["args"]]
        mf = combine_multiple(parts)
        trace.append({"op": op, "n_args": len(parts),
                      "verdict": _verdict(mf), "mass": _mf_dict(mf)})
        return mf

    if op == "or":
        # Dempster's rule on De-Morganed masses: not(and(not a, not b)).
        # Equivalently, flip supports ↔ refutes in each, combine, flip back.
        parts = [_eval_node(cog, reasoner, a, trace) for a in expr["args"]]
        flipped = [
            MassFunction(supports=m.refutes, refutes=m.supports,
                         uncertain=m.uncertain, theta=m.theta)
            for m in parts
        ]
        combined = combine_multiple(flipped)
        mf = MassFunction(
            supports=combined.refutes,
            refutes=combined.supports,
            uncertain=combined.uncertain,
            theta=combined.theta,
        )
        trace.append({"op": op, "n_args": len(parts),
                      "verdict": _verdict(mf), "mass": _mf_dict(mf)})
        return mf

    if op == "if":
        # if(premise, conclusion) = or(not premise, conclusion)
        premise = _eval_node(cog, reasoner, expr["args"][0], trace)
        conclusion = _eval_node(cog, reasoner, expr["args"][1], trace)
        not_p = MassFunction(
            supports=premise.refutes, refutes=premise.supports,
            uncertain=premise.uncertain, theta=premise.theta,
        )
        # OR via De-Morgan
        flipped_np = MassFunction(
            supports=not_p.refutes, refutes=not_p.supports,
            uncertain=not_p.uncertain, theta=not_p.theta,
        )
        flipped_c = MassFunction(
            supports=conclusion.refutes, refutes=conclusion.supports,
            uncertain=conclusion.uncertain, theta=conclusion.theta,
        )
        combined = combine_dempster(flipped_np, flipped_c)
        mf = MassFunction(
            supports=combined.refutes, refutes=combined.supports,
            uncertain=combined.uncertain, theta=combined.theta,
        )
        trace.append({"op": op, "verdict": _verdict(mf),
                      "mass": _mf_dict(mf)})
        return mf

    if op in ("exists", "forall"):
        anchor_type = expr["type"]
        body = expr["body"]
        anchors = [
            name for name, t in cog.schema.types.items()
            if t == anchor_type
        ]
        if not anchors:
            # No candidates → uninformative
            mf = MassFunction(theta=1.0)
            trace.append({"op": op, "type": anchor_type,
                          "verdict": "NOT ENOUGH INFO",
                          "mass": _mf_dict(mf),
                          "note": f"no anchors of type {anchor_type!r}"})
            return mf
        parts: list[MassFunction] = []
        for name in anchors:
            sub_expr = _substitute(body, "?x", name)
            parts.append(_eval_node(cog, reasoner, sub_expr, trace))
        if op == "exists":
            # OR-combine
            flipped = [
                MassFunction(supports=m.refutes, refutes=m.supports,
                             uncertain=m.uncertain, theta=m.theta)
                for m in parts
            ]
            combined = combine_multiple(flipped)
            mf = MassFunction(
                supports=combined.refutes, refutes=combined.supports,
                uncertain=combined.uncertain, theta=combined.theta,
            )
        else:
            # AND-combine
            mf = combine_multiple(parts)
        trace.append({"op": op, "type": anchor_type,
                      "n_candidates": len(anchors),
                      "verdict": _verdict(mf), "mass": _mf_dict(mf)})
        return mf

    # Unreachable — validate() would have caught it
    raise LogicalExprError(f"unhandled op: {op!r}")


def _substitute(expr: Any, placeholder: str, value: str) -> Any:
    """Deep-copy `expr`, replacing string occurrences of `placeholder`
    with `value`. Used by forall/exists to bind the quantifier."""
    if isinstance(expr, dict):
        return {k: _substitute(v, placeholder, value) for k, v in expr.items()}
    if isinstance(expr, list):
        return [_substitute(item, placeholder, value) for item in expr]
    if isinstance(expr, str) and expr == placeholder:
        return value
    return expr


def _mf_dict(m: MassFunction) -> dict:
    return {
        "supports":  round(m.supports, 4),
        "refutes":   round(m.refutes, 4),
        "uncertain": round(m.uncertain, 4),
        "theta":     round(m.theta, 4),
    }


def _verdict(m: MassFunction, theta_floor: float = 0.85) -> str:
    if m.theta > theta_floor:
        return "NOT ENOUGH INFO"
    scores = {
        "SUPPORTS":       m.supports,
        "REFUTES":        m.refutes,
        "NOT ENOUGH INFO": m.uncertain + m.theta,
    }
    return max(scores, key=scores.get)


# ══════════════════════════════════════════════════════════════════════
# Tool description — what an LLM sees in its tool schema
# ══════════════════════════════════════════════════════════════════════

TOOL_DESCRIPTION = """\
Evaluate a logical expression against the infon knowledge graph.

The expression is a JSON object. Allowed operators:
  triple:  {"op":"triple","s":"<actor>","p":"<relation>","o":"<object>"}
  and:     {"op":"and","args":[expr, expr, ...]}
  or:      {"op":"or","args":[expr, expr, ...]}
  not:     {"op":"not","args":[expr]}
  if:      {"op":"if","args":[premise_expr, conclusion_expr]}
  exists:  {"op":"exists","type":"actor","body":<expr with "?x" placeholder>}
  forall:  {"op":"forall","type":"actor","body":<expr with "?x" placeholder>}

Returns: verdict, Dempster-Shafer mass (supports / refutes / uncertain / θ),
and a trace of every sub-expression's result for inspection.

Example:
  {"op":"and","args":[
     {"op":"triple","s":"toyota","p":"invests","o":"battery"},
     {"op":"not","args":[
        {"op":"triple","s":"toyota","p":"acquires","o":"catl"}]}]}

Tips:
- Subject must be an actor anchor; predicate must be a relation anchor;
  object can be any anchor type. Check the schema via list_anchors().
- θ near 1.0 means "no evidence either way" — the tool is being honest
  about ignorance, not guessing.
"""


__all__ = [
    "evaluate",
    "validate",
    "LogicalExprError",
    "TOOL_DESCRIPTION",
]
