"""LLM zero-shot fact-checking baseline using Amazon Bedrock.

In replay_only mode (default), only cached responses are used — no real
API calls are made. Use record mode to populate the cache with real responses.

The verdict_to_mass, _truncate_evidence, and _build_prompt functions are
pure (no I/O) and are tested independently.
"""

from __future__ import annotations

import json

from benchmarks.types import EvalClaim, MassFunction
from baselines._llm_cache import LLMCache, LLMCacheMissError

# Maximum number of evidence documents to include
_MAX_EVIDENCE_DOCS = 5
# Maximum words per evidence document before truncation
_MAX_WORDS_PER_DOC = 600

SYSTEM_PROMPT = """You are a fact-checking assistant. Given a claim and evidence passages, determine whether the evidence SUPPORTS or REFUTES the claim, or if there is Not Enough Information (NEI).

Respond with a single JSON line: {"verdict": "SUPPORTS"|"REFUTES"|"NEI", "confidence": 0.0-1.0}

Rules:
- confidence reflects your certainty given ONLY the provided evidence (not your pretraining)
- A verdict of "NEI" MUST have confidence 1.0 — you are certain there is not enough information
- Do not include any text outside the JSON line"""

USER_PROMPT_TEMPLATE = """Claim: {claim}

Evidence:
{evidence}

Verdict (JSON only):"""


def verdict_to_mass(verdict: str | None, confidence: float | None) -> MassFunction:
    """Map an LLM verdict+confidence to a Dempster-Shafer MassFunction.

    Mapping rules (consistent with DS semantics):
      SUPPORTS  → m_s = confidence, m_theta = 1 - confidence
      REFUTES   → m_r = confidence, m_theta = 1 - confidence
      NEI       → m_theta = 1.0  (confidence is ignored; NEI = full ignorance)
      parse failure / None → m_theta = 1.0  (safe fallback)
    """
    if verdict == "SUPPORTS" and confidence is not None:
        return MassFunction(m_s=confidence, m_r=0.0, m_u=0.0, m_theta=1.0 - confidence)
    elif verdict == "REFUTES" and confidence is not None:
        return MassFunction(m_s=0.0, m_r=confidence, m_u=0.0, m_theta=1.0 - confidence)
    elif verdict == "NEI":
        return MassFunction(m_s=0.0, m_r=0.0, m_u=0.0, m_theta=1.0)
    else:
        # Parse failure or unknown verdict → full ignorance
        return MassFunction(m_s=0.0, m_r=0.0, m_u=0.0, m_theta=1.0)


def _truncate_evidence(evidence_docs: list[str]) -> str:
    """Format and truncate evidence docs for the LLM prompt.

    Rules:
      - Empty list          → "[none provided]"
      - More than 5 docs    → keep first 5, append "[N document(s) omitted]"
      - Doc > 600 words     → truncate to 600 words and append "[truncated]"
      - Result is numbered: [1] doc1, [2] doc2, ...
    """
    if not evidence_docs:
        return "[none provided]"

    n_total = len(evidence_docs)
    selected = evidence_docs[:_MAX_EVIDENCE_DOCS]
    n_omitted = n_total - len(selected)

    parts = []
    for i, doc in enumerate(selected, start=1):
        words = doc.split()
        if len(words) > _MAX_WORDS_PER_DOC:
            truncated = " ".join(words[:_MAX_WORDS_PER_DOC]) + " [truncated]"
        else:
            truncated = doc
        parts.append(f"[{i}] {truncated}")

    result = "\n\n".join(parts)
    if n_omitted > 0:
        result += f"\n\n[{n_omitted} document{'s' if n_omitted > 1 else ''} omitted]"

    return result


def _build_prompt(claim_text: str, evidence_text: str) -> tuple[str, str]:
    """Build the (system_prompt, user_prompt) pair for the LLM call.

    Returns a 2-tuple of strings. Deterministic for identical inputs.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        claim=claim_text,
        evidence=evidence_text,
    )
    return SYSTEM_PROMPT, user_prompt


def _parse_response(response_text: str) -> tuple[str | None, float | None]:
    """Parse the LLM's JSON response into (verdict, confidence).

    Returns (None, None) on any parse error.
    """
    try:
        data = json.loads(response_text.strip())
        verdict = data.get("verdict")
        confidence = data.get("confidence")
        if verdict not in ("SUPPORTS", "REFUTES", "NEI"):
            return None, None
        if confidence is not None:
            confidence = float(confidence)
        return verdict, confidence
    except (json.JSONDecodeError, ValueError, AttributeError):
        return None, None


class LLMZeroShot:
    """Zero-shot LLM fact-checking baseline via Amazon Bedrock.

    In replay_only mode (default), only pre-recorded responses are returned.
    In record mode, real Bedrock calls are made and results appended to cache.

    IMPORTANT: Never make LLM calls outside of record mode.
    """

    name = "llm_zeroshot"
    MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    def __init__(
        self,
        cache_path: str,
        mode: str = "replay_only",
        max_input_tokens: int = 5_000_000,
    ) -> None:
        self.cache = LLMCache(cache_path, mode=mode)
        self.mode = mode
        self.max_input_tokens = max_input_tokens

    def evaluate(self, claim: EvalClaim) -> MassFunction:
        """Evaluate a claim against its evidence docs.

        In replay_only mode, raises LLMCacheMissError if the key is not
        in the cache (no Bedrock call is made).
        """
        evidence_text = _truncate_evidence(claim.evidence_docs)
        system_p, user_p = _build_prompt(claim.claim_text, evidence_text)
        key = LLMCache.make_key(self.MODEL_ID, system_p, user_p, 0.0, 256, "bedrock")

        try:
            entry = self.cache.get(key)
            response_text = entry["response_text"]
        except LLMCacheMissError:
            if self.mode == "replay_only":
                raise
            response_text = self._call_bedrock(system_p, user_p)

        verdict, confidence = _parse_response(response_text)
        return verdict_to_mass(verdict, confidence)

    def _call_bedrock(self, system_prompt: str, user_prompt: str) -> str:
        """Make a real Bedrock API call and record the response.

        Only called when mode == "record". Never called in replay_only mode.
        """
        import boto3

        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 256,
            "temperature": 0.0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        response = client.invoke_model(
            modelId=self.MODEL_ID,
            body=json.dumps(body),
        )
        parsed = json.loads(response["body"].read())
        input_tokens = parsed.get("usage", {}).get("input_tokens", 0)
        output_tokens = parsed.get("usage", {}).get("output_tokens", 0)
        text = parsed["content"][0]["text"]

        key = LLMCache.make_key(self.MODEL_ID, system_prompt, user_prompt, 0.0, 256, "bedrock")
        self.cache.put(key, text, input_tokens, output_tokens)
        return text
