#!/usr/bin/env python3
"""
Structured extraction model benchmark.

Tests 6 models on 20 articles, comparing quality + throughput for
geopolitical event extraction. Outputs results to S3.

Models tested:
  1. NuExtract-2.0-4B  (baseline, Qwen2.5-VL-3B)
  2. NuExtract-2.0-8B  (same family, Qwen2.5-VL-7B)
  3. Qwen2.5-7B-Instruct  (strong structured output)
  4. Qwen3-8B  (newest Qwen, thinking mode)
  5. Phi-4-mini-instruct  (3.8B, MIT, efficient)
  6. Gemma-3-12b-it  (12B, best reasoning in class)

Two-pass test:
  7. Relevance-only pass (tiny schema) → full extraction on relevant only

Usage:
    python benchmark.py                        # Run all models
    python benchmark.py --models nuextract-8b qwen2.5-7b   # Specific models
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime

import boto3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

S3_BUCKET = "sirius-dimefiled-results"
S3_OUTPUT_PREFIX = "benchmark/results/"
S3_INPUT_PREFIX = "nuextract-v2-events/"
REGION = "us-east-1"
TEXT_TRUNCATION = 6000  # Increased from 4000
MAX_NEW_TOKENS = 8192

MODELS = {
    "nuextract-4b": {
        "hf_name": "numind/NuExtract-2.0-4B",
        "type": "nuextract",
        "trust_remote_code": True,
        "max_model_len": 20480,
    },
    "nuextract-8b": {
        "hf_name": "numind/NuExtract-2.0-8B",
        "type": "nuextract",
        "trust_remote_code": True,
        "max_model_len": 20480,
    },
    "qwen2.5-7b": {
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "type": "generic",
        "trust_remote_code": False,
        "max_model_len": 32768,
    },
    "qwen3-8b": {
        "hf_name": "Qwen/Qwen3-8B",
        "type": "generic",
        "trust_remote_code": False,
        "max_model_len": 32768,
        "extra_chat_template_kwargs": {"enable_thinking": False},
    },
    "phi4-mini": {
        "hf_name": "microsoft/Phi-4-mini-instruct",
        "type": "generic",
        "trust_remote_code": True,
        "max_model_len": 32768,
    },
    "gemma3-12b": {
        "hf_name": "google/gemma-3-12b-it",
        "type": "generic",
        "trust_remote_code": False,
        "max_model_len": 32768,
    },
}

# ---------------------------------------------------------------------------
# Extraction schema (common across all models)
# ---------------------------------------------------------------------------

# NuExtract template format (for NuExtract models)
NUEXTRACT_TEMPLATE = """{
  "grayzone_relevance": "enum(high,medium,low,none)",
  "actions": [{
    "action_description": "verbatim-string",
    "l1_domain": "enum(D,I,M,E,F,IN,L,EN)",
    "l2_subdomain": "string",
    "l3_code": "string",
    "escalation_classification": "enum(escalation,de-escalation,cooperation-building,cooperation-maintaining,neutral,ambiguous)",
    "dominant_action_type": "enum(confrontational,cooperative,diplomatic,informational,economic)",
    "scope": "enum(bilateral,regional,global)",
    "claim_type": "enum(territorial-maritime,territorial-land,territorial-airspace,economic-zone,resource-rights,legal-jurisdiction,access-rights,no-claims)",
    "initiators": [{
      "name": "string",
      "canonical_name": "string",
      "country_code": "string",
      "organisation_type": "enum(government,military,intelligence,law-enforcement,legislature,judiciary,diplomatic,state-media,media,civil-society,private-sector,armed-group,individual)",
      "role": "string"
    }],
    "targets": [{
      "name": "string",
      "canonical_name": "string",
      "country_code": "string",
      "organisation_type": "enum(government,military,intelligence,law-enforcement,legislature,judiciary,diplomatic,state-media,media,civil-society,private-sector,armed-group,individual)",
      "role": "string"
    }],
    "locations": [{
      "name": "string",
      "canonical_name": "string",
      "level": "enum(city,admin_region,country,body_of_water,strait,island,reef_shoal,military_base,port,border_zone,airspace,region,other)",
      "country_code": "string",
      "macro_region": "enum(East Asia,Southeast Asia,South Asia,Central Asia,Middle East,Europe,North America,South America,Africa,Oceania,Arctic,Indo-Pacific,Global)"
    }],
    "date_raw": "string",
    "date_iso": "string",
    "influence_patterns": ["enum(escalation-spiral,tit-for-tat,provocation-response,containment,deterrence,firebreak,power-transition,consolidation,erosion,fait-accompli,alliance-formation,wedge-driving,balancing,signaling,testing-probing,demonstration,normalization,cumulative-gain,threshold-testing)"],
    "contestation_dynamics": "enum(escalating-contestation,maintaining-contestation,reducing-contestation,neutral-no-active-claims,building-cooperation,strengthening-alliance)",
    "state_indicators": {
      "actor_posture": "enum(assertive,defensive,cooperative,neutral,coercive,provocative,conciliatory,withdrawn)",
      "regional_tension": "enum(increasing,decreasing,stable,volatile)",
      "prior_state_description": "string",
      "resulting_state_description": "string"
    },
    "cost_profile": {
      "resource_type": "enum(military-assets,diplomatic-capital,economic-leverage,political-capital,intelligence-resources,legal-standing,reputation,financial-instruments,none-apparent)",
      "cost_magnitude": "enum(negligible,low,moderate,significant,major)",
      "risk_level": "enum(none,low,moderate,high,extreme)",
      "reversibility": "enum(easily-reversible,reversible,difficult-to-reverse,irreversible)"
    },
    "implicit_signals": {
      "stated_justification": "string",
      "unstated_objective_hint": "enum(capability-demonstration,precedent-setting,normalization,option-creation,option-foreclosure,audience-signaling,dependency-creation,testing-boundaries,none-apparent)",
      "audience_targeting": "enum(domestic,regional,international,adversary,ally,neutral)"
    }
  }],
  "edges": [{
    "source_action_index": "integer",
    "target_action_index": "integer",
    "relation_type": "enum(causes,enables,prevents,responds-to,reinforces,contradicts,supports,coordinates-with,escalates-to,de-escalates-to,triggers,retaliates-for)",
    "strength": "enum(definite,likely,possible,speculative)"
  }]
}"""

# JSON schema for vLLM structured output constraint (generic models)
EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "grayzone_relevance": {"type": "string", "enum": ["high", "medium", "low", "none"]},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action_description": {"type": "string"},
                    "l1_domain": {"type": "string"},
                    "l2_subdomain": {"type": "string"},
                    "l3_code": {"type": "string"},
                    "escalation_classification": {"type": "string"},
                    "dominant_action_type": {"type": "string"},
                    "scope": {"type": "string"},
                    "claim_type": {"type": "string"},
                    "initiators": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "canonical_name": {"type": "string"},
                                "country_code": {"type": "string"},
                                "organisation_type": {"type": "string"},
                                "role": {"type": "string"},
                            },
                            "required": ["name"],
                        },
                    },
                    "targets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "canonical_name": {"type": "string"},
                                "country_code": {"type": "string"},
                                "organisation_type": {"type": "string"},
                                "role": {"type": "string"},
                            },
                            "required": ["name"],
                        },
                    },
                    "locations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "canonical_name": {"type": "string"},
                                "level": {"type": "string"},
                                "country_code": {"type": "string"},
                                "macro_region": {"type": "string"},
                            },
                            "required": ["name"],
                        },
                    },
                    "date_raw": {"type": "string"},
                    "date_iso": {"type": "string"},
                    "influence_patterns": {"type": "array", "items": {"type": "string"}},
                    "contestation_dynamics": {"type": "string"},
                    "state_indicators": {
                        "type": "object",
                        "properties": {
                            "actor_posture": {"type": "string"},
                            "regional_tension": {"type": "string"},
                            "prior_state_description": {"type": "string"},
                            "resulting_state_description": {"type": "string"},
                        },
                    },
                    "cost_profile": {
                        "type": "object",
                        "properties": {
                            "resource_type": {"type": "string"},
                            "cost_magnitude": {"type": "string"},
                            "risk_level": {"type": "string"},
                            "reversibility": {"type": "string"},
                        },
                    },
                    "implicit_signals": {
                        "type": "object",
                        "properties": {
                            "stated_justification": {"type": "string"},
                            "unstated_objective_hint": {"type": "string"},
                            "audience_targeting": {"type": "string"},
                        },
                    },
                },
                "required": ["action_description", "initiators", "targets"],
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_action_index": {"type": "integer"},
                    "target_action_index": {"type": "integer"},
                    "relation_type": {"type": "string"},
                    "strength": {"type": "string"},
                },
            },
        },
    },
    "required": ["grayzone_relevance", "actions"],
}

# Relevance-only schema for two-pass test
RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "grayzone_relevance": {"type": "string", "enum": ["high", "medium", "low", "none"]},
        "summary": {"type": "string"},
    },
    "required": ["grayzone_relevance"],
}

RELEVANCE_NUEXTRACT_TEMPLATE = '{"grayzone_relevance": "enum(high,medium,low,none)", "summary": "string"}'

# System prompt for generic models
SYSTEM_PROMPT = """You are an expert geopolitical event extraction system. Given a news article, extract structured information about geopolitical events into the exact JSON schema provided.

Instructions:
- Extract ALL distinct actions/events described in the article
- For each action, identify initiators (actors taking the action) and targets (actors affected)
- Use ISO country codes (e.g., CN, US, PH, RU)
- Dates should be in ISO format (YYYY-MM-DD) when possible
- l1_domain codes: D=Diplomatic, I=Informational, M=Military, E=Economic, F=Financial, IN=Intelligence, L=Law Enforcement, EN=Environmental
- Be precise about escalation classification and influence patterns
- For implicit_signals, reason about what is NOT explicitly stated but implied
- Extract inter-action edges when one action causes, enables, or responds to another

Output ONLY valid JSON matching the required schema. No explanatory text."""

RELEVANCE_SYSTEM_PROMPT = """You are a geopolitical relevance classifier. Given a news article, determine if it is relevant to geopolitical gray zone activities (state competition, military posturing, economic coercion, diplomatic pressure, information operations, etc.).

Output JSON with: grayzone_relevance (high/medium/low/none) and a one-sentence summary."""


# ---------------------------------------------------------------------------
# Test data preparation
# ---------------------------------------------------------------------------

def load_test_articles(s3_client, n=20):
    """Load test articles from NuExtract S3 output + fetch original text from URLs."""
    import http.client
    import urllib.parse
    from html.parser import HTMLParser

    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text_parts = []
            self._skip = False
            self._skip_tags = {"script", "style", "nav", "header", "footer", "aside"}

        def handle_starttag(self, tag, attrs):
            if tag in self._skip_tags:
                self._skip = True

        def handle_endtag(self, tag):
            if tag in self._skip_tags:
                self._skip = False

        def handle_data(self, data):
            if not self._skip:
                text = data.strip()
                if text:
                    self.text_parts.append(text)

        def get_text(self):
            return " ".join(self.text_parts)

    # Get URLs from NuExtract output
    print("[DATA] Loading URLs from NuExtract output files...")
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_INPUT_PREFIX, MaxKeys=200):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json"):
                keys.append(obj["Key"])

    import random
    random.seed(42)
    random.shuffle(keys)

    articles = []
    urls_seen = set()
    for key in keys[:30]:
        if len(articles) >= n:
            break
        try:
            resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(resp["Body"].read().decode("utf-8"))
            docs = data if isinstance(data, list) else [data]
            for doc in docs:
                if len(articles) >= n:
                    break
                if not doc.get("relevant"):
                    continue
                src = doc.get("_source", {})
                url = src.get("url", "") or doc.get("url", "")
                if not url or url in urls_seen:
                    continue
                urls_seen.add(url)
                crawl_date = src.get("crawl_date", "") or doc.get("crawl_date", "")
                title = src.get("title", "") or doc.get("title", "")
                existing_events = doc.get("events", [])

                # Fetch article text from URL
                try:
                    parsed = urllib.parse.urlparse(url)
                    if parsed.scheme not in ("http", "https"):
                        continue
                    Conn = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
                    conn = Conn(parsed.hostname, parsed.port, timeout=10)
                    path = parsed.path + ("?" + parsed.query if parsed.query else "")
                    conn.request("GET", path or "/", headers={
                        "User-Agent": "Mozilla/5.0 (compatible; research benchmark)"
                    })
                    resp_url = conn.getresponse()
                    html = resp_url.read().decode("utf-8", errors="replace")
                    conn.close()
                    extractor = TextExtractor()
                    extractor.feed(html)
                    text = extractor.get_text()
                    if len(text) < 200:
                        continue
                except Exception as e:
                    print(f"  [SKIP] {url[:60]}... fetch failed: {e}")
                    continue

                articles.append({
                    "url": url,
                    "crawl_date": crawl_date,
                    "title": title,
                    "text": text[:TEXT_TRUNCATION],
                    "text_len": len(text),
                    "existing_extraction": existing_events[:2],
                })
                print(f"  [{len(articles)}/{n}] {title[:60]}... ({len(text)} chars)")
        except Exception as e:
            print(f"  [ERROR] {key}: {e}")
            continue

    print(f"[DATA] Loaded {len(articles)} test articles")
    return articles


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

VALID_L1 = {"D", "I", "M", "E", "F", "IN", "L", "EN"}
VALID_ESCALATION = {"escalation", "de-escalation", "cooperation-building",
                    "cooperation-maintaining", "neutral", "ambiguous"}
VALID_ACTION_TYPE = {"confrontational", "cooperative", "diplomatic",
                     "informational", "economic"}
VALID_RELEVANCE = {"high", "medium", "low", "none"}


def score_extraction(parsed: dict, article: dict) -> dict:
    """Score extraction quality across multiple dimensions.

    Returns dict of metric_name -> score (0.0 to 1.0).
    """
    scores = {}

    # 1. Valid JSON (already parsed if we got here)
    scores["valid_json"] = 1.0

    # 2. Relevance field
    rel = parsed.get("grayzone_relevance", "")
    scores["has_relevance"] = 1.0 if rel in VALID_RELEVANCE else 0.0

    actions = parsed.get("actions", [])
    if not actions:
        scores["has_actions"] = 0.0
        scores["avg_action_quality"] = 0.0
        scores["total"] = sum(scores.values()) / max(len(scores), 1)
        return scores

    scores["has_actions"] = 1.0
    scores["action_count"] = min(len(actions), 5) / 5.0  # Normalize

    action_scores = []
    for action in actions[:5]:  # Score up to 5 actions
        ascore = {}

        # Action description
        desc = action.get("action_description", "")
        ascore["has_description"] = 1.0 if len(desc) > 20 else 0.0

        # L1 domain
        l1 = action.get("l1_domain", "")
        ascore["valid_l1"] = 1.0 if l1 in VALID_L1 else 0.0

        # L2 subdomain
        l2 = action.get("l2_subdomain", "")
        ascore["has_l2"] = 1.0 if l2 and len(l2) > 1 else 0.0

        # L3 code
        l3 = action.get("l3_code", "")
        ascore["has_l3"] = 1.0 if l3 and len(l3) > 3 else 0.0

        # Escalation
        esc = action.get("escalation_classification", "")
        ascore["valid_escalation"] = 1.0 if esc in VALID_ESCALATION else 0.0

        # Action type
        at = action.get("dominant_action_type", "")
        ascore["valid_action_type"] = 1.0 if at in VALID_ACTION_TYPE else 0.0

        # Initiators
        inits = action.get("initiators", [])
        has_named_inits = any(i.get("name", "") for i in inits if isinstance(i, dict))
        ascore["has_initiators"] = 1.0 if has_named_inits else 0.0

        has_country_codes = any(i.get("country_code", "") for i in inits if isinstance(i, dict))
        ascore["has_country_codes"] = 1.0 if has_country_codes else 0.0

        # Targets
        tgts = action.get("targets", [])
        has_named_tgts = any(t.get("name", "") for t in tgts if isinstance(t, dict))
        ascore["has_targets"] = 1.0 if has_named_tgts else 0.0

        # Locations
        locs = action.get("locations", [])
        has_locs = any(l.get("name", "") for l in locs if isinstance(l, dict))
        ascore["has_locations"] = 1.0 if has_locs else 0.0

        has_macro = any(l.get("macro_region", "") for l in locs if isinstance(l, dict))
        ascore["has_macro_region"] = 1.0 if has_macro else 0.0

        # Dates
        ascore["has_date_iso"] = 1.0 if action.get("date_iso", "") else 0.0
        ascore["has_date_raw"] = 1.0 if action.get("date_raw", "") else 0.0

        # Influence patterns
        patterns = action.get("influence_patterns", [])
        ascore["has_influence_patterns"] = 1.0 if patterns else 0.0

        # State indicators
        state = action.get("state_indicators", {})
        ascore["has_state_indicators"] = 1.0 if state.get("actor_posture") else 0.0
        ascore["has_state_description"] = 1.0 if state.get("resulting_state_description") else 0.0

        # Cost profile
        cost = action.get("cost_profile", {})
        ascore["has_cost_profile"] = 1.0 if cost.get("resource_type") else 0.0

        # Implicit signals (hardest field — tests reasoning)
        implicit = action.get("implicit_signals", {})
        ascore["has_justification"] = 1.0 if implicit.get("stated_justification") else 0.0
        ascore["has_unstated_objective"] = 1.0 if implicit.get("unstated_objective_hint") and implicit["unstated_objective_hint"] != "none-apparent" else 0.0

        action_scores.append(sum(ascore.values()) / max(len(ascore), 1))

    scores["avg_action_quality"] = sum(action_scores) / max(len(action_scores), 1)

    # Edges
    edges = parsed.get("edges", [])
    scores["has_edges"] = 1.0 if edges else 0.0

    # Overall
    scores["total"] = sum(scores.values()) / max(len(scores), 1)
    return scores


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------

def run_model(model_key: str, articles: list[dict]) -> dict:
    """Run a single model on all articles and return results."""
    config = MODELS[model_key]
    hf_name = config["hf_name"]
    model_type = config["type"]

    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {model_key} ({hf_name})")
    print(f"{'='*70}")

    # Import vLLM
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    # Load model
    print(f"  Loading model...")
    t0 = time.time()
    try:
        llm = LLM(
            model=hf_name,
            trust_remote_code=config.get("trust_remote_code", False),
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_len=config.get("max_model_len", 20480),
            enable_prefix_caching=True,
        )
    except Exception as e:
        print(f"  FAILED to load model: {e}")
        traceback.print_exc()
        return {"model": model_key, "error": str(e), "results": []}

    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    # Build sampling params
    if model_type == "nuextract":
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=MAX_NEW_TOKENS,
            structured_outputs=StructuredOutputsParams(json=EXTRACTION_JSON_SCHEMA),
        )
    else:
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=MAX_NEW_TOKENS,
            structured_outputs=StructuredOutputsParams(json=EXTRACTION_JSON_SCHEMA),
        )

    # Build messages for each article
    results = []
    all_messages = []
    for article in articles:
        text = article["text"]
        crawl_date = article.get("crawl_date", "")[:10]

        if model_type == "nuextract":
            messages = [{"role": "user", "content": text}]
        else:
            user_content = f"Article published: {crawl_date}\n\n{text}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        all_messages.append(messages)

    # Run batch inference
    print(f"  Running inference on {len(articles)} articles...")
    t0 = time.time()
    try:
        chat_kwargs = {}
        if model_type == "nuextract":
            chat_kwargs["chat_template_kwargs"] = {"template": NUEXTRACT_TEMPLATE}
        elif config.get("extra_chat_template_kwargs"):
            chat_kwargs["chat_template_kwargs"] = config["extra_chat_template_kwargs"]

        outputs = llm.chat(all_messages, sampling_params, **chat_kwargs)
    except Exception as e:
        print(f"  INFERENCE FAILED: {e}")
        traceback.print_exc()
        # Try one-by-one as fallback
        print(f"  Falling back to one-by-one inference...")
        outputs = []
        for i, msgs in enumerate(all_messages):
            try:
                out = llm.chat([msgs], sampling_params, **chat_kwargs)
                outputs.extend(out)
            except Exception as e2:
                print(f"    Article {i} failed: {e2}")
                outputs.append(None)

    inference_time = time.time() - t0
    print(f"  Inference completed in {inference_time:.1f}s "
          f"({inference_time/max(len(articles),1):.2f}s/article)")

    # Parse and score
    total_tokens = 0
    for i, output in enumerate(outputs):
        if output is None:
            results.append({
                "article_idx": i,
                "url": articles[i]["url"][:80],
                "error": "inference_failed",
                "scores": {"total": 0.0},
            })
            continue

        try:
            raw_text = output.outputs[0].text
            tokens_generated = len(output.outputs[0].token_ids)
            total_tokens += tokens_generated
            parsed = json.loads(raw_text)

            scores = score_extraction(parsed, articles[i])
            results.append({
                "article_idx": i,
                "url": articles[i]["url"][:80],
                "tokens_generated": tokens_generated,
                "scores": scores,
                "extraction": parsed,
            })
        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            results.append({
                "article_idx": i,
                "url": articles[i]["url"][:80],
                "error": f"parse_error: {e}",
                "raw_output": (output.outputs[0].text[:500]
                               if output and output.outputs else ""),
                "scores": {"total": 0.0, "valid_json": 0.0},
            })

    # Compute summary stats
    valid = [r for r in results if "error" not in r]
    error_count = len(results) - len(valid)

    avg_scores = {}
    if valid:
        all_score_keys = set()
        for r in valid:
            all_score_keys.update(r["scores"].keys())
        for key in sorted(all_score_keys):
            vals = [r["scores"].get(key, 0.0) for r in valid]
            avg_scores[key] = sum(vals) / len(vals)

    summary = {
        "model": model_key,
        "hf_name": hf_name,
        "model_type": model_type,
        "load_time_s": round(load_time, 1),
        "inference_time_s": round(inference_time, 1),
        "articles_tested": len(articles),
        "articles_successful": len(valid),
        "articles_failed": error_count,
        "avg_time_per_article_s": round(inference_time / max(len(articles), 1), 2),
        "total_tokens_generated": total_tokens,
        "tokens_per_second": round(total_tokens / max(inference_time, 0.1), 1),
        "avg_scores": avg_scores,
    }

    print(f"\n  Results for {model_key}:")
    print(f"    Successful: {len(valid)}/{len(articles)}")
    print(f"    Avg time/article: {summary['avg_time_per_article_s']}s")
    print(f"    Tokens/sec: {summary['tokens_per_second']}")
    print(f"    Avg quality scores:")
    for k, v in sorted(avg_scores.items()):
        print(f"      {k}: {v:.3f}")

    # Cleanup GPU memory
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {"summary": summary, "results": results}


def run_relevance_pass(articles: list[dict], model_key: str = "nuextract-4b") -> dict:
    """Run a relevance-only pass to test two-pass architecture."""
    config = MODELS[model_key]
    hf_name = config["hf_name"]

    print(f"\n{'='*70}")
    print(f"TWO-PASS TEST: Relevance filter with {model_key}")
    print(f"{'='*70}")

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    print(f"  Loading model for relevance pass...")
    t0 = time.time()
    llm = LLM(
        model=hf_name,
        trust_remote_code=config.get("trust_remote_code", False),
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=config.get("max_model_len", 20480),
        enable_prefix_caching=True,
    )
    load_time = time.time() - t0

    # Relevance-only schema — tiny output
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        structured_outputs=StructuredOutputsParams(json=RELEVANCE_SCHEMA),
    )

    all_messages = []
    for article in articles:
        if config["type"] == "nuextract":
            messages = [{"role": "user", "content": article["text"]}]
        else:
            messages = [
                {"role": "system", "content": RELEVANCE_SYSTEM_PROMPT},
                {"role": "user", "content": article["text"]},
            ]
        all_messages.append(messages)

    print(f"  Running relevance classification on {len(articles)} articles...")
    t0 = time.time()
    chat_kwargs = {}
    if config["type"] == "nuextract":
        chat_kwargs["chat_template_kwargs"] = {"template": RELEVANCE_NUEXTRACT_TEMPLATE}

    outputs = llm.chat(all_messages, sampling_params, **chat_kwargs)
    relevance_time = time.time() - t0

    # Parse results
    relevant_indices = []
    for i, output in enumerate(outputs):
        try:
            parsed = json.loads(output.outputs[0].text)
            rel = parsed.get("grayzone_relevance", "none")
            if rel in ("high", "medium"):
                relevant_indices.append(i)
        except Exception:
            relevant_indices.append(i)  # If unclear, include

    print(f"  Relevance pass: {len(relevant_indices)}/{len(articles)} marked relevant")
    print(f"  Relevance pass time: {relevance_time:.1f}s "
          f"({relevance_time/max(len(articles),1):.3f}s/article)")

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "relevance_time_s": round(relevance_time, 1),
        "total_articles": len(articles),
        "relevant_count": len(relevant_indices),
        "relevant_indices": relevant_indices,
        "time_per_article_s": round(relevance_time / max(len(articles), 1), 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Structured extraction model benchmark")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help="Models to benchmark")
    parser.add_argument("--articles", type=int, default=20,
                        help="Number of test articles")
    parser.add_argument("--two-pass", action="store_true", default=True,
                        help="Also test two-pass architecture")
    args = parser.parse_args()

    print("=" * 70)
    print("STRUCTURED EXTRACTION MODEL BENCHMARK")
    print(f"Date: {datetime.utcnow().isoformat()}")
    print(f"Models: {args.models}")
    print(f"Text truncation: {TEXT_TRUNCATION} chars")
    print("=" * 70)

    s3 = boto3.client("s3", region_name=REGION)

    # Load test articles
    articles = load_test_articles(s3, n=args.articles)
    if not articles:
        print("ERROR: No test articles loaded. Exiting.")
        sys.exit(1)

    # Save test articles to S3 for reproducibility
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"{S3_OUTPUT_PREFIX}test_articles.json",
        Body=json.dumps(articles, ensure_ascii=False),
        ContentType="application/json",
    )

    all_results = {}

    # Run each model
    for model_key in args.models:
        try:
            result = run_model(model_key, articles)
            all_results[model_key] = result

            # Write intermediate results to S3 after each model
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=f"{S3_OUTPUT_PREFIX}{model_key}_results.json",
                Body=json.dumps(result, ensure_ascii=False, default=str),
                ContentType="application/json",
            )
            print(f"\n  Saved results to s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}{model_key}_results.json")
        except Exception as e:
            print(f"\n  MODEL {model_key} FAILED: {e}")
            traceback.print_exc()
            all_results[model_key] = {"error": str(e)}

    # Two-pass test: relevance filter with nuextract-4b, then full extraction with best model
    if args.two_pass and "nuextract-4b" in args.models:
        try:
            relevance_result = run_relevance_pass(articles, "nuextract-4b")
            all_results["two_pass_relevance"] = relevance_result
        except Exception as e:
            print(f"\n  TWO-PASS relevance failed: {e}")
            traceback.print_exc()

    # Final comparison report
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"\n{'Model':<20} {'Load(s)':<8} {'Infer(s)':<9} {'tok/s':<8} "
          f"{'Quality':<8} {'Actions':<8} {'Implicit':<8} {'Errors':<6}")
    print("-" * 85)

    for model_key in args.models:
        r = all_results.get(model_key, {})
        if "error" in r and "summary" not in r:
            print(f"{model_key:<20} {'FAILED'}")
            continue
        s = r.get("summary", {})
        avg = s.get("avg_scores", {})
        print(f"{model_key:<20} {s.get('load_time_s','?'):<8} "
              f"{s.get('inference_time_s','?'):<9} "
              f"{s.get('tokens_per_second','?'):<8} "
              f"{avg.get('total',0):<8.3f} "
              f"{avg.get('avg_action_quality',0):<8.3f} "
              f"{avg.get('has_unstated_objective',0) if isinstance(avg.get('has_unstated_objective',0), float) else 0:<8.3f} "
              f"{s.get('articles_failed',0):<6}")

    if "two_pass_relevance" in all_results:
        rr = all_results["two_pass_relevance"]
        print(f"\nTwo-pass relevance: {rr.get('relevant_count')}/{rr.get('total_articles')} "
              f"relevant, {rr.get('time_per_article_s')}s/article")

    # Write final results
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"{S3_OUTPUT_PREFIX}all_results.json",
        Body=json.dumps(all_results, ensure_ascii=False, default=str),
        ContentType="application/json",
    )
    print(f"\nAll results saved to s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}all_results.json")


if __name__ == "__main__":
    main()
