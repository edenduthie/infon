#!/usr/bin/env python3
"""Adversarial test suite for the cognition extraction pipeline.

Tests polarity detection, double negation, contradictory pairs, malformed
input resilience, and edge-case handling. Prints a structured report with
per-corpus summaries, specific failures, and improvement recommendations.

Usage:
    cd /Users/cpro/Desktop/ontology-workshop && python tests/test_adversarial.py
"""

from __future__ import annotations

import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

CORPUS_A: list[dict] = [
    # --- Clear negation (should be polarity=0) ---
    {"text": "Russia did not deploy troops to the border.", "id": "a01",
     "expected_polarity": 0, "label": "clear-neg"},
    {"text": "China has not invested in African technology.", "id": "a02",
     "expected_polarity": 0, "label": "clear-neg"},
    {"text": "The US didn't sanction Russia over the nuclear deal.", "id": "a03",
     "expected_polarity": 0, "label": "clear-neg"},
    {"text": "Iran hasn't deployed military forces to the maritime border.", "id": "a04",
     "expected_polarity": 0, "label": "clear-neg"},
    {"text": "NATO won't cooperate with Russia on European security.", "id": "a05",
     "expected_polarity": 0, "label": "clear-neg"},

    # --- Hedged negation (should be polarity=0) ---
    {"text": "China has refused to negotiate on trade.", "id": "a06",
     "expected_polarity": 0, "label": "hedged-neg",
     "note": "refused implies negation but no negation cue word present"},
    {"text": "India declined to invest in the European technology fund.", "id": "a07",
     "expected_polarity": 0, "label": "hedged-neg",
     "note": "declined has no regex cue"},
    {"text": "Japan halted all trade with Russia over the military conflict.", "id": "a08",
     "expected_polarity": 0, "label": "hedged-neg",
     "note": "halted implies negation with no cue word"},

    # --- Implicit negation (should be polarity=0) ---
    {"text": "The sanctions failed to restrict nuclear development.", "id": "a09",
     "expected_polarity": 0, "label": "implicit-neg",
     "note": "failed to = negation, but no regex cue word"},
    {"text": "The EU lacks the ability to deploy humanitarian aid to Africa.", "id": "a10",
     "expected_polarity": 0, "label": "implicit-neg",
     "note": "lacks = negation, no cue word"},

    # --- Double subject negation (should be polarity=0) ---
    {"text": "Neither Iran nor Russia condemned the attack.", "id": "a11",
     "expected_polarity": 0, "label": "double-subj-neg"},
    {"text": "Neither China nor India invested in European military technology.", "id": "a12",
     "expected_polarity": 0, "label": "double-subj-neg"},

    # --- Scope traps: negation word present but meaning is AFFIRMED ---
    {"text": "Not only did the US invest, they deployed military assets to East Asia.", "id": "a13",
     "expected_polarity": 1, "label": "scope-trap",
     "note": "not only = affirmation, but regex sees 'not'"},
    {"text": "No one can deny that China has invested heavily in African technology.", "id": "a14",
     "expected_polarity": 1, "label": "scope-trap",
     "note": "'no' and 'deny' present, but meaning is affirmed"},
    {"text": "It is not surprising that Russia deployed troops to the border.", "id": "a15",
     "expected_polarity": 1, "label": "scope-trap",
     "note": "'not surprising' = affirmation, regex sees 'not'"},
]

CORPUS_B: list[dict] = [
    # Double negation / litotes: two negation cues cancel out => AFFIRMED
    {"text": "Iran has not denied pursuing nuclear technology.", "id": "b01",
     "expected_polarity": 1, "label": "double-neg",
     "note": "not + denied = affirmed"},
    {"text": "The EU cannot ignore humanitarian concerns in the Middle East.", "id": "b02",
     "expected_polarity": 1, "label": "double-neg",
     "note": "cannot ignore = must address"},
    {"text": "It is not impossible that China will invest in African technology.", "id": "b03",
     "expected_polarity": 1, "label": "double-neg",
     "note": "not impossible = possible/tentative affirm"},
    {"text": "NATO never fails to deploy when territory is threatened.", "id": "b04",
     "expected_polarity": 1, "label": "double-neg",
     "note": "never fails = always succeeds"},
    {"text": "Russia is not unwilling to negotiate on nuclear matters.", "id": "b05",
     "expected_polarity": 1, "label": "double-neg",
     "note": "not unwilling = willing"},
    {"text": "The US hasn't stopped investing in European military technology.", "id": "b06",
     "expected_polarity": 1, "label": "double-neg",
     "note": "hasn't stopped = continues"},
    {"text": "India cannot afford not to cooperate with Japan on maritime security.", "id": "b07",
     "expected_polarity": 1, "label": "double-neg",
     "note": "cannot afford not to = must cooperate"},
    {"text": "There is no lack of evidence that Iran attacked the maritime convoy.", "id": "b08",
     "expected_polarity": 1, "label": "double-neg",
     "note": "no lack = plenty of evidence"},
    {"text": "China didn't deny deploying naval forces to the strait.", "id": "b09",
     "expected_polarity": 1, "label": "double-neg",
     "note": "didn't deny = effectively confirmed"},
    {"text": "It wasn't that Russia never invested in African energy infrastructure.", "id": "b10",
     "expected_polarity": 1, "label": "double-neg",
     "note": "wasn't + never = did invest"},
]

CORPUS_C_PAIRS: list[tuple[dict, dict]] = [
    (
        {"text": "Russia deployed troops to the border.", "id": "c01a",
         "expected_polarity": 1, "label": "affirm"},
        {"text": "Russia withdrew all troops from the border.", "id": "c01b",
         "expected_polarity": 1, "label": "affirm-opposite",
         "note": "semantically opposite but no negation word"},
    ),
    (
        {"text": "China invested heavily in African infrastructure.", "id": "c02a",
         "expected_polarity": 1, "label": "affirm"},
        {"text": "China pulled all investment from Africa.", "id": "c02b",
         "expected_polarity": 1, "label": "affirm-opposite"},
    ),
    (
        {"text": "The US sanctioned Iran over nuclear proliferation.", "id": "c03a",
         "expected_polarity": 1, "label": "affirm"},
        {"text": "The US lifted sanctions on Iran regarding nuclear energy.", "id": "c03b",
         "expected_polarity": 1, "label": "affirm-opposite"},
    ),
    (
        {"text": "NATO cooperated with the EU on European defense.", "id": "c04a",
         "expected_polarity": 1, "label": "affirm"},
        {"text": "NATO ended all cooperation with the EU on defense.", "id": "c04b",
         "expected_polarity": 1, "label": "affirm-opposite"},
    ),
    (
        {"text": "Iran condemned the Israeli military operation.", "id": "c05a",
         "expected_polarity": 1, "label": "affirm"},
        {"text": "Iran praised the Israeli military operation.", "id": "c05b",
         "expected_polarity": 1, "label": "affirm-opposite"},
    ),
]

CORPUS_D: list[dict] = [
    # Empty / near-empty
    {"text": "", "id": "d01", "label": "empty"},
    {"text": " ", "id": "d02", "label": "whitespace-only"},
    {"text": "Nuclear", "id": "d03", "label": "single-word"},
    {"text": "attack", "id": "d04", "label": "single-word-relation"},

    # Fragments
    {"text": "The deployment of", "id": "d05", "label": "fragment"},
    {"text": "sanctions on the", "id": "d06", "label": "fragment"},

    # Off-domain noise
    {"text": "The recipe calls for two cups of flour and one egg.", "id": "d07",
     "label": "off-domain"},
    {"text": "My cat likes to sleep on the couch on sunny afternoons.", "id": "d08",
     "label": "off-domain"},

    # Garbled
    {"text": "China invest not the sanctions deploy nuclear.", "id": "d09",
     "label": "garbled"},
    {"text": "Russia attack trade cooperate Iran EU condemn deploy.", "id": "d10",
     "label": "garbled-keyword-soup"},

    # Very long run-on sentence (300+ words)
    {"text": (
        "The geopolitical landscape of the twenty-first century is shaped by an "
        "intricate web of alliances and rivalries where the United States seeks to "
        "maintain its influence across Europe and the Indo-Pacific while China "
        "continues to expand its Belt and Road Initiative investing billions in "
        "African and Asian infrastructure projects that challenge Western economic "
        "dominance and Russia under its current leadership has embarked on military "
        "deployments along its western border raising concerns among NATO allies "
        "who in turn have increased defense spending and deployed additional troops "
        "to Eastern European member states while Iran pursues its nuclear program "
        "despite international sanctions and Israel conducts military operations "
        "that draw condemnation from the United Nations and humanitarian "
        "organizations worldwide and India and Japan are strengthening their "
        "maritime partnership in the Indo-Pacific to counter Chinese naval "
        "expansion in the South China Sea and the European Union is negotiating "
        "trade agreements with African nations while simultaneously imposing "
        "technology export controls on countries deemed security risks and the "
        "Middle East remains a theater of proxy conflicts where multiple state and "
        "non-state actors compete for influence territory and resources with "
        "humanitarian crises in Yemen and Syria continuing to displace millions "
        "while diplomatic talks in Geneva and Vienna attempt to broker ceasefires "
        "and peace agreements that have thus far failed to produce lasting results "
        "despite the efforts of international mediators and regional stakeholders "
        "who recognize that without cooperation and compromise the cycle of "
        "conflict and suffering will continue indefinitely across the region and "
        "beyond affecting global trade maritime shipping routes and the stability "
        "of energy markets that the entire world depends upon for economic growth "
        "and development."
    ), "id": "d11", "label": "long-run-on"},

    # Numbers only
    {"text": "2024 15.7 Q3", "id": "d12", "label": "numbers-only"},

    # Unicode / emoji
    {"text": "\U0001f680 Russia launches \U0001f3af", "id": "d13", "label": "emoji"},

    # Repeated words
    {"text": "attack attack attack attack attack", "id": "d14",
     "label": "repetition"},

    # Mixed signals
    {"text": "Israel cooperated on the attack against humanitarian targets.",
     "id": "d15", "label": "mixed-signals",
     "note": "cooperated + attack + humanitarian = confusing roles"},
]

CORPUS_E: list[dict] = [
    # Passive voice
    {"text": "Nuclear technology was deployed by Iran.", "id": "e01",
     "expected_subj": "iran", "expected_pred": "deploy",
     "expected_obj_contains": "nuclear",
     "label": "passive",
     "note": "subject and object may be swapped in passive"},

    # Nested clauses
    {"text": "The US, which had sanctioned Russia, invested in European military technology.",
     "id": "e02", "label": "nested-clause",
     "note": "two events: US sanction Russia + US invest in tech"},

    # Conditional
    {"text": "If China were to attack Taiwan, NATO would deploy.", "id": "e03",
     "label": "conditional",
     "note": "hypothetical, tense=conditional expected"},

    # Reported speech
    {"text": "Russia claimed it did not attack civilian targets.", "id": "e04",
     "expected_polarity": 0, "label": "reported-speech",
     "note": "'did not' triggers negation regex -- but it's reported speech"},

    # Questions
    {"text": "Did Iran invest in nuclear technology?", "id": "e05",
     "label": "question",
     "note": "question, not assertion -- should ideally not produce affirmed infons"},

    # Relative clause
    {"text": "India, which cooperates with Japan on maritime security, condemned the attack on the strait.",
     "id": "e06", "label": "relative-clause"},

    # Appositive
    {"text": "Beijing, China's capital, announced new sanctions on European technology exports.",
     "id": "e07", "label": "appositive"},

    # Temporal distance
    {"text": "Last year the EU invested in African humanitarian projects, but this year they sanctioned several governments.",
     "id": "e08", "label": "temporal-shift"},

    # Ambiguous subject
    {"text": "They deployed nuclear submarines to the Pacific.",
     "id": "e09", "label": "ambiguous-pronoun",
     "note": "pronoun 'they' has no referent"},

    # Nominalisation
    {"text": "The Russian investment in Indian military technology surprised European analysts.",
     "id": "e10", "label": "nominalisation",
     "note": "nominalised verb 'investment', no active predicate"},
]


# ---------------------------------------------------------------------------
# Report data structures
# ---------------------------------------------------------------------------

@dataclass
class InfonSummary:
    """Lightweight view of an extracted infon for reporting."""
    subject: str
    predicate: str
    object: str
    polarity: int
    confidence: float
    sentence: str

    def triple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


@dataclass
class CorpusResult:
    name: str
    total_docs: int = 0
    total_infons: int = 0
    affirmed: int = 0
    negated: int = 0
    infons: list[InfonSummary] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_corpus(cog, docs: list[dict], corpus_name: str) -> CorpusResult:
    """Ingest documents one-at-a-time (fresh DB each call) and collect results."""
    from cognition import Cognition, CognitionConfig
    from cognition.extract import extract_infons

    result = CorpusResult(name=corpus_name, total_docs=len(docs))

    # Use the pipeline's extract_infons directly so we can inspect infons
    # before they go into the store (avoid contamination between corpora).
    infons, _edges = extract_infons(docs, cog.encoder, cog.schema, cog.config)
    result.total_infons = len(infons)
    for inf in infons:
        s = InfonSummary(
            subject=inf.subject,
            predicate=inf.predicate,
            object=inf.object,
            polarity=inf.polarity,
            confidence=inf.confidence,
            sentence=inf.sentence,
        )
        result.infons.append(s)
        if inf.polarity == 1:
            result.affirmed += 1
        else:
            result.negated += 1

    return result


def banner(text: str, char: str = "=", width: int = 78) -> str:
    return f"\n{char * width}\n  {text}\n{char * width}"


def table_row(cols: list[str], widths: list[int]) -> str:
    parts = []
    for c, w in zip(cols, widths):
        parts.append(str(c).ljust(w))
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Analysis per corpus
# ---------------------------------------------------------------------------

def analyse_a(result: CorpusResult, docs: list[dict]) -> None:
    """Corpus A: polarity accuracy check."""
    by_id = {}
    for inf in result.infons:
        by_id.setdefault(inf.sentence, []).append(inf)

    for doc in docs:
        sent = doc["text"]
        expected = doc.get("expected_polarity")
        label = doc.get("label", "")
        note = doc.get("note", "")
        infons_for = by_id.get(sent, [])

        if not infons_for:
            result.notes.append(f"[{doc['id']}] No infons extracted: {sent[:60]}...")
            continue

        for inf in infons_for:
            if expected is not None and inf.polarity != expected:
                result.failures.append(
                    f"[{doc['id']}] POLARITY MISMATCH ({label}): "
                    f"expected={expected} got={inf.polarity} | "
                    f"({inf.subject},{inf.predicate},{inf.object}) "
                    f"-- {sent[:80]}..."
                    + (f" NOTE: {note}" if note else "")
                )


def analyse_b(result: CorpusResult, docs: list[dict]) -> None:
    """Corpus B: double negation -- all should be polarity=1 but regex gives 0."""
    by_id = {}
    for inf in result.infons:
        by_id.setdefault(inf.sentence, []).append(inf)

    for doc in docs:
        sent = doc["text"]
        expected = doc.get("expected_polarity", 1)
        infons_for = by_id.get(sent, [])

        if not infons_for:
            result.notes.append(f"[{doc['id']}] No infons extracted: {sent[:60]}...")
            continue

        for inf in infons_for:
            if inf.polarity != expected:
                result.failures.append(
                    f"[{doc['id']}] DOUBLE-NEG FAIL: expected={expected} got={inf.polarity} | "
                    f"({inf.subject},{inf.predicate},{inf.object}) -- "
                    f"{sent[:80]}... NOTE: {doc.get('note', '')}"
                )


def analyse_c(result: CorpusResult, pairs: list[tuple[dict, dict]]) -> None:
    """Corpus C: contradictory pairs -- check if same SPO appears with both polarities."""
    by_sent = defaultdict(list)
    for inf in result.infons:
        by_sent[inf.sentence].append(inf)

    for (doc_a, doc_b) in pairs:
        triples_a = {inf.triple() for inf in by_sent.get(doc_a["text"], [])}
        triples_b = {inf.triple() for inf in by_sent.get(doc_b["text"], [])}
        overlap = triples_a & triples_b
        if overlap:
            pols_a = {t: [i.polarity for i in by_sent[doc_a["text"]] if i.triple() == t] for t in overlap}
            pols_b = {t: [i.polarity for i in by_sent[doc_b["text"]] if i.triple() == t] for t in overlap}
            for t in overlap:
                result.notes.append(
                    f"[{doc_a['id']}/{doc_b['id']}] Same triple {t} in both sentences: "
                    f"polA={pols_a[t]} polB={pols_b[t]}"
                )
                if set(pols_a[t]) == set(pols_b[t]):
                    result.failures.append(
                        f"[{doc_a['id']}/{doc_b['id']}] CONTRADICTION NOT CAPTURED: "
                        f"triple {t} has same polarity in both contradictory sentences"
                    )

        # Also note if either sentence produced no infons
        if not by_sent.get(doc_a["text"]):
            result.notes.append(f"[{doc_a['id']}] No infons from: {doc_a['text'][:60]}")
        if not by_sent.get(doc_b["text"]):
            result.notes.append(f"[{doc_b['id']}] No infons from: {doc_b['text'][:60]}")


def analyse_d(result: CorpusResult, docs: list[dict]) -> None:
    """Corpus D: broken / malformed -- ideally extract nothing or low confidence."""
    by_sent = defaultdict(list)
    for inf in result.infons:
        by_sent[inf.sentence].append(inf)

    for doc in docs:
        sent = doc["text"]
        label = doc["label"]
        infons_for = by_sent.get(sent, [])

        if label in ("empty", "whitespace-only", "numbers-only", "off-domain"):
            if infons_for:
                result.failures.append(
                    f"[{doc['id']}] SPURIOUS EXTRACTION ({label}): "
                    f"{len(infons_for)} infon(s) from: {sent[:60]!r}"
                )
            else:
                result.notes.append(
                    f"[{doc['id']}] Correctly produced 0 infons ({label})"
                )
        elif label in ("single-word", "single-word-relation", "fragment"):
            if infons_for:
                result.notes.append(
                    f"[{doc['id']}] Extracted {len(infons_for)} infon(s) from fragment ({label}): {sent!r}"
                )
            else:
                result.notes.append(
                    f"[{doc['id']}] Correctly produced 0 infons ({label})"
                )
        elif label == "long-run-on":
            result.notes.append(
                f"[{doc['id']}] Long run-on: extracted {len(infons_for)} infon(s)"
            )
        elif label == "garbled" or label == "garbled-keyword-soup":
            if infons_for:
                confs = [i.confidence for i in infons_for]
                result.notes.append(
                    f"[{doc['id']}] Garbled text produced {len(infons_for)} infon(s) "
                    f"(conf range: {min(confs):.3f}-{max(confs):.3f})"
                )
        elif label == "emoji":
            if infons_for:
                result.notes.append(
                    f"[{doc['id']}] Emoji text produced {len(infons_for)} infon(s): "
                    f"{[(i.subject, i.predicate, i.object) for i in infons_for]}"
                )
        elif label == "repetition":
            if infons_for:
                result.notes.append(
                    f"[{doc['id']}] Repetition produced {len(infons_for)} infon(s)"
                )
        elif label == "mixed-signals":
            if infons_for:
                roles = [(i.subject, i.predicate, i.object) for i in infons_for]
                result.notes.append(
                    f"[{doc['id']}] Mixed signals: triples = {roles}"
                )


def analyse_e(result: CorpusResult, docs: list[dict]) -> None:
    """Corpus E: edge cases -- role assignment, tense, question handling."""
    by_sent = defaultdict(list)
    for inf in result.infons:
        by_sent[inf.sentence].append(inf)

    for doc in docs:
        sent = doc["text"]
        label = doc["label"]
        note = doc.get("note", "")
        infons_for = by_sent.get(sent, [])

        if not infons_for:
            result.notes.append(f"[{doc['id']}] No infons extracted ({label}): {sent[:60]}...")
            continue

        # Passive voice check
        if label == "passive":
            expected_subj = doc.get("expected_subj", "")
            expected_pred = doc.get("expected_pred", "")
            expected_obj_contains = doc.get("expected_obj_contains", "")
            for inf in infons_for:
                subj_ok = (inf.subject == expected_subj) if expected_subj else True
                pred_ok = (inf.predicate == expected_pred) if expected_pred else True
                obj_ok = (expected_obj_contains in inf.object) if expected_obj_contains else True
                if not (subj_ok and pred_ok and obj_ok):
                    result.failures.append(
                        f"[{doc['id']}] PASSIVE ROLE MISMATCH: got ({inf.subject},{inf.predicate},{inf.object}), "
                        f"expected subj={expected_subj} pred={expected_pred} obj~={expected_obj_contains} "
                        f"-- {note}"
                    )

        # Conditional check
        elif label == "conditional":
            for inf in infons_for:
                result.notes.append(
                    f"[{doc['id']}] Conditional: ({inf.subject},{inf.predicate},{inf.object}) "
                    f"pol={inf.polarity} conf={inf.confidence:.3f}"
                )

        # Reported speech
        elif label == "reported-speech":
            for inf in infons_for:
                if inf.polarity == 0:
                    result.notes.append(
                        f"[{doc['id']}] Reported speech correctly triggers negation regex: "
                        f"({inf.subject},{inf.predicate},{inf.object}) pol={inf.polarity}"
                    )

        # Question
        elif label == "question":
            for inf in infons_for:
                result.notes.append(
                    f"[{doc['id']}] Question produced infon: ({inf.subject},{inf.predicate},{inf.object}) "
                    f"pol={inf.polarity} conf={inf.confidence:.3f} "
                    f"-- ideally questions should not produce affirmed infons"
                )

        else:
            for inf in infons_for:
                result.notes.append(
                    f"[{doc['id']}] ({label}): ({inf.subject},{inf.predicate},{inf.object}) "
                    f"pol={inf.polarity} conf={inf.confidence:.3f}"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(banner("ADVERSARIAL TEST SUITE -- Cognition Extraction Pipeline"))
    print()

    # ---- Initialise cognition ----
    print("Loading cognition pipeline...")
    from cognition import Cognition, CognitionConfig

    config = CognitionConfig(
        schema_path="data/tui_schema.json",
        db_path=":memory:",
    )
    cog = Cognition(config)
    print(f"  Encoder: {cog.encoder.model_name}")
    print(f"  Anchors: {len(cog.schema.names)}")
    print(f"  Backend: {config.backend}")
    print()

    results: list[CorpusResult] = []

    # ---- Corpus A: Simple negation ----
    print("Running Corpus A (Simple negation)...")
    res_a = run_corpus(cog, CORPUS_A, "A: Simple Negation")
    analyse_a(res_a, CORPUS_A)
    results.append(res_a)

    # ---- Corpus B: Double negation / litotes ----
    print("Running Corpus B (Double negation / litotes)...")
    res_b = run_corpus(cog, CORPUS_B, "B: Double Negation")
    analyse_b(res_b, CORPUS_B)
    results.append(res_b)

    # ---- Corpus C: Contradictory pairs ----
    print("Running Corpus C (Contradictory pairs)...")
    all_c_docs = []
    for a, b in CORPUS_C_PAIRS:
        all_c_docs.extend([a, b])
    res_c = run_corpus(cog, all_c_docs, "C: Contradictory Pairs")
    analyse_c(res_c, CORPUS_C_PAIRS)
    results.append(res_c)

    # ---- Corpus D: Broken / malformed ----
    print("Running Corpus D (Broken / malformed)...")
    res_d = run_corpus(cog, CORPUS_D, "D: Broken / Malformed")
    analyse_d(res_d, CORPUS_D)
    results.append(res_d)

    # ---- Corpus E: Edge cases ----
    print("Running Corpus E (Edge cases)...")
    res_e = run_corpus(cog, CORPUS_E, "E: Edge Cases")
    analyse_e(res_e, CORPUS_E)
    results.append(res_e)

    # ==================================================================
    # REPORT
    # ==================================================================
    print(banner("RESULTS SUMMARY"))

    # Per-corpus summary table
    widths = [28, 8, 10, 10, 10, 10]
    header = table_row(["Corpus", "Docs", "Infons", "Affirmed", "Negated", "Failures"], widths)
    print(f"\n{header}")
    print("  ".join(["-" * w for w in widths]))

    total_failures = 0
    for r in results:
        total_failures += len(r.failures)
        print(table_row([
            r.name,
            str(r.total_docs),
            str(r.total_infons),
            str(r.affirmed),
            str(r.negated),
            str(len(r.failures)),
        ], widths))

    print()

    # ---- Per-corpus details ----
    for r in results:
        print(banner(f"CORPUS: {r.name}", char="-"))

        if r.failures:
            print(f"\n  FAILURES ({len(r.failures)}):")
            for f in r.failures:
                wrapped = textwrap.fill(f, width=100, initial_indent="    * ",
                                        subsequent_indent="      ")
                print(wrapped)

        if r.notes:
            print(f"\n  NOTES ({len(r.notes)}):")
            for n in r.notes:
                wrapped = textwrap.fill(n, width=100, initial_indent="    - ",
                                        subsequent_indent="      ")
                print(wrapped)

        if not r.failures and not r.notes:
            print("\n  (no issues)")

    # ---- Overall assessment ----
    print(banner("OVERALL ASSESSMENT"))

    # Count specific failure categories
    scope_trap_fails = sum(1 for r in results for f in r.failures if "scope-trap" in f.lower() or "SCOPE" in f)
    double_neg_fails = sum(1 for r in results for f in r.failures if "DOUBLE-NEG" in f)
    hedged_neg_fails = sum(1 for r in results for f in r.failures if "hedged-neg" in f.lower())
    implicit_neg_fails = sum(1 for r in results for f in r.failures if "implicit-neg" in f.lower())
    passive_fails = sum(1 for r in results for f in r.failures if "PASSIVE" in f)
    spurious_fails = sum(1 for r in results for f in r.failures if "SPURIOUS" in f)
    contradiction_fails = sum(1 for r in results for f in r.failures if "CONTRADICTION" in f)

    print(f"""
  Total test documents:       {sum(r.total_docs for r in results)}
  Total infons extracted:     {sum(r.total_infons for r in results)}
  Total failures:             {total_failures}

  Failure breakdown:
    Scope-trap (false neg):   {scope_trap_fails}  -- 'not only' etc. incorrectly negated
    Double negation:          {double_neg_fails}  -- litotes/double-neg read as negated
    Hedged negation:          {hedged_neg_fails}  -- 'refused/declined' missed
    Implicit negation:        {implicit_neg_fails}  -- 'failed to/lacks' missed
    Passive role swap:        {passive_fails}  -- subject/object reversed
    Spurious extraction:      {spurious_fails}  -- infons from noise/empty input
    Contradiction undetected: {contradiction_fails}  -- same triple, same polarity in opposite sentences
""")

    # Pass/fail
    # We expect the current regex-based polarity to fail on scope traps and
    # double negation. The question is how badly.
    critical = spurious_fails  # extracting from empty/noise is a data-quality risk
    if total_failures == 0:
        verdict = "PASS -- all tests passed (unexpected for regex polarity!)"
    elif critical > 0:
        verdict = f"FAIL (CRITICAL) -- {critical} spurious extraction(s) from noise/empty input"
    elif total_failures <= 5:
        verdict = f"CONDITIONAL PASS -- {total_failures} known-limitation failure(s)"
    else:
        verdict = f"FAIL -- {total_failures} failure(s) found"

    print(f"  VERDICT: {verdict}")

    # ---- Recommendations ----
    print(banner("RECOMMENDATIONS FOR PIPELINE IMPROVEMENTS"))
    recs = []

    if scope_trap_fails > 0:
        recs.append(
            "1. SCOPE-AWARE NEGATION: Replace the flat regex negation detector with "
            "a dependency-parse-based approach (e.g., spaCy) that checks whether the "
            "negation word actually governs the main predicate. Phrases like 'not only', "
            "'not surprising that', and 'no one can deny' should not flip polarity."
        )
    if double_neg_fails > 0:
        recs.append(
            "2. DOUBLE-NEGATION HANDLING: Count negation cues per clause. An even "
            "number of negations should resolve to polarity=1 (affirmed). Consider a "
            "cue-counting heuristic or a lightweight NLI model for polarity classification."
        )
    if hedged_neg_fails + implicit_neg_fails > 0:
        recs.append(
            "3. EXPANDED NEGATION LEXICON: Add implicit negation verbs to the cue list: "
            "'refused', 'declined', 'halted', 'failed', 'lacks', 'unable', 'ceased', "
            "'prevented', 'blocked', 'rejected'. Or use a sentiment/negation lexicon."
        )
    if passive_fails > 0:
        recs.append(
            "4. PASSIVE VOICE DETECTION: The current pipeline assigns S/P/O roles based "
            "on schema type (actor->subject, relation->predicate, feature->object), not "
            "syntactic role. Add a passive-voice detector to swap subject/object when the "
            "sentence uses passive construction."
        )
    if contradiction_fails > 0:
        recs.append(
            "5. SEMANTIC CONTRADICTION DETECTION: Pairs like 'deployed troops' / 'withdrew "
            "troops' produce the same (S,P,O) triple with the same polarity. Add antonym "
            "awareness (deploy vs. withdraw, invest vs. divest) or leverage SPLADE "
            "activation differences to flag contradictions during consolidation."
        )
    if spurious_fails > 0:
        recs.append(
            "6. INPUT VALIDATION: Add a pre-filter that rejects documents with too few "
            "tokens, no schema-relevant content, or below a minimum sentence length. "
            "This prevents noise from polluting the knowledge graph."
        )

    # Always recommend
    recs.append(
        f"{'7' if recs else '1'}. QUESTION / CONDITIONAL MODALITY: Currently questions and "
        "conditionals produce infons with the same polarity as declarative statements. "
        "Add a modality field (declarative, interrogative, conditional, imperative) "
        "and filter or down-weight non-declarative infons."
    )

    for rec in recs:
        wrapped = textwrap.fill(rec, width=90, initial_indent="  ",
                                subsequent_indent="    ")
        print(wrapped)
        print()

    print(banner("END OF ADVERSARIAL TEST REPORT"))

    # Exit code based on critical failures
    sys.exit(1 if critical > 0 else 0)


if __name__ == "__main__":
    main()
