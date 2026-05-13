"""Generate a synthetic R&D memo corpus for the Chimera workshop.

Each memo is a short 1-3 sentence internal note, a patent excerpt, or a
customer request. We embed *gate outcomes* directly so downstream modules
can check whether the extraction pipeline recovers them.

Output
------
data/memos.jsonl         — {id, text, timestamp, memo_type}
data/ground_truth_gates.jsonl
                          — {memo_id, gate1, gate2, gate3, gate4, chimera_candidate}
data/gold_triples.jsonl  — {memo_id, subject, predicate, object}
                           (what the extractor *should* produce, for eval)

Design notes
------------
- ~300 memos across 12 months
- mix of 4 memo types: r&d note / customer request / patent filing /
  gate review
- ~25% of ideas pass all 4 gates (funded)
- ~30% fail Gate 1 (no market) — these don't become chimera candidates
- ~25% pass Gate 1 but fail Gate 3 (niche volume) — THESE are the
  chimera candidates
- ~20% fail other gates
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(7)

HERE = Path(__file__).parent.parent
DATA = HERE / "data"
SCHEMA = json.loads((DATA / "schema.json").read_text())

# ─── entity pools ─────────────────────────────────────────────────────────
MANUFACTURERS = ["dow", "huntsman", "basf", "3m", "henkel", "ashland", "wacker"]
DOWNSTREAM    = ["medtech_co", "aerospace_co", "auto_oem",
                 "startup_biomat", "startup_coatings"]
DOW_LABS      = ["dow_midland_lab", "dow_polymers_lab", "dow_bio_lab"]
REGULATORS    = ["fda", "reach", "epa"]

CHEMISTRIES   = ["epoxy", "polyurethane", "silicone", "acrylate",
                 "cyanoacrylate", "pressure_sensitive", "hot_melt",
                 "bio_adhesive", "tissue_glue"]
CURE_TYPES    = ["uv_cure", "moisture_cure", "two_part_cure"]
SUSTAIN_PROPS = ["biodegradable", "low_voc", "recyclable"]
MECH_PROPS    = ["high_tg", "low_shrinkage", "high_shear", "fatigue_resist"]
REG_PROPS     = ["fda_cleared", "reach_compliant", "iso10993"]

MARKETS       = ["med_device_market", "aerospace_market", "auto_market",
                 "electronics_market", "consumer_market", "packaging_market"]

# ─── display tokens per anchor ────────────────────────────────────────────
def tok(anchor: str) -> str:
    return SCHEMA[anchor]["tokens"][0]


# ─── memo templates keyed on memo_type × (gate pass/fail pattern) ─────────
def render_rd_note(chem: str, prop: str, lab: str, ts: str) -> str:
    a = tok(lab); b = tok(chem); c = tok(prop)
    return random.choice([
        f"{a} synthesized a new {b} with {c}.",
        f"{a} tested a {b} formulation for {c}.",
        f"The {a} reports a {b} system that exhibits {c}.",
    ])


def render_customer_request(oem: str, chem: str, market: str) -> str:
    a = tok(oem); b = tok(chem); m = tok(market)
    return random.choice([
        f"{a} requested a {b} suited for the {m}.",
        f"A partnership inquiry from {a} for {b} in {m} was received.",
        f"{a} is requesting a bespoke {b} for {m} applications.",
    ])


def render_patent(firm: str, chem: str, cure: str) -> str:
    a = tok(firm); b = tok(chem); c = tok(cure)
    return random.choice([
        f"{a} filed for ip on a {c} {b} formulation.",
        f"{a} patents a new {c} {b} system.",
    ])


def render_gate_review(firm: str, chem: str, gate_outcome_tok: str) -> str:
    a = tok(firm); b = tok(chem)
    return random.choice([
        f"The gate review for {a}'s {b} program determined the project was {gate_outcome_tok}.",
        f"{a}'s {b} candidate was classified as {gate_outcome_tok} in this quarter's review.",
    ])


# ─── memo generators with embedded gate truth ─────────────────────────────
def gen_funded(idx: int, ts: str):
    """Passes all 4 gates — funded for scale-up."""
    chem = random.choice(["polyurethane", "epoxy", "acrylate", "silicone"])
    prop = random.choice(MECH_PROPS + ["high_tg"])
    lab = random.choice(DOW_LABS)
    text = render_rd_note(chem, prop, lab, ts)
    text += " " + random.choice([
        "The material cleared gate review for commercial scale-up.",
        f"Addressable volume in {tok(random.choice(['auto_market', 'packaging_market', 'aerospace_market']))} is large enough for world-scale production.",
    ])
    return {
        "id": f"memo_{idx:04d}", "text": text, "timestamp": ts,
        "memo_type": "funded",
        "gates": {"gate1": True, "gate2": True, "gate3": True, "gate4": True,
                   "chimera_candidate": False},
        "triples": [
            (lab, "synthesize", chem),
            ("dow", "pass_review", chem),
        ],
    }


def gen_chimera(idx: int, ts: str):
    """Passes Gate 1 + 2, fails Gate 3 on volume. These are the prize."""
    chem = random.choice(["tissue_glue", "bio_adhesive", "cyanoacrylate",
                           "silicone"])
    # Niche customer request first
    oem = random.choice(["medtech_co", "startup_biomat", "startup_coatings"])
    market = random.choice(["med_device_market", "electronics_market"])
    sent1 = render_customer_request(oem, chem, market)
    # Gate 3 failure — volume too small
    sent2 = random.choice([
        f"Projected annual volume is under the {tok('gate3_fail')} threshold for world-scale manufacturing.",
        f"The program was {tok('reject')} because the market volume is {tok('gate3_fail')}.",
        f"This candidate is classified as a {tok('chimera_candidate')}.",
    ])
    return {
        "id": f"memo_{idx:04d}", "text": f"{sent1} {sent2}", "timestamp": ts,
        "memo_type": "chimera",
        "gates": {"gate1": True, "gate2": True, "gate3": False, "gate4": True,
                   "chimera_candidate": True},
        "triples": [
            (oem, "partner", chem),
            ("dow", "reject", chem),
        ],
    }


def gen_gate1_fail(idx: int, ts: str):
    """Fails market test — no customer pull."""
    chem = random.choice(CHEMISTRIES)
    lab = random.choice(DOW_LABS)
    text = render_rd_note(chem, random.choice(MECH_PROPS), lab, ts)
    text += " " + random.choice([
        "No customer demand was identified during market discovery.",
        "The commercial hypothesis failed external validation.",
        f"{tok(random.choice(MANUFACTURERS))} exited this segment last year.",
    ])
    return {
        "id": f"memo_{idx:04d}", "text": text, "timestamp": ts,
        "memo_type": "gate1_fail",
        "gates": {"gate1": False, "gate2": True, "gate3": True, "gate4": True,
                   "chimera_candidate": False},
        "triples": [
            (lab, "synthesize", chem),
        ],
    }


def gen_gate2_fail(idx: int, ts: str):
    """Passes market, fails core-competency."""
    chem = random.choice(CHEMISTRIES)
    text = f"A request for {tok(chem)} was received but the chemistry falls outside {tok('dow')}'s core expertise."
    return {
        "id": f"memo_{idx:04d}", "text": text, "timestamp": ts,
        "memo_type": "gate2_fail",
        "gates": {"gate1": True, "gate2": False, "gate3": True, "gate4": True,
                   "chimera_candidate": False},
        "triples": [],
    }


def gen_gate4_fail(idx: int, ts: str):
    """Passes 1/2/3, fails sustainability."""
    chem = random.choice(CHEMISTRIES)
    text = (f"{tok('dow_midland_lab')} developed a {tok(chem)} that meets scale targets "
            f"but fails our {tok('sustainability')} criteria.")
    return {
        "id": f"memo_{idx:04d}", "text": text, "timestamp": ts,
        "memo_type": "gate4_fail",
        "gates": {"gate1": True, "gate2": True, "gate3": True, "gate4": False,
                   "chimera_candidate": False},
        "triples": [
            ("dow_midland_lab", "synthesize", chem),
        ],
    }


def gen_patent(idx: int, ts: str):
    firm = random.choice(MANUFACTURERS)
    chem = random.choice(CHEMISTRIES)
    cure = random.choice(CURE_TYPES)
    text = render_patent(firm, chem, cure)
    return {
        "id": f"memo_{idx:04d}", "text": text, "timestamp": ts,
        "memo_type": "patent",
        "gates": None,
        "triples": [(firm, "patent", chem)],
    }


def gen_background(idx: int, ts: str):
    """Neutral market color — no gate implication."""
    firm = random.choice(MANUFACTURERS)
    chem = random.choice(CHEMISTRIES)
    market = random.choice(MARKETS)
    text = random.choice([
        f"{tok(firm)} continues to supply {tok(chem)} to the {tok(market)}.",
        f"{tok(firm)} scaled up {tok(chem)} production last quarter.",
        f"Analyst coverage noted {tok(firm)}'s expansion in {tok(chem)}.",
    ])
    return {
        "id": f"memo_{idx:04d}", "text": text, "timestamp": ts,
        "memo_type": "background",
        "gates": None,
        "triples": [(firm, "supply", chem)],
    }


# ─── mix and save ─────────────────────────────────────────────────────────
def timestep_to_iso(month: int) -> str:
    return f"2026-{month:02d}-01"


def main():
    N_MONTHS = 12
    mix = {
        "funded":     0.20,
        "chimera":    0.25,
        "gate1_fail": 0.15,
        "gate2_fail": 0.08,
        "gate4_fail": 0.07,
        "patent":     0.10,
        "background": 0.15,
    }
    generators = {
        "funded": gen_funded, "chimera": gen_chimera,
        "gate1_fail": gen_gate1_fail, "gate2_fail": gen_gate2_fail,
        "gate4_fail": gen_gate4_fail, "patent": gen_patent,
        "background": gen_background,
    }

    memos = []
    idx = 0
    for month in range(1, N_MONTHS + 1):
        ts = timestep_to_iso(month)
        n_this_month = random.randint(20, 30)
        for _ in range(n_this_month):
            # weighted random memo type
            r = random.random()
            cum = 0.0; chosen = None
            for name, w in mix.items():
                cum += w
                if r <= cum: chosen = name; break
            memos.append(generators[chosen](idx, ts))
            idx += 1

    # Write memos
    memo_path = DATA / "memos.jsonl"
    with memo_path.open("w") as f:
        for m in memos:
            public = {"id": m["id"], "text": m["text"],
                       "timestamp": m["timestamp"], "memo_type": m["memo_type"]}
            f.write(json.dumps(public) + "\n")
    print(f"Wrote {len(memos)} memos to {memo_path}")

    # Write ground truth gates
    gt_path = DATA / "ground_truth_gates.jsonl"
    with gt_path.open("w") as f:
        for m in memos:
            if m["gates"] is None: continue
            row = {"memo_id": m["id"], **m["gates"]}
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {sum(1 for m in memos if m['gates'])} gate ground-truth rows to {gt_path}")

    # Write gold triples
    triples_path = DATA / "gold_triples.jsonl"
    with triples_path.open("w") as f:
        for m in memos:
            for (s, p, o) in m["triples"]:
                f.write(json.dumps({"memo_id": m["id"], "subject": s,
                                     "predicate": p, "object": o}) + "\n")
    print(f"Wrote gold triples to {triples_path}")

    # Summary
    from collections import Counter
    by_type = Counter(m["memo_type"] for m in memos)
    print(f"\nmemo type mix: {dict(by_type)}")
    by_month = Counter(m["timestamp"] for m in memos)
    print(f"memos per month: {dict(sorted(by_month.items()))}")
    chim = sum(1 for m in memos if m["gates"] and m["gates"]["chimera_candidate"])
    print(f"chimera candidates (validated-no): {chim}")


if __name__ == "__main__":
    main()
