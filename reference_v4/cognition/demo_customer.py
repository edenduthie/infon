"""Customer demo: step-by-step walkthrough of cognition's key innovations.

Rich terminal graphics with animations showing the full pipeline from
raw text to MCTS reasoning, with detailed infon metadata inspection.

Run:
    python cognition/demo_customer.py              # interactive (press Enter)
    python cognition/demo_customer.py --no-pause   # continuous output

To adapt for a new domain, edit the DOMAIN dict below — everything else
is generic engine visualisation.
"""

import sys
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cognition import Cognition
from cognition.config import CognitionConfig
from cognition.structural import (
    StructuralAnalyzer, KanExtension,
    FeatureGapFunctor, GhostDetector,
)

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.columns import Columns
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.style import Style
from rich.markup import escape
from rich import box

console = Console()

NO_PAUSE = "--no-pause" in sys.argv


# ═══════════════════════════════════════════════════════════════════════
# DOMAIN CONFIGURATION — edit this block to change domain
# ═══════════════════════════════════════════════════════════════════════
#
# Everything the demo needs to know about your domain lives here.
# The engine code, visualisations, and Rich formatting are generic —
# only DOMAIN needs to change when switching from automotive to pharma,
# finance, defense, etc.

DOMAIN = {
    # ── Schema ────────────────────────────────────────────────────────
    "schema_path": str(Path(__file__).parent.parent / "data" / "dimefil_schema.json"),

    # ── Corpus ────────────────────────────────────────────────────────
    "documents": [
        # ── Phase 1: Diplomatic collapse & war outbreak ───────────────
        {"text": "Iran temporarily restricted transit through the Strait of Hormuz, "
                 "sending a strategic signal to global shipping markets. War risk "
                 "insurance premiums immediately spiked as underwriters reassessed "
                 "Persian Gulf exposure.",
         "id": "hormuz_signal", "timestamp": "2026-02-18"},
        {"text": "US and Israeli forces launched coordinated strikes on Iranian military "
                 "targets. Tankers began diverting from the Strait of Hormuz as the "
                 "waterway became effectively closed to commercial traffic, halting "
                 "approximately 21 million barrels per day of crude oil transit.",
         "id": "war_outbreak", "timestamp": "2026-02-28"},
        {"text": "Five commercial vessels were struck by a concentrated wave of missile "
                 "and drone attacks in the Strait of Hormuz. The US-flagged product tanker "
                 "Stena Imperative was hit by projectiles while berthed at Port of Bahrain, "
                 "killing one shipyard worker and injuring two others.",
         "id": "vessel_attacks", "timestamp": "2026-03-02"},

        # ── Phase 2: Insurance & shipping collapse ────────────────────
        {"text": "London marine insurers expanded the high-risk zone to include Bahrain, "
                 "Kuwait, Oman, and Qatar. War risk insurance premiums surged over 1000 "
                 "percent following a US submarine strike on an Iranian naval vessel. "
                 "Reinsurers issued seven-day cancellation notices on marine war-risk coverage.",
         "id": "insurance_collapse", "timestamp": "2026-03-06"},
        {"text": "The Trump administration unveiled a 20 billion dollar maritime reinsurance "
                 "facility to restore shipping confidence through the Strait of Hormuz. "
                 "Chubb was designated as lead administrator of the federal insurance backstop.",
         "id": "insurance_backstop", "timestamp": "2026-03-06"},
        {"text": "Kuwait declared force majeure with precautionary reductions in crude oil "
                 "production and refinery throughput following Iranian attacks on maritime "
                 "infrastructure in the Persian Gulf.",
         "id": "kuwait_force_majeure", "timestamp": "2026-03-07"},

        # ── Phase 3: LNG crisis & Ras Laffan attack ──────────────────
        {"text": "Iranian drone and missile strikes caused extensive damage to the Ras Laffan "
                 "LNG export facility in Qatar, knocking out 17 percent of the country's LNG "
                 "capacity. QatarEnergy CEO Saad al-Kaabi estimated repairs would take three "
                 "to five years, with 20 billion dollars in lost annual revenue.",
         "id": "ras_laffan_attack", "timestamp": "2026-03-18"},
        {"text": "The gas crisis dwarfed the oil shock as approximately 20 percent of global "
                 "LNG trade was halted. European gas prices surged over 50 percent following "
                 "the Ras Laffan attack. EU gas storage fell below 30 percent, and Standard "
                 "Chartered warned prices could spike above 90 dollars per MWh by summer.",
         "id": "lng_crisis", "timestamp": "2026-03-25"},
        {"text": "Pakistan's power grid entered crisis as Qatar LNG supplies were cut off "
                 "by the Hormuz closure. India's LPG market was thrown into chaos with spot "
                 "prices doubling. Moldova became the first European country to report acute "
                 "gas shortages linked to the conflict.",
         "id": "asia_energy_crisis", "timestamp": "2026-03-19"},

        # ── Phase 4: Supply chain cascades ────────────────────────────
        {"text": "Europe faced a summer jet fuel crisis as accelerated refinery closures "
                 "and increased dependence on Middle Eastern kerosene exposed critical "
                 "supply vulnerabilities. Lufthansa announced cancellation of 20,000 "
                 "short-haul European flights from its summer schedule.",
         "id": "jet_fuel_crisis", "timestamp": "2026-04-18"},
        {"text": "Long-haul European airfares increased approximately 105 dollars per ticket "
                 "since the onset of the Middle East crisis. Air Canada scrapped key US routes "
                 "citing unsustainable fuel costs. Aviation fuel now represented over 40 percent "
                 "of airline operating expenses.",
         "id": "aviation_impact", "timestamp": "2026-04-21"},
        {"text": "Asia imports over 50 percent of its seaborne naphtha from the Middle East. "
                 "Chemical plant utilization across the region fell from 83 percent in February "
                 "to an estimated 57 percent in May. South Korea banned naphtha exports for "
                 "five months to protect domestic petrochemical production.",
         "id": "petrochemical_crisis", "timestamp": "2026-03-27"},
        {"text": "European ethylene prices surged 450 euros per tonne and propylene jumped "
                 "465 euros per tonne compared to March levels. Polyethylene and polypropylene "
                 "saw 27 to 36 percent month-on-month price increases. China's coal-based "
                 "plants ran at full capacity to fill the gap.",
         "id": "petrochem_prices", "timestamp": "2026-04-15"},
        {"text": "Lack of ammonia availability from the Middle East pushed Asian fertilizer "
                 "prices to multi-year highs. Urea prices jumped on tight supply while demand "
                 "dropped due to low affordability. Brazil allocated 3 billion dollars under "
                 "its Brasil Soberano plan to secure fertilizer imports for agriculture.",
         "id": "fertilizer_crisis", "timestamp": "2026-04-10"},
        {"text": "Fears of a global food catastrophe grew as the Iran crisis dragged on. "
                 "The fertilizer supply chain from natural gas to ammonia to urea to crop "
                 "was broken at the first link. UK consumer prices rose to 3.3 percent in "
                 "March with motor fuel up 4.9 percent.",
         "id": "food_security_impact", "timestamp": "2026-04-21"},

        # ── Phase 5: Refinery & infrastructure attacks ────────────────
        {"text": "The Russian LNG tanker Arctic Metagaz suffered an explosion and fire "
                 "150 nautical miles southeast of Malta. The crew abandoned ship and the "
                 "vessel drifted uncontrolled through the Mediterranean. Moscow alleged "
                 "a Ukrainian drone boat attack on the shadow fleet vessel.",
         "id": "arctic_metagaz_fire", "timestamp": "2026-03-03"},
        {"text": "The Novokuibyshevsk Refinery operated by Rosneft halted primary crude "
                 "processing following a Ukrainian drone strike. Russia subsequently "
                 "announced suspension of Kazakh oil flows through the Druzhba pipeline "
                 "to Germany effective May 1st.",
         "id": "russian_refinery_strike", "timestamp": "2026-04-18"},

        # ── Phase 6: Escalation & ceasefire attempts ──────────────────
        {"text": "Houthi forces launched attacks widening the Iran war, threatening the "
                 "Bab al-Mandab Strait in solidarity with Tehran. Brent crude headed for "
                 "a record monthly rise. The dual chokepoint crisis meant both the Strait "
                 "of Hormuz and Red Sea routes were now disrupted.",
         "id": "houthi_escalation", "timestamp": "2026-03-30"},
        {"text": "Iran seized two MSC containerships in the Strait of Hormuz in what "
                 "shipping analyst Xeneta described as weaponization of trade. Iranian "
                 "gunboats fired on commercial shipping. Over 270,000 TEU of containers "
                 "were stranded after carriers halted Arabian Gulf bookings.",
         "id": "iran_ship_seizure", "timestamp": "2026-04-22"},
        {"text": "Only three ships passed through the Strait of Hormuz in 24 hours as "
                 "traffic reached a virtual standstill. The IMO prepared an evacuation "
                 "plan for hundreds of trapped Persian Gulf vessels and approximately "
                 "20,000 stranded seafarers.",
         "id": "hormuz_standstill", "timestamp": "2026-04-21"},
        {"text": "Trump extended a two-week ceasefire indefinitely at Pakistan's request "
                 "while maintaining the naval blockade. Brent crude traded at 103 dollars "
                 "per barrel. The IEA declared the Strait of Hormuz had lost its status "
                 "as a reliable global energy route.",
         "id": "ceasefire_extension", "timestamp": "2026-04-22"},

        # ── Mitigation & alternatives ─────────────────────────────────
        {"text": "The IEA chief proposed an Iraq-Turkey pipeline as an alternative to "
                 "bypass the Strait of Hormuz. Japan created a 10 billion dollar fund "
                 "to help Southeast Asian economies procure crude oil through non-Gulf "
                 "supply routes.",
         "id": "alternative_routes", "timestamp": "2026-04-19"},
        {"text": "US LNG exports soared to a record 11.7 million metric tons in March "
                 "amid panic buying from European and Asian buyers. China had strategically "
                 "accumulated 1.5 billion barrels of crude reserves and state oil giants "
                 "began selling crude for May loadings.",
         "id": "supply_mitigation", "timestamp": "2026-04-09"},
    ],

    # ── Narrative — displayed in title / intro / timeline header ──────
    "narrative": "Strait of Hormuz crisis and global supply chain cascade",
    "intro_question": "How does a chokepoint closure cascade through jet fuel, fertilizer, and petrochemicals?",

    # ── Example triple — used in the hyperedge diagram (Step 2) ───────
    "example_subject":   "irgc",                # anchor name (actor)
    "example_predicate": "blockade",            # anchor name (relation)
    "example_object":    "strait_of_hormuz",    # anchor name (location)

    # ── Temporal chain labels (Step 3 diagram) ────────────────────────
    "chain_predicates": ["blockade", "shipping_insurance", "lng", "supply_chain"],
    "chain_gaps_days":  [6, 12, 14],

    # ── Timeline doc-id keywords → sentiment (Step 3) ─────────────────
    "negative_doc_keywords": ["attack", "crisis", "fire", "seizure", "strike", "standstill", "houthi"],
    "positive_doc_keywords": ["ceasefire", "backstop", "mitigation", "alternative", "supply_mitigation"],

    # ── Hierarchy expansion (Step 4) ──────────────────────────────────
    "hierarchy_parent": "middle_east",
    "hierarchy_query_broad": "Middle East energy supply disruption",
    "hierarchy_query_specific": "Strait of Hormuz shipping blockade",

    # ── Dempster-Shafer opposing sentences (Step 5) ───────────────────
    "ds_affirm_sentence": "The Strait of Hormuz closure has caused catastrophic disruption to global energy supply chains.",
    "ds_negate_sentence": "Alternative pipeline routes and strategic reserves have largely mitigated the Hormuz closure impact.",
    "ds_affirm_confidence": 0.85,
    "ds_negate_confidence": 0.6,

    # ── Claim for full DS verification (Step 5) ──────────────────────
    "claim": "The Hormuz blockade caused a global supply chain crisis cascading from energy through fertilizer to food security",

    # ── MCTS query (Step 6) ──────────────────────────────────────────
    "mcts_query": "Did the Strait of Hormuz closure cause a cascading supply chain failure from energy to agriculture?",

    # ── Multi-hop narrative shown before MCTS (Step 6) ────────────────
    "mcts_hops": [
        ("naval blockade", "2026-02"),
        ("LNG/oil supply break", "2026-03"),
        ("fertilizer & food crisis", "2026-04"),
    ],

    # ── Comparison narrative (end of Step 6) ─────────────────────────
    "comparison_explanation": (
        "Flat retrieval finds the fertilizer crisis but can't trace WHY.\n"
        "MCTS discovers the causal chain: Hormuz blockade → LNG halt →\n"
        "ammonia plant shutdowns → fertilizer price spike → food security.\n"
        "Hierarchy expansion lets 'Middle East' find Hormuz, Ras Laffan,\n"
        "Kuwait — specific nodes flat retrieval misses entirely."
    ),

    # ── Contrary view (Step 7) ───────────────────────────────────────
    "contrary_claim": "The Strait of Hormuz crisis has been effectively contained by alternative supply routes",
    "contrary_query": "Have alternative routes and reserves mitigated the Hormuz supply disruption?",

    # ── Kan extension (Step 12) ──────────────────────────────────────
    "kan_source_label": "Kinetic Phase (Feb-Mar 2026)",
    "kan_target_label": "Cascade Phase (Apr 2026)",
    "kan_source_cutoff": "2026-03-31",
}


# ═══════════════════════════════════════════════════════════════════════
# PALETTE — visual constants (not domain-specific)
# ═══════════════════════════════════════════════════════════════════════

ANCHOR_COLORS = {
    "actor": "magenta",
    "relation": "yellow",
    "feature": "green",
    "market": "bright_blue",
    "location": "bright_blue",
}
ANCHOR_LABELS = {
    "actor": "ACTOR",
    "relation": "RELATION",
    "feature": "FEATURE",
    "market": "MARKET",
    "location": "LOCATION",
}
SUPPORT_COLOR = "green"
REFUTE_COLOR = "red"
UNCERTAIN_COLOR = "yellow"
ACCENT = "bright_cyan"
DIM = "dim"


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def pause():
    if NO_PAUSE:
        console.print()
        return
    console.print()
    console.input("  [dim]\\[press Enter to continue][/]")
    console.print()


def flash_text(text, style="bold white on blue", times=3, interval=0.12):
    """Flash text on/off using Live for clean in-place rendering."""
    if NO_PAUSE:
        console.print(Align.center(Text(f" {text} ", style=style)))
        return
    with Live(console=console, transient=True, refresh_per_second=20) as live:
        for _ in range(times):
            live.update(Align.center(Text(f" {text} ", style=style)))
            time.sleep(interval)
            live.update(Text(""))
            time.sleep(interval * 0.5)
    console.print(Align.center(Text(f" {text} ", style=style)))


def bar_str(value, max_val=1.0, width=30, fill="█", empty="░"):
    """Static bar string (no markup)."""
    frac = max(0.0, min(1.0, value / max_val)) if max_val > 0 else 0
    filled = int(frac * width)
    return fill * filled + empty * (width - filled)


def animate_bar(label, value, max_val=1.0, width=30, color="green", duration=0.35):
    """Animate a bar filling up using Live, then print the final state."""
    final_bar = bar_str(value, max_val, width)
    final_line = Text.from_markup(
        f"    [{color}]{final_bar}[/] [{color} bold]{value:.2f}[/]  [dim]{label}[/]"
    )

    if NO_PAUSE:
        console.print(final_line)
        return

    steps = 15
    with Live(console=console, transient=True, refresh_per_second=30) as live:
        for i in range(steps + 1):
            frac = i / steps
            partial_val = value * frac
            b = bar_str(partial_val, max_val, width)
            live.update(Text.from_markup(
                f"    [{color}]{b}[/] [{color} bold]{partial_val:.2f}[/]  [dim]{label}[/]"
            ))
            time.sleep(duration / steps)
    console.print(final_line)


# ═══════════════════════════════════════════════════════════════════════
# TITLE SCREEN
# ═══════════════════════════════════════════════════════════════════════

def render_title():
    title_text = Text()
    title_text.append("S P L A D E   I N F O N   E N G I N E\n", style="bold white")
    title_text.append("Visualizing Situational Intelligence", style=f"dim {ACCENT}")

    flat_panel = Panel(
        "[white]───────────▶[/]\n\n[dim]Single dense vector\nNo structure, no types[/]",
        title="[dim white]FLAT VECTOR[/]",
        subtitle="[dim]standard RAG[/]",
        border_style="dim",
        box=box.ROUNDED,
        width=28,
    )
    hyper_panel = Panel(
        f"[{ACCENT}]╱╲  ╱╲  ╱╲\n╱  ╲╱  ╲╱  ╲╱╲[/]\n"
        f"[{ACCENT}]Typed concept graph\nGrounded triples + time[/]",
        title=f"[bold {ACCENT}]INFON / HYPERGRAPH[/]",
        subtitle=f"[{ACCENT}]this engine[/]",
        border_style=ACCENT,
        box=box.ROUNDED,
        width=28,
    )

    comparison = Table.grid(padding=(0, 4))
    comparison.add_column(justify="center", width=28)
    comparison.add_column(justify="center", width=28)
    comparison.add_row(flat_panel, hyper_panel)

    steps = Table.grid(padding=(0, 2))
    steps.add_column(style="bold white", width=3, justify="right")
    steps.add_column(style=f"bold {ACCENT}", width=22)
    steps.add_column(style="dim")
    steps.add_row("1", "Change of Basis", "SPLADE → typed anchor projection")
    steps.add_row("2", "Infon Extraction", "grounded (S,P,O) triples with spans")
    steps.add_row("3", "Temporal Graph", "NEXT edges link facts across time")
    steps.add_row("4", "Hierarchy Expansion", "parent queries find child-level evidence")
    steps.add_row("5", "Dempster-Shafer", "principled belief under contradiction")
    steps.add_row("6", "Graph MCTS", "tree search for multi-hop reasoning")
    steps.add_row("7", "Contrary View", "invert the lens — surface counter-evidence")
    steps.add_row("", "", "")
    steps.add_row("8", "Kano + Conjoint", "classify & rank anchors by market impact")
    steps.add_row("9", "Feature Gap Functor", "F: Schema → Discourse distortion map")
    steps.add_row("10", "Ghosts + Polarization", "dead anchors & sentiment H¹")
    steps.add_row("11", "Narrative Lifecycle", "β₀/β₁ persistent homology on chains")
    steps.add_row("12", "Contagion + Kan Ext.", "risk fan-out & market transfer")
    steps.add_row("13", "Driver Tree", "McKinsey-style decomposition from topology")

    inner = Table.grid(padding=(1, 0))
    inner.add_column()
    inner.add_row(Align.center(title_text))
    inner.add_row(Align.center(comparison))
    inner.add_row(Align.center(steps))

    n_docs = len(DOMAIN["documents"])
    narrative = DOMAIN["narrative"]
    intro_q = DOMAIN["intro_question"]

    console.print()
    console.print(Panel(inner, border_style=ACCENT, box=box.DOUBLE_EDGE, padding=(1, 3)))
    console.print()
    console.print(f"  [dim italic]We'll ingest {n_docs} documents about {narrative},[/]")
    console.print(f"  [dim italic]then answer a question no single article can: \"{intro_q}\"[/]")


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: CHANGE OF BASIS
# ═══════════════════════════════════════════════════════════════════════

def render_change_of_basis(cog):
    console.print()
    console.rule(f"[bold white]STEP 1[/]  [{ACCENT}]CHANGE OF BASIS[/]", style=ACCENT)
    console.print("  [dim]SPLADE encodes text into your concept space — no training needed[/]\n")

    sentence = DOMAIN["documents"][0]["text"]
    console.print(f"  [dim]Input:[/]")
    console.print(f"  [{ACCENT}]\"{sentence[:72]}...\"[/]\n")

    with Progress(
        SpinnerColumn("dots12", style=ACCENT),
        TextColumn(f"[{ACCENT}]Encoding through SPLADE + AnchorProjector...[/]"),
        console=console, transient=True,
    ) as progress:
        progress.add_task("", total=None)
        activations = cog.encoder.encode_single(sentence)
        if not NO_PAUSE:
            time.sleep(0.5)

    sorted_acts = sorted(activations.items(), key=lambda x: -x[1])

    # Flow diagram
    flow = Group(
        Panel(
            "[bold]SPLADE[/]  [dim]log(1 + ReLU(MLM_logits)) + max-pool[/]\n"
            "→ [white]30,522-dim[/] sparse vocabulary activation",
            title="[dim]sentence  ▼[/]", border_style=ACCENT, box=box.ROUNDED, width=58,
        ),
        Align.center(Text("│\n▼", style="dim"), width=58),
        Panel(
            f"[bold]AnchorProjector[/]  [dim]token-ID max-pooling per anchor[/]\n"
            f"→ [bold white]{len(activations)}[/] typed anchor activations [dim](your concepts)[/]",
            border_style=ACCENT, box=box.ROUNDED, width=58,
        ),
        Align.center(Text("│\n▼", style="dim"), width=58),
    )
    console.print(flow)
    console.print()

    # Ontology grid
    grid = Table(
        title="[bold]ONTOLOGY GRID[/]  [dim]— each node is a typed anchor[/]",
        title_style="white",
        border_style=ACCENT,
        box=box.ROUNDED,
        show_header=True,
        header_style=f"bold {ACCENT}",
        padding=(0, 1),
    )
    grid.add_column("Type", width=10, style="dim")
    grid.add_column("Anchor", width=16, style="bold white")
    grid.add_column("Activation", width=24)
    grid.add_column("Score", width=8, justify="right")

    max_score = sorted_acts[0][1] if sorted_acts else 1.0
    for name, score in sorted_acts[:8]:
        if score < 0.05:
            break
        atype = cog.schema.types.get(name, "feature")
        tc = ANCHOR_COLORS.get(atype, "dim")
        tl = ANCHOR_LABELS.get(atype, "?")
        bw = int((score / max_score) * 16)
        b = f"[{tc}]{'█' * bw}{'░' * (16 - bw)}[/]"
        grid.add_row(f"[{tc}]{tl}[/]", f"[white]{name}[/]", b, f"[{tc}]{score:.2f}[/]")

    console.print(grid)
    console.print()

    for bullet in [
        "No training — define a JSON schema with token lists and go",
        "Literal change of basis — linear algebra, not learned embeddings",
        "Every activation grounded to specific BERT vocabulary tokens",
        "Works across any domain — just swap the schema JSON",
    ]:
        console.print(f"    [{SUPPORT_COLOR}]•[/] {bullet}")
    console.print()


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: INFON EXTRACTION + METADATA
# ═══════════════════════════════════════════════════════════════════════

def _infon_card(inf, cog, rank):
    """Build a single infon metadata card as a Panel."""
    pol_word = f"[{SUPPORT_COLOR}]affirmed (+1)[/]" if inf.polarity == 1 else f"[{REFUTE_COLOR}]negated (0)[/]"
    atype_s = cog.schema.types.get(inf.subject, "?")
    atype_p = cog.schema.types.get(inf.predicate, "?")
    atype_o = cog.schema.types.get(inf.object, "?")
    sc_s = ANCHOR_COLORS.get(atype_s, "white")
    sc_p = ANCHOR_COLORS.get(atype_p, "white")
    sc_o = ANCHOR_COLORS.get(atype_o, "white")

    def kv_grid():
        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim", width=14, justify="right")
        t.add_column(width=55)
        return t

    # Core Triple
    core = kv_grid()
    core.add_row("subject", f"[{sc_s} bold]{inf.subject}[/]  [dim]type={atype_s}[/]")
    core.add_row("predicate", f"[{sc_p} bold]{inf.predicate}[/]  [dim]type={atype_p}[/]")
    core.add_row("object", f"[{sc_o} bold]{inf.object}[/]  [dim]type={atype_o}[/]")
    core.add_row("polarity", pol_word)
    core.add_row("confidence", f"[{ACCENT}]{inf.confidence:.3f}[/]  [dim](geometric mean of role scores)[/]")

    # Grounding
    ground = kv_grid()
    sent_text = escape(inf.sentence[:70]) + ("..." if len(inf.sentence) > 70 else "")
    ground.add_row("sentence", f"[italic]\"{sent_text}\"[/]")
    ground.add_row("doc_id", f"[white]{inf.doc_id}[/]")
    ground.add_row("sent_id", f"[dim]{inf.sent_id}[/]")
    if inf.spans:
        for role, span in inf.spans.items():
            text = escape(span.get("text", ""))
            start = span.get("start", "?")
            end = span.get("end", "?")
            ground.add_row(f"span.{role}", f"[{ACCENT}]\"{text}\"[/]  [dim]chars \\[{start}:{end}][/]")
    if inf.support:
        for role, stype in inf.support.items():
            sc = SUPPORT_COLOR if stype == "direct" else UNCERTAIN_COLOR if stype == "semantic" else "bright_blue"
            ground.add_row(f"support.{role}", f"[{sc}]{stype}[/]")

    # Hierarchy
    hier = kv_grid()
    for role_name, meta_field in [("subject", inf.subject_meta), ("predicate", inf.predicate_meta), ("object", inf.object_meta)]:
        if meta_field:
            meta_str = "  ".join(f"[dim]{k}=[/][white]{v}[/]" for k, v in meta_field.items() if v)
            if meta_str.strip():
                hier.add_row(f"{role_name}_meta", meta_str)

    # Spatial
    spatial = kv_grid()
    if inf.locations:
        for loc in inf.locations:
            loc_parts = [f"{lk}=[white]{loc[lk]}[/]" for lk in ["name", "level", "country_code", "macro_region"] if loc.get(lk)]
            spatial.add_row("location", f"[dim]{' '.join(loc_parts)}[/]")
    else:
        spatial.add_row("locations", "[dim]none inferred[/]")

    # Temporal
    temporal = kv_grid()
    temporal.add_row("timestamp", f"[white]{inf.timestamp or 'none'}[/]")
    temporal.add_row("precision", f"[dim]{inf.precision}[/]")
    temporal.add_row("tense", f"[dim]{inf.tense}[/]")
    temporal.add_row("aspect", f"[dim]{inf.aspect}[/]")
    if inf.temporal_refs:
        for ref in inf.temporal_refs:
            temporal.add_row("temporal_ref", f"[{ACCENT}]\"{ref.get('text', '')}\"[/]  [dim]precision={ref.get('precision', '?')}[/]")

    # Importance — compact scoring bars
    def score_bar(val, color, w=16):
        bw = int(max(0, min(1, val)) * w)
        return f"[{color}]{'█' * bw}{'░' * (w - bw)}[/] [{color}]{val:.3f}[/]"

    imp = kv_grid()
    imp.add_row("activation", score_bar(inf.activation, ACCENT))
    imp.add_row("coherence", score_bar(inf.coherence, UNCERTAIN_COLOR))
    imp.add_row("specificity", score_bar(inf.specificity, SUPPORT_COLOR))
    imp.add_row("novelty", score_bar(inf.novelty, "magenta"))
    imp.add_row("importance", score_bar(min(inf.importance, 1.0), "bold white"))
    imp.add_row("reinforcement", f"[dim]{inf.reinforcement_count}[/]")
    imp.add_row("decay_rate", f"[dim]{inf.decay_rate}[/]")

    inner = Group(
        Rule("[bold]Core Triple[/]", style="dim"),
        core,
        Rule("[bold]Grounding[/]", style="dim"),
        ground,
        Rule("[bold]Hierarchy Metadata[/]", style="dim"),
        hier,
        Rule("[bold]Spatial Context[/]", style="dim"),
        spatial,
        Rule("[bold]Temporal Context[/]", style="dim"),
        temporal,
        Rule("[bold]Importance Scoring[/]", style="dim"),
        imp,
    )

    return Panel(
        inner,
        title=f"[bold white]Infon {rank}[/]  [dim]{inf.infon_id[:16]}...[/]",
        border_style=ACCENT,
        box=box.ROUNDED,
        width=78,
    )


def _hyperedge_diagram(subj, pred, obj, schema_types):
    """Build the directed-hyperedge diagram from DOMAIN example triple."""
    st = schema_types.get(subj, "?")
    pt = schema_types.get(pred, "?")
    ot = schema_types.get(obj, "?")
    sc = ANCHOR_COLORS.get(st, "white")
    pc = ANCHOR_COLORS.get(pt, "white")
    oc = ANCHOR_COLORS.get(ot, "white")
    return Panel(
        f"[{sc}]{subj}[/] ──[{pc}]INITIATES[/]──▶ [bold white]<<{pred}, {subj}, {obj}; +>>[/]\n"
        f"[dim]\\[{st}][/]                   [dim]│                              │[/]\n"
        f"                       [{pc}]ASSERTS[/] [dim]│[/]    [{pc}]TARGETS[/] [dim]│[/]\n"
        f"                          [dim]▼[/]              [dim]▼[/]\n"
        f"                    [{pc}]{pred}[/]         [{oc}]{obj}[/]\n"
        f"                  [dim]\\[{pt}][/]       [dim]\\[{ot}][/]",
        title="[bold]Directed Hyperedge[/]",
        border_style=ACCENT,
        box=box.ROUNDED,
        width=64,
    )


def render_extraction(cog, all_infons):
    console.print()
    console.rule(f"[bold white]STEP 2[/]  [{ACCENT}]INFON EXTRACTION[/]", style=ACCENT)
    console.print("  [dim]Sentences become grounded (S, P, O) triples with full metadata[/]\n")

    console.print(_hyperedge_diagram(
        DOMAIN["example_subject"],
        DOMAIN["example_predicate"],
        DOMAIN["example_object"],
        cog.schema.types,
    ))
    console.print()

    shown_keys = set()
    rank = 0
    for inf in all_infons:
        key = (inf.subject, inf.predicate, inf.object)
        if key in shown_keys:
            continue
        shown_keys.add(key)
        rank += 1
        if rank > 4:
            break
        console.print(_infon_card(inf, cog, rank))
        console.print()

    n_remaining = len(set((i.subject, i.predicate, i.object) for i in all_infons)) - rank
    if n_remaining > 0:
        console.print(f"  [dim]... and {n_remaining} more unique triples[/]")


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: TEMPORAL GRAPH
# ═══════════════════════════════════════════════════════════════════════

def _temporal_chain_diagram(cog):
    """Build the temporal chain diagram from DOMAIN chain_predicates."""
    subj = DOMAIN["example_subject"]
    preds = DOMAIN["chain_predicates"]
    gaps = DOMAIN["chain_gaps_days"]
    sc = ANCHOR_COLORS.get(cog.schema.types.get(subj, ""), "magenta")
    st = cog.schema.types.get(subj, "actor")

    lines = []
    lines.append(
        f"[{sc}]{subj}[/] ──[yellow]INITIATES[/]──▶ "
        f"[bold white]Infon₁[/] ──[yellow]ASSERTS[/]──▶ [yellow]{preds[0]}[/]"
    )
    lines.append(f"[dim]\\[{st}][/]                  [dim]│[/]")

    for i, pred in enumerate(preds[1:], start=1):
        gap = gaps[i - 1] if i - 1 < len(gaps) else "?"
        lines.append(f"                       [dim]│[/] [{ACCENT}]NEXT[/] [dim](per anchor, by timestamp)[/]")
        lines.append(f"                       [dim]│ gap: {gap} days[/]")
        lines.append(f"                       [dim]▼[/]")
        subscript = chr(0x2080 + i + 1)  # ₂, ₃, ₄, ...
        lines.append(
            f"                     [bold white]Infon{subscript}[/] ──[yellow]ASSERTS[/]──▶ [yellow]{pred}[/]"
        )
        if i < len(preds) - 1:
            lines.append(f"                       [dim]│[/]")

    return Panel(
        "\n".join(lines),
        title="[bold]Temporal Chain[/]  [dim]— NEXT edges link sequential facts[/]",
        border_style=ACCENT, box=box.ROUNDED, width=64,
    )


def render_temporal_graph(cog, all_infons):
    console.print()
    console.rule(f"[bold white]STEP 3[/]  [{ACCENT}]TEMPORAL GRAPH[/]", style=ACCENT)
    console.print("  [dim]NEXT edges link facts across time per shared anchor[/]\n")

    with Progress(
        SpinnerColumn("dots12", style=ACCENT),
        TextColumn(f"[{ACCENT}]Running consolidation...[/]"),
        console=console, transient=True,
    ) as progress:
        progress.add_task("", total=None)
        cog.consolidate()
        if not NO_PAUSE:
            time.sleep(0.3)

    stats = cog.stats()
    edges = cog.store.get_edges(edge_type="NEXT", limit=300)

    stats_tbl = Table.grid(padding=(0, 2))
    stats_tbl.add_column(style="dim", width=20, justify="right")
    stats_tbl.add_column(style=f"bold {ACCENT}")
    stats_tbl.add_row("Constraints", str(stats["constraint_count"]))
    stats_tbl.add_row("NEXT edges", str(len(edges)))
    stats_tbl.add_row("Has sequences", str(stats["has_sequences"]))
    console.print(stats_tbl)
    console.print()

    console.print(_temporal_chain_diagram(cog))
    console.print()

    # Timeline
    docs = DOMAIN["documents"]
    narrative = DOMAIN["narrative"]
    neg_kw = DOMAIN["negative_doc_keywords"]
    pos_kw = DOMAIN["positive_doc_keywords"]

    console.print(f"  [bold {ACCENT}]Timeline ({narrative})[/]")
    console.print(f"  [dim]{'─' * 64}[/]")

    docs_sorted = sorted(docs, key=lambda d: d["timestamp"])
    min_dt = datetime.strptime(docs_sorted[0]["timestamp"], "%Y-%m-%d")
    max_dt = datetime.strptime(docs_sorted[-1]["timestamp"], "%Y-%m-%d")
    span_days = max((max_dt - min_dt).days, 1)

    for doc in docs_sorted:
        ts = doc["timestamp"]
        src = doc["id"]
        d = datetime.strptime(ts, "%Y-%m-%d")
        frac = (d - min_dt).days / span_days
        pos = int(frac * 50) + 4

        if any(kw in src for kw in neg_kw):
            sym, color = "▼", REFUTE_COLOR
        elif any(kw in src for kw in pos_kw):
            sym, color = "▲", SUPPORT_COLOR
        else:
            sym, color = "●", ACCENT

        console.print(f"  {' ' * pos}[{color}]{sym}[/]  [dim]{ts} {src}[/]")

    console.print()

    # Top constraints
    constraints = cog.store.get_constraints(limit=50)
    console.print(f"  [bold {ACCENT}]Top Constraints[/]  [dim](cross-document aggregation)[/]")
    console.print(f"  [dim]{'─' * 64}[/]")
    for c in constraints[:5]:
        s_bar = bar_str(min(c.score, 1.0), width=14)
        console.print(f"  [magenta]{c.subject}[/] ──[yellow]\\[ {c.predicate} ][/]──▶ [green]{c.object}[/]")
        console.print(f"    [{SUPPORT_COLOR}]{s_bar}[/] [{SUPPORT_COLOR}]{c.score:.3f}[/]  "
                      f"[dim]evidence={c.evidence}  docs={c.doc_count}  "
                      f"strength={c.strength:.2f}  persistence={c.persistence}[/]")
    console.print()


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: HIERARCHY EXPANSION
# ═══════════════════════════════════════════════════════════════════════

def render_hierarchy_expansion(cog):
    console.print()
    console.rule(f"[bold white]STEP 4[/]  [{ACCENT}]HIERARCHY EXPANSION[/]", style=ACCENT)
    console.print("  [dim]Parent-level queries automatically find child-level evidence[/]\n")

    parent = DOMAIN["hierarchy_parent"]
    broad_q = DOMAIN["hierarchy_query_broad"]
    specific_q = DOMAIN["hierarchy_query_specific"]

    # Show the hierarchy tree
    children = cog.schema.get_children(parent)
    descendants = cog.schema.get_descendants(parent)

    tree = Tree(
        Text.from_markup(f"[bold {ACCENT}]{parent}[/]  [dim](parent — no infons stored here)[/]"),
        guide_style=ACCENT,
    )
    for child in children:
        child_color = ANCHOR_COLORS.get(cog.schema.types.get(child, ""), "dim")
        grandchildren = cog.schema.get_children(child)
        if grandchildren:
            branch = tree.add(Text.from_markup(
                f"[{child_color} bold]{child}[/]  [dim]{cog.schema.types.get(child, '')}[/]"
            ))
            for gc in grandchildren[:4]:
                gc_color = ANCHOR_COLORS.get(cog.schema.types.get(gc, ""), "dim")
                branch.add(Text.from_markup(
                    f"[{gc_color}]{gc}[/]  [dim]{cog.schema.types.get(gc, '')}[/]"
                ))
            if len(grandchildren) > 4:
                branch.add(Text(f"... +{len(grandchildren) - 4} more", style="dim"))
        else:
            tree.add(Text.from_markup(
                f"[{child_color} bold]{child}[/]  [dim]{cog.schema.types.get(child, '')}[/]"
            ))

    console.print(Panel(
        tree,
        title=f"[bold]Schema Hierarchy[/]  [dim]{len(descendants)} descendants[/]",
        border_style=ACCENT,
        box=box.ROUNDED,
        width=64,
    ))
    console.print()

    # The problem: querying at parent level with flat retrieval
    console.print(Panel(
        f"[dim]Query:[/] [bold]\"{broad_q}\"[/]\n\n"
        f"[dim]SPLADE activates:[/]  [{ACCENT}]{parent}[/]  [dim](parent anchor)[/]\n\n"
        f"[dim]Without expansion:[/]\n"
        f"  [yellow]WHERE object = '{parent}'  →  0 infons[/]\n"
        f"  [dim]All data stored under children: senkaku, japan, china...[/]\n\n"
        f"[dim]With descendant expansion:[/]\n"
        f"  [{SUPPORT_COLOR}]WHERE object IN ('{parent}', 'china', 'japan', 'senkaku', ...)[/]\n"
        f"  [{SUPPORT_COLOR}]→ finds all {len(descendants)} child-level infons[/]",
        title="[bold]The Problem & Solution[/]",
        border_style="yellow",
        box=box.ROUNDED,
        width=68,
    ))
    console.print()

    # Run both queries
    flash_text("QUERYING AT TWO HIERARCHY LEVELS", style="bold white on dark_blue")
    console.print()

    result_broad = cog.query(broad_q, top_k=50, include_chains=True)
    result_specific = cog.query(specific_q, top_k=50, include_chains=True)

    # Broad query results
    broad_anchors = set()
    for inf in result_broad.infons:
        broad_anchors.update([inf.subject, inf.predicate, inf.object])
    child_hits = broad_anchors & set(descendants)

    console.print(f"  [bold {ACCENT}]Broad Query:[/] \"{broad_q}\"")
    console.print(f"    Infons retrieved:  [bold]{len(result_broad.infons)}[/]")
    console.print(f"    Constraints:       [bold]{len(result_broad.constraints)}[/]")
    console.print(f"    Child anchors hit: [{SUPPORT_COLOR}]{sorted(child_hits)[:6]}[/]")
    if result_broad.infons:
        top = result_broad.infons[0]
        sc = ANCHOR_COLORS.get(cog.schema.types.get(top.subject, ""), "dim")
        pc = ANCHOR_COLORS.get(cog.schema.types.get(top.predicate, ""), "dim")
        oc = ANCHOR_COLORS.get(cog.schema.types.get(top.object, ""), "dim")
        console.print(f"    Top infon:         [{sc}]{top.subject}[/] ──[{pc}]{top.predicate}[/]──▶ [{oc}]{top.object}[/]")
        console.print(f"                       [dim]\"{escape(top.sentence[:60])}...\"[/]")
    console.print()

    # Specific query results
    console.print(f"  [bold {ACCENT}]Specific Query:[/] \"{specific_q}\"")
    console.print(f"    Infons retrieved:  [bold]{len(result_specific.infons)}[/]")
    console.print(f"    [dim]No expansion needed — SPLADE resolves 'Senkaku' directly[/]")
    if result_specific.infons:
        top = result_specific.infons[0]
        sc = ANCHOR_COLORS.get(cog.schema.types.get(top.subject, ""), "dim")
        pc = ANCHOR_COLORS.get(cog.schema.types.get(top.predicate, ""), "dim")
        oc = ANCHOR_COLORS.get(cog.schema.types.get(top.object, ""), "dim")
        console.print(f"    Top infon:         [{sc}]{top.subject}[/] ──[{pc}]{top.predicate}[/]──▶ [{oc}]{top.object}[/]")
    console.print()

    # Side-by-side comparison
    tbl = Table(
        title="[bold]Hierarchy Expansion in Action[/]",
        border_style=ACCENT,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    tbl.add_column("", width=22, style="dim", justify="right")
    tbl.add_column("Broad (parent)", width=20, justify="center")
    tbl.add_column("Specific (leaf)", width=20, justify="center")

    tbl.add_row("Query level", f"[{ACCENT}]{parent}[/]", f"[{ACCENT}]senkaku[/]")
    tbl.add_row("Expansion", f"[{SUPPORT_COLOR}]→ {len(descendants)} descendants[/]", "[dim]none (leaf)[/]")
    tbl.add_row("Infons found", f"[bold]{len(result_broad.infons)}[/]", f"[bold]{len(result_specific.infons)}[/]")
    tbl.add_row("Constraints", f"[bold]{len(result_broad.constraints)}[/]", f"[bold]{len(result_specific.constraints)}[/]")
    tbl.add_row("NEXT chains", f"[bold]{len(result_broad.edges)}[/]", f"[bold]{len(result_specific.edges)}[/]")

    console.print(tbl)
    console.print()

    for bullet in [
        "SPLADE determines the specificity level — faithful to the user's words",
        "Hierarchy expansion happens at query time, not encoding time",
        "Parent nodes expand downward — leaf nodes never expand upward",
        "Same mechanism works for DIMEFIL domains: 'military' → all subtypes",
    ]:
        console.print(f"    [{SUPPORT_COLOR}]•[/] {bullet}")
    console.print()


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: DEMPSTER-SHAFER
# ═══════════════════════════════════════════════════════════════════════

def render_dempster_shafer(cog, all_infons):
    from cognition.dempster_shafer import (
        mass_from_polarity, combine_dempster, verify_claim,
    )
    from cognition import Infon

    console.print()
    console.rule(f"[bold white]STEP 5[/]  [{ACCENT}]DEMPSTER-SHAFER BELIEF[/]", style=ACCENT)
    console.print("  [dim]Principled fusion of conflicting evidence[/]\n")

    console.print(f"  [dim]Frame of discernment:[/]  Θ = {{ [{SUPPORT_COLOR}]SUPPORTS[/], "
                  f"[{REFUTE_COLOR}]REFUTES[/], [{UNCERTAIN_COLOR}]UNCERTAIN[/] }}")
    console.print(f"  [dim]Each evidence source → mass function m: 2^Θ → \\[0,1][/]\n")

    # Two opposing sources — from DOMAIN config
    ex_s = DOMAIN["example_subject"]
    ex_p = DOMAIN["example_predicate"]
    ex_o = DOMAIN["example_object"]

    affirm = Infon(subject=ex_s, predicate=ex_p, object=ex_o,
                   polarity=1, confidence=DOMAIN["ds_affirm_confidence"],
                   sentence=DOMAIN["ds_affirm_sentence"])
    negate = Infon(subject=ex_s, predicate=ex_p, object=ex_o,
                   polarity=0, confidence=DOMAIN["ds_negate_confidence"],
                   sentence=DOMAIN["ds_negate_sentence"])

    m_pos = mass_from_polarity(affirm)
    m_neg = mass_from_polarity(negate)

    aff_display = escape(DOMAIN["ds_affirm_sentence"])
    neg_display = escape(DOMAIN["ds_negate_sentence"])
    neg_short = neg_display[:40] + "..." if len(neg_display) > 40 else neg_display

    src_tbl = Table(
        title="[bold]Two Opposing Evidence Sources[/]",
        border_style=ACCENT,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    src_tbl.add_column("", width=10)
    src_tbl.add_column("Sentence", width=44)
    src_tbl.add_column("m(S)", width=8, justify="right", style=SUPPORT_COLOR)
    src_tbl.add_column("m(R)", width=8, justify="right", style=REFUTE_COLOR)
    src_tbl.add_column("m(θ)", width=8, justify="right", style="dim")
    src_tbl.add_row(
        f"[{SUPPORT_COLOR}]Source A[/]",
        f"[italic]\"{aff_display}\"[/]",
        f"{m_pos.supports:.3f}", f"{m_pos.refutes:.3f}", f"{m_pos.theta:.3f}",
    )
    src_tbl.add_row(
        f"[{REFUTE_COLOR}]Source B[/]",
        f"[italic]\"{neg_short}\"[/]",
        f"{m_neg.supports:.3f}", f"{m_neg.refutes:.3f}", f"{m_neg.theta:.3f}",
    )
    console.print(src_tbl)
    console.print()

    # Combination
    flash_text("COMBINING VIA DEMPSTER'S RULE", style="bold white on dark_red")
    console.print()

    combined = combine_dempster(m_pos, m_neg)

    console.print(Panel(
        f"[dim]         Source A              Source B[/]\n"
        f"[dim]    ┌──────────────┐     ┌──────────────┐[/]\n"
        f"[dim]    │[/] [{SUPPORT_COLOR}]m(S)={m_pos.supports:.2f}[/]      [dim]│     │[/] [{REFUTE_COLOR}]m(R)={m_neg.refutes:.2f}[/]      [dim]│[/]\n"
        f"[dim]    │[/] [dim]m(θ)={m_pos.theta:.2f}[/]      [dim]│     │[/] [dim]m(θ)={m_neg.theta:.2f}[/]      [dim]│[/]\n"
        f"[dim]    └──────┬───────┘     └──────┬───────┘[/]\n"
        f"[dim]           └──────┬─────────────┘[/]\n"
        f"[dim]                  ▼[/]\n"
        f"    [bold]m(S)∩m(R) = ∅ → conflict[/]\n"
        f"    [dim]normalize by 1/(1−conflict)[/]\n"
        f"[dim]                  ▼[/]\n"
        f"    [{SUPPORT_COLOR}]m(S) = {combined.supports:.3f}[/]   "
        f"[{REFUTE_COLOR}]m(R) = {combined.refutes:.3f}[/]   "
        f"[dim]m(θ) = {combined.theta:.3f}[/]",
        title="[bold]Dempster's Rule of Combination[/]",
        border_style=ACCENT, box=box.ROUNDED, width=62,
    ))
    console.print()
    console.print(f"  [dim italic]Refuting evidence ({combined.refutes:.2f}) outweighs support ({combined.supports:.2f})[/]")
    console.print(f"  [dim italic]— DS doesn't average, it fuses and redistributes conflict mass.[/]")
    console.print()

    # Full claim verification
    claim = DOMAIN["claim"]
    claim_anchors = cog.encoder.encode_single(claim)

    flash_text("FULL CLAIM VERIFICATION  (4 sources × top infons)", style="bold white on dark_blue")
    console.print()

    verdict = verify_claim(all_infons, claim_anchors=claim_anchors,
                           schema_types=cog.schema.types)

    console.print(f"  [bold]Claim:[/] \"{claim}\"")
    console.print(f"  [dim]Evidence sources per infon: polarity, triple alignment, anchor distance, confidence[/]")
    console.print(f"  [dim]Infons evaluated: {verdict.n_evidence}[/]\n")

    animate_bar("belief(SUPPORTS)", verdict.belief_supports, color=SUPPORT_COLOR)
    animate_bar("belief(REFUTES)", verdict.belief_refutes, color=REFUTE_COLOR)
    animate_bar("belief(UNCERTAIN)", verdict.belief_uncertain, color=UNCERTAIN_COLOR)
    console.print()

    vc = SUPPORT_COLOR if verdict.label == "SUPPORTS" else REFUTE_COLOR if verdict.label == "REFUTES" else UNCERTAIN_COLOR
    flash_text(f"VERDICT: {verdict.label}", style=f"bold white on {vc}")
    console.print()


# ═══════════════════════════════════════════════════════════════════════
# STEP 6: GRAPH MCTS
# ═══════════════════════════════════════════════════════════════════════

def _mass_label(m):
    if m.supports > m.refutes and m.supports > 0.08:
        return f"[{SUPPORT_COLOR}]supports[/]"
    elif m.refutes > m.supports and m.refutes > 0.08:
        return f"[{REFUTE_COLOR}]refutes[/]"
    return "[dim]neutral[/]"


def _build_mcts_tree(root, cog):
    """Build a Rich Tree from the MCTS traversal tree."""
    rm = root.belief_mass
    root_label = Text()
    root_label.append("◉ ROOT  ", style=f"bold {ACCENT}")
    root_label.append(f"anchors={{{', '.join(root.anchor_path[:4])}}}  ", style="dim")
    root_label.append(f"infons={len(root.infons)}  visits={root.visit_count}", style="dim")

    tree = Tree(root_label, guide_style="dim")
    tree.add(Text.from_markup(
        f"[dim]belief:[/] [{SUPPORT_COLOR}]S={rm.supports:.3f}[/]  "
        f"[{REFUTE_COLOR}]R={rm.refutes:.3f}[/]  [dim]θ={rm.theta:.3f}[/]"
    ))

    for child in root.children:
        cm = child.belief_mass
        cl = _mass_label(cm)
        anchor_tail = child.anchor_path[-1] if child.anchor_path else "?"
        ac = ANCHOR_COLORS.get(cog.schema.types.get(anchor_tail, ""), "dim")

        edge_type = "NEXT" if child.edges_followed else "shared"
        gap = ""
        if child.edges_followed:
            g = child.edges_followed[0].metadata.get("gap_days", "?")
            gap = f"  [dim]gap={g}d[/]"

        child_label = Text.from_markup(
            f"[{ac} bold]{anchor_tail}[/]  [dim]({edge_type}){gap}[/]  "
            f"[dim]infons={len(child.infons)}[/]  {cl}  "
            f"[{SUPPORT_COLOR}]S={cm.supports:.2f}[/] [{REFUTE_COLOR}]R={cm.refutes:.2f}[/]"
        )
        child_branch = tree.add(child_label)

        for gc in child.children[:5]:
            gm = gc.belief_mass
            gl = _mass_label(gm)
            gc_anchor = gc.anchor_path[-1] if gc.anchor_path else "?"
            gc_color = ANCHOR_COLORS.get(cog.schema.types.get(gc_anchor, ""), "dim")
            n_sub = len(gc.children)
            sub_note = f"  [dim](+{n_sub})[/]" if n_sub > 0 else ""
            child_branch.add(Text.from_markup(
                f"[{gc_color}]{gc_anchor}[/]  {gl}  "
                f"[{SUPPORT_COLOR}]S={gm.supports:.2f}[/] "
                f"[{REFUTE_COLOR}]R={gm.refutes:.2f}[/]{sub_note}"
            ))

        if len(child.children) > 5:
            child_branch.add(Text(f"... +{len(child.children) - 5} more", style="dim"))

    return tree


def render_mcts(cog, all_infons):
    from cognition.graph_mcts import GraphMCTS

    console.print()
    console.rule(f"[bold white]STEP 6[/]  [{ACCENT}]GRAPH MCTS[/]", style=ACCENT)
    console.print("  [dim]AlphaGo-style tree search over the knowledge graph[/]\n")

    query = DOMAIN["mcts_query"]
    hops = DOMAIN["mcts_hops"]

    console.print(f"  [dim]The question requires connecting facts across documents and time:[/]\n")
    hop_parts = [f"[{ACCENT}]{label}[/] ({date})" for label, date in hops]
    console.print(f"    " + " ──▶ ".join(hop_parts) + "\n")

    # MCTS loop diagram
    console.print(Panel(
        "[bold white]SELECT[/]   ──▶  UCB1 picks most promising unexplored leaf\n"
        "[dim]  │          exploitation + 1.4 × √(ln(N)/nᵢ)[/]\n"
        "[dim]  ▼[/]\n"
        "[bold white]EXPAND[/]   ──▶  follow NEXT edges + shared-anchor links\n"
        "[dim]  │          top 5 children per expansion[/]\n"
        "[dim]  ▼[/]\n"
        "[bold white]EVALUATE[/] ──▶  NLI head scores infons against query\n"
        "[dim]  │          → produces mass function per infon[/]\n"
        "[dim]  ▼[/]\n"
        "[bold white]BACKPROP[/] ──▶  DS combination up the traversal tree\n"
        "[dim]  ╰──────▶  repeat until entropy converges[/]",
        title="[bold]MCTS LOOP[/]  [dim](AlphaGo-style, over knowledge graph)[/]",
        border_style=ACCENT, box=box.ROUNDED, width=64,
    ))
    console.print()
    console.print(f"  [bold]Query:[/] \"{query}\"\n")

    mcts = GraphMCTS(
        store=cog.store, encoder=cog.encoder, schema=cog.schema,
        max_iterations=8, max_depth=4, exploration_bias=1.4,
    )

    with Progress(
        SpinnerColumn("dots12", style=ACCENT),
        TextColumn(f"[{ACCENT}]Running MCTS traversal...[/]"),
        console=console, transient=True,
    ) as progress:
        progress.add_task("", total=None)
        result = mcts.search(query, verbose=False)

    flash_text("MCTS COMPLETE", style="bold white on dark_green")
    console.print()

    # Tree visualization
    root = result.traversal_tree
    if root:
        tree = _build_mcts_tree(root, cog)
        console.print(Panel(
            tree,
            title="[bold]MCTS Traversal Tree[/]",
            border_style=ACCENT,
            box=box.ROUNDED,
            width=80,
        ))
    console.print()

    # Iteration log table
    if result.iteration_log:
        iter_tbl = Table(
            title="[bold]Iteration Log[/]  [dim](belief convergence)[/]",
            border_style=ACCENT,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
        )
        iter_tbl.add_column("Iter", width=5, justify="center")
        iter_tbl.add_column("Belief", width=24)
        iter_tbl.add_column("S", width=7, justify="right", style=SUPPORT_COLOR)
        iter_tbl.add_column("R", width=7, justify="right", style=REFUTE_COLOR)
        iter_tbl.add_column("θ", width=7, justify="right", style="dim")
        iter_tbl.add_column("H", width=6, justify="right", style="dim")
        iter_tbl.add_column("+nodes", width=7, justify="right", style="dim")

        for entry in result.iteration_log:
            rb = entry["root_belief"]
            sw = int(rb["supports"] * 24)
            rw = int(rb["refutes"] * 24)
            rest = max(0, 24 - sw - rw)
            mini_bar = (f"[{SUPPORT_COLOR}]{'█' * sw}[/]"
                        f"[{REFUTE_COLOR}]{'█' * rw}[/]"
                        f"[dim]{'░' * rest}[/]")
            iter_tbl.add_row(
                str(entry["iteration"]), mini_bar,
                f"{rb['supports']:.3f}", f"{rb['refutes']:.3f}", f"{rb['theta']:.3f}",
                f"{entry['root_entropy']:.2f}", f"+{entry['new_children']}",
            )
        console.print(iter_tbl)
    console.print()

    # Evidence chains
    if result.chains_discovered:
        console.print(f"  [bold {ACCENT}]Discovered Evidence Chains[/]")
        console.print(f"  [dim]{'─' * 64}[/]")
        seen = set()
        num = 0
        for chain in result.chains_discovered:
            ck = tuple(chain)
            if ck in seen:
                continue
            seen.add(ck)
            num += 1
            if num > 5:
                break
            parts = []
            for anchor in chain:
                ac = ANCHOR_COLORS.get(cog.schema.types.get(anchor, ""), "dim")
                parts.append(f"[{ac}]{anchor}[/]")
            console.print(f"    [dim]Chain {num}:[/] " + " [dim]→[/] ".join(parts))
        console.print()

    # Final verdict bars
    m = result.combined_mass
    animate_bar("SUPPORTS", m.supports, color=SUPPORT_COLOR)
    animate_bar("REFUTES", m.refutes, color=REFUTE_COLOR)
    animate_bar("uncertainty", m.uncertain + m.theta, color=UNCERTAIN_COLOR)
    console.print()

    vc = SUPPORT_COLOR if result.verdict == "SUPPORTS" else REFUTE_COLOR if result.verdict == "REFUTES" else UNCERTAIN_COLOR
    flash_text(f"VERDICT: {result.verdict}", style=f"bold white on {vc}")
    console.print(f"  [dim]{result.nodes_explored} nodes explored, {result.infons_evaluated} infons evaluated, "
                  f"{result.iterations} iterations in {result.elapsed_s:.2f}s[/]")
    console.print()

    return result, query


# ═══════════════════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════════════════

def render_comparison(cog, mcts_result, query):
    from cognition.dempster_shafer import verify_claim

    flat_result = cog.query(query, top_k=20, include_chains=True)
    flat_verdict = verify_claim(
        flat_result.infons,
        claim_anchors=flat_result.anchors_activated,
        schema_types=cog.schema.types,
    )

    flat_chains = len([e for e in flat_result.edges if e.edge_type == "NEXT"])
    mm = mcts_result.combined_mass

    tbl = Table(
        title="[bold]Flat Retrieval vs Graph MCTS[/]",
        border_style=ACCENT,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    tbl.add_column("", width=20, style="dim", justify="right")
    tbl.add_column("Flat top-k", width=20, justify="center")
    tbl.add_column("Graph MCTS", width=20, justify="center")

    fvc = SUPPORT_COLOR if flat_verdict.label == "SUPPORTS" else REFUTE_COLOR if flat_verdict.label == "REFUTES" else UNCERTAIN_COLOR
    mvc = SUPPORT_COLOR if mcts_result.verdict == "SUPPORTS" else REFUTE_COLOR if mcts_result.verdict == "REFUTES" else UNCERTAIN_COLOR

    tbl.add_row("Verdict", f"[{fvc} bold]{flat_verdict.label}[/]", f"[{mvc} bold]{mcts_result.verdict}[/]")
    tbl.add_row("belief(S)", f"[{SUPPORT_COLOR}]{flat_verdict.belief_supports:.3f}[/]",
                f"[{SUPPORT_COLOR}]{mm.supports:.3f}[/]")
    tbl.add_row("belief(R)", f"[{REFUTE_COLOR}]{flat_verdict.belief_refutes:.3f}[/]",
                f"[{REFUTE_COLOR}]{mm.refutes:.3f}[/]")
    tbl.add_row("Infons evaluated", str(flat_verdict.n_evidence), str(mcts_result.infons_evaluated))
    tbl.add_row("Evidence chains", str(flat_chains), str(len(mcts_result.chains_discovered)))
    tbl.add_row("Nodes explored", "[dim]—[/]", str(mcts_result.nodes_explored))
    tbl.add_row("Time", "[dim]—[/]", f"{mcts_result.elapsed_s:.2f}s")

    console.print(tbl)
    console.print()

    for line in DOMAIN["comparison_explanation"].split("\n"):
        console.print(f"  [dim italic]{line}[/]")
    console.print()


# ═══════════════════════════════════════════════════════════════════════
# STEP 7: CONTRARY VIEW
# ═══════════════════════════════════════════════════════════════════════

def render_contrary_view(cog, all_infons):
    from cognition.dempster_shafer import verify_claim

    console.print()
    console.rule(f"[bold white]STEP 7[/]  [{ACCENT}]CONTRARY VIEW[/]", style=ACCENT)
    console.print("  [dim]Invert the evidential lens — same data, opposite thesis[/]\n")

    claim = DOMAIN["contrary_claim"]
    claim_anchors = cog.encoder.encode_single(claim)

    # Retrieve relevant infons via query so verify_claim sees the right evidence
    relevant = cog.query(claim, top_k=50, include_chains=False)
    evidence = relevant.infons if relevant.infons else all_infons

    # Normal verdict
    normal_verdict = verify_claim(
        evidence, claim_anchors=claim_anchors,
        schema_types=cog.schema.types,
    )
    # Contrary verdict
    contrary_verdict = verify_claim(
        evidence, claim_anchors=claim_anchors,
        schema_types=cog.schema.types,
        contrary=True,
    )

    console.print(f"  [bold]Claim:[/] \"{claim}\"\n")

    # How it works
    console.print(Panel(
        f"[dim]MassFunction.invert() swaps SUPPORTS ↔ REFUTES:[/]\n\n"
        f"  [bold]Normal frame[/]\n"
        f"    Evidence that [italic]confirms[/] the claim → [{SUPPORT_COLOR}]SUPPORTS[/]\n"
        f"    Evidence that [italic]contradicts[/] the claim → [{REFUTE_COLOR}]REFUTES[/]\n\n"
        f"  [bold]Contrary frame[/]  [dim](contrary=True)[/]\n"
        f"    Evidence that [italic]confirms[/] the claim → [{REFUTE_COLOR}]REFUTES[/] the contrary\n"
        f"    Evidence that [italic]contradicts[/] the claim → [{SUPPORT_COLOR}]SUPPORTS[/] the contrary\n\n"
        f"  [dim]Query-time only — nothing stored changes.[/]",
        title="[bold]How Contrary Inversion Works[/]",
        border_style=ACCENT, box=box.ROUNDED, width=68,
    ))
    console.print()

    # Side-by-side verdicts
    flash_text("NORMAL vs CONTRARY VERDICTS", style="bold white on dark_red")
    console.print()

    tbl = Table(
        title="[bold]Same Evidence, Opposite Thesis[/]",
        border_style=ACCENT,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    tbl.add_column("", width=20, style="dim", justify="right")
    tbl.add_column("Normal", width=22, justify="center")
    tbl.add_column("Contrary", width=22, justify="center")

    nvc = SUPPORT_COLOR if normal_verdict.label == "SUPPORTS" else REFUTE_COLOR if normal_verdict.label == "REFUTES" else UNCERTAIN_COLOR
    cvc = SUPPORT_COLOR if contrary_verdict.label == "SUPPORTS" else REFUTE_COLOR if contrary_verdict.label == "REFUTES" else UNCERTAIN_COLOR

    tbl.add_row("Verdict", f"[{nvc} bold]{normal_verdict.label}[/]", f"[{cvc} bold]{contrary_verdict.label}[/]")
    tbl.add_row("belief(SUPPORTS)", f"[{SUPPORT_COLOR}]{normal_verdict.belief_supports:.3f}[/]",
                f"[{SUPPORT_COLOR}]{contrary_verdict.belief_supports:.3f}[/]")
    tbl.add_row("belief(REFUTES)", f"[{REFUTE_COLOR}]{normal_verdict.belief_refutes:.3f}[/]",
                f"[{REFUTE_COLOR}]{contrary_verdict.belief_refutes:.3f}[/]")
    tbl.add_row("belief(UNCERTAIN)", f"[{UNCERTAIN_COLOR}]{normal_verdict.belief_uncertain:.3f}[/]",
                f"[{UNCERTAIN_COLOR}]{contrary_verdict.belief_uncertain:.3f}[/]")
    tbl.add_row("Infons evaluated", str(normal_verdict.n_evidence), str(contrary_verdict.n_evidence))

    console.print(tbl)
    console.print()

    # Animated bars for contrary
    console.print(f"  [bold {ACCENT}]Contrary Belief Distribution[/]")
    animate_bar("SUPPORTS (contrary)", contrary_verdict.belief_supports, color=SUPPORT_COLOR)
    animate_bar("REFUTES (contrary)", contrary_verdict.belief_refutes, color=REFUTE_COLOR)
    animate_bar("UNCERTAIN", contrary_verdict.belief_uncertain, color=UNCERTAIN_COLOR)
    console.print()

    # Query-level contrary ranking
    contrary_q = DOMAIN["contrary_query"]
    flash_text("CONTRARY QUERY RANKING", style="bold white on dark_blue")
    console.print()

    normal_result = cog.query(contrary_q, top_k=10, include_chains=False)
    contrary_result = cog.query(contrary_q, top_k=10, include_chains=False, contrary=True)

    rank_tbl = Table(
        title=f"[bold]Query:[/] \"{contrary_q}\"",
        border_style=ACCENT,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    rank_tbl.add_column("#", width=3, justify="right")
    rank_tbl.add_column("Normal Ranking", width=34)
    rank_tbl.add_column("Contrary Ranking", width=34)

    n_rows = min(5, len(normal_result.infons), len(contrary_result.infons))
    for i in range(n_rows):
        ni = normal_result.infons[i]
        ci = contrary_result.infons[i]
        np_sym = "+" if ni.polarity == 1 else f"[{REFUTE_COLOR}]−[/]"
        cp_sym = "+" if ci.polarity == 1 else f"[{REFUTE_COLOR}]−[/]"
        n_v = normal_result.valence.get(ni.infon_id, 0)
        c_v = contrary_result.valence.get(ci.infon_id, 0)
        rank_tbl.add_row(
            str(i + 1),
            f"{np_sym} {ni.predicate},{ni.subject[:8]}→{ni.object[:8]}  [dim]v={n_v:+.2f}[/]",
            f"{cp_sym} {ci.predicate},{ci.subject[:8]}→{ci.object[:8]}  [dim]v={c_v:+.2f}[/]",
        )

    console.print(rank_tbl)
    console.print()

    for bullet in [
        "Same infons, same evidence — only the ranking lens changes",
        "Negated infons (polarity=0) float to the top in contrary mode",
        "Valence signs flip: what was positive is now negative",
        "DS verdict inverts: SUPPORTS ↔ REFUTES, NEI stays NEI",
        "Enables red-team / devil's-advocate analysis over any claim",
    ]:
        console.print(f"    [{SUPPORT_COLOR}]•[/] {bullet}")
    console.print()


# ═══════════════════════════════════════════════════════════════════════
# STEP 8: KANO + CONJOINT
# ═══════════════════════════════════════════════════════════════════════

KANO_COLORS = {
    "must_be": "red",
    "one_dimensional": "bright_blue",
    "attractive": "green",
    "indifferent": "dim",
    "reverse": "bright_red",
}
KANO_LABELS = {
    "must_be": "MUST-BE",
    "one_dimensional": "LINEAR",
    "attractive": "DELIGHTER",
    "indifferent": "INDIFFERENT",
    "reverse": "REVERSE",
}


def render_kano_conjoint(sa_results):
    console.print()
    console.rule(f"[bold white]STEP 8[/]  [{ACCENT}]KANO + CONJOINT ANALYSIS[/]", style=ACCENT)
    console.print("  [dim]Classify anchors by market impact — no surveys, just graph topology[/]\n")

    kano = sa_results["kano"]
    conjoint = sa_results["conjoint"]

    # Kano explanation
    console.print(Panel(
        f"[bold]Kano Model[/]  [dim]from adoption rate × polarity delta[/]\n\n"
        f"  [red]MUST-BE[/]       adoption ≥ 90%  — table stakes, absence hurts\n"
        f"  [bright_blue]LINEAR[/]        adoption 30-90% — more is better\n"
        f"  [green]DELIGHTER[/]     adoption 5-30%  — surprise value\n"
        f"  [dim]INDIFFERENT[/]   adoption < 5%   — nobody cares\n"
        f"  [bright_red]REVERSE[/]       negative delta  — presence hurts\n\n"
        f"  [bold]Conjoint[/]  [dim]part-worth utility = mean(importance × polarity × confidence)[/]",
        title="[bold]How It Works[/]",
        border_style=ACCENT, box=box.ROUNDED, width=68,
    ))
    console.print()

    # Kano classification table
    flash_text("KANO CLASSIFICATION", style="bold white on dark_blue")
    console.print()

    tbl = Table(
        title="[bold]Anchor Kano Classification[/]",
        border_style=ACCENT, box=box.ROUNDED,
        show_header=True, header_style="bold",
    )
    tbl.add_column("Anchor", width=20, style="bold white")
    tbl.add_column("Class", width=14)
    tbl.add_column("Adoption", width=12, justify="right")
    tbl.add_column("Sat.Δ", width=10, justify="right")
    tbl.add_column("Score", width=10, justify="right")

    for k in kano[:12]:
        kc = KANO_COLORS.get(k.kano_class, "dim")
        kl = KANO_LABELS.get(k.kano_class, k.kano_class)
        tbl.add_row(
            k.anchor,
            f"[{kc}]{kl}[/]",
            f"{k.adoption_rate:.1%}",
            f"[{'green' if k.satisfaction_delta > 0 else 'red'}]{k.satisfaction_delta:+.3f}[/]",
            f"[{kc}]{k.score:+.4f}[/]",
        )
    console.print(tbl)
    console.print()

    # Conjoint utility bars
    console.print(f"  [bold {ACCENT}]Conjoint Part-Worth Utilities[/]  [dim](top 8 by importance)[/]")
    for c in conjoint[:8]:
        color = SUPPORT_COLOR if c.utility > 0 else REFUTE_COLOR
        animate_bar(f"{c.anchor} (imp={c.importance:.2f})", abs(c.utility),
                    max_val=max(abs(conjoint[0].utility), 0.01) if conjoint else 1.0,
                    color=color)
    console.print()

    # Distribution summary
    from collections import Counter
    dist = Counter(k.kano_class for k in kano)
    parts = [f"[{KANO_COLORS.get(k, 'dim')}]{KANO_LABELS.get(k, k)}: {v}[/]" for k, v in dist.most_common()]
    console.print(f"  [dim]Distribution:[/]  {' · '.join(parts)}")
    console.print()


# ═══════════════════════════════════════════════════════════════════════
# STEP 9: FEATURE GAP FUNCTOR
# ═══════════════════════════════════════════════════════════════════════

def render_feature_gap(sa_results, cog):
    console.print()
    console.rule(f"[bold white]STEP 9[/]  [{ACCENT}]FEATURE GAP FUNCTOR[/]", style=ACCENT)
    console.print("  [dim]F: Schema → Discourse — measure what's defined vs what's discussed[/]\n")

    gaps = sa_results["feature_gap"]
    fgf = FeatureGapFunctor()
    summary = fgf.summary(gaps)

    # Functor diagram
    console.print(Panel(
        f"[bold]Schema Set (Ontology)[/]        [bold]Discourse Set (Infons)[/]\n"
        f"  [dim]Anchors you defined[/]            [dim]Anchors SPLADE activated[/]\n"
        f"  [dim]in the JSON schema[/]             [dim]from ingested documents[/]\n\n"
        f"            [bold white]F: Schema ─────▶ Discourse[/]\n\n"
        f"  [{REFUTE_COLOR}]Silent Value[/]                  [{REFUTE_COLOR}]Hallucinated Value[/]\n"
        f"  [dim]Schema − Discourse[/]              [dim]Discourse − Schema[/]\n"
        f"  [dim]Defined but never discussed[/]     [dim]Discussed but never defined[/]",
        title="[bold]The Functor[/]",
        border_style=ACCENT, box=box.ROUNDED, width=62,
    ))
    console.print()

    flash_text("COMPUTING GAP", style="bold white on dark_red")
    console.print()

    # Summary stats
    stats_tbl = Table.grid(padding=(0, 3))
    stats_tbl.add_column(style="dim", width=20, justify="right")
    stats_tbl.add_column(style=f"bold {ACCENT}")
    stats_tbl.add_row("Total features", str(summary["total_features"]))
    stats_tbl.add_row("Overlap", f"[{SUPPORT_COLOR}]{summary['overlap']} ({summary['overlap_pct']}%)[/]")
    stats_tbl.add_row("Silent (wasted)", f"[{REFUTE_COLOR}]{summary['silent']} ({summary['silent_pct']}%)[/]")
    stats_tbl.add_row("Hallucinated", f"[{UNCERTAIN_COLOR}]{summary['hallucinated']} ({summary['hallucinated_pct']}%)[/]")
    console.print(stats_tbl)
    console.print()

    # Gap table (top entries by type)
    tbl = Table(
        title="[bold]Feature Gap Detail[/]",
        border_style=ACCENT, box=box.ROUNDED,
        show_header=True, header_style="bold",
    )
    tbl.add_column("Anchor", width=22, style="bold white")
    tbl.add_column("Gap Type", width=14)
    tbl.add_column("Schema", width=8, justify="center")
    tbl.add_column("Discourse", width=10, justify="center")
    tbl.add_column("Mentions", width=10, justify="right")

    shown = 0
    for g in gaps:
        if shown >= 10:
            break
        gc = (REFUTE_COLOR if g.gap_type == "silent"
              else UNCERTAIN_COLOR if g.gap_type == "hallucinated"
              else SUPPORT_COLOR if g.gap_type == "overlap"
              else "dim")
        tbl.add_row(
            g.anchor,
            f"[{gc}]{g.gap_type.upper()}[/]",
            f"[{SUPPORT_COLOR}]✓[/]" if g.spec_present else "[dim]—[/]",
            f"[{SUPPORT_COLOR}]✓[/]" if g.discourse_present else "[dim]—[/]",
            str(g.discourse_count),
        )
        shown += 1
    console.print(tbl)
    console.print()

    overlap_r = fgf.overlap_ratio(gaps) if gaps else 0.0
    animate_bar("Schema-Discourse Overlap", overlap_r, color=SUPPORT_COLOR)
    animate_bar("Silent (wasted R&D)", summary["silent_pct"] / 100.0 if gaps else 0, color=REFUTE_COLOR)
    console.print()


# ═══════════════════════════════════════════════════════════════════════
# STEP 10: GHOSTS + POLARIZATION
# ═══════════════════════════════════════════════════════════════════════

def render_ghost_polarization(sa_results, cog):
    console.print()
    console.rule(f"[bold white]STEP 10[/]  [{ACCENT}]GHOST DETECTION + POLARIZATION INDEX[/]", style=ACCENT)
    console.print("  [dim]Dead anchors (CAC → ∞) and sentiment contradiction density (H¹)[/]\n")

    ghosts = sa_results["ghosts"]
    polarization = sa_results["polarization"]

    # ── Ghost Detection ──
    flash_text("GHOST DETECTION", style="bold white on dark_red")
    console.print()

    gd = GhostDetector()
    ghost_rate = gd.ghost_rate(ghosts)
    by_type = gd.by_type(ghosts)

    ghost_count = sum(1 for g in ghosts if g.is_ghost)
    live_count = len(ghosts) - ghost_count

    console.print(Panel(
        f"[dim]A ghost is a schema anchor with[/] [bold]zero infons[/] [dim]referencing it.[/]\n"
        f"[dim]Defined in the ontology but invisible to the corpus — CAC → ∞.[/]\n\n"
        f"  [{SUPPORT_COLOR}]Live anchors:[/]   [bold]{live_count}[/]  [dim](at least 1 infon)[/]\n"
        f"  [{REFUTE_COLOR}]Ghost anchors:[/]  [bold]{ghost_count}[/]  [dim](zero infons)[/]\n"
        f"  [dim]Ghost rate:[/]      [bold]{ghost_rate:.1%}[/]",
        title="[bold]Graph Laplacian — Disconnected Nodes[/]",
        border_style=ACCENT, box=box.ROUNDED, width=62,
    ))
    console.print()

    # Ghost by type table
    if by_type:
        tbl = Table(
            title="[bold]Ghost Rate by Anchor Type[/]",
            border_style=ACCENT, box=box.ROUNDED,
            show_header=True, header_style="bold",
        )
        tbl.add_column("Type", width=14)
        tbl.add_column("Total", width=8, justify="right")
        tbl.add_column("Ghosts", width=8, justify="right", style=REFUTE_COLOR)
        tbl.add_column("Ghost %", width=10, justify="right")
        tbl.add_column("", width=20)

        for atype, info in by_type.items():
            pct = info["ghost_pct"]
            bw = int(min(pct / 100, 1.0) * 16)
            bar = f"[{REFUTE_COLOR}]{'█' * bw}{'░' * (16 - bw)}[/]"
            tbl.add_row(atype, str(info["total"]), str(info["ghosts"]),
                        f"[{REFUTE_COLOR if pct > 50 else UNCERTAIN_COLOR if pct > 20 else 'dim'}]{pct:.1f}%[/]",
                        bar)
        console.print(tbl)
        console.print()

    # Top ghost anchors
    top_ghosts = [g for g in ghosts if g.is_ghost][:8]
    if top_ghosts:
        console.print(f"  [bold {ACCENT}]Sample Ghosts[/]  [dim](schema-defined, zero signals)[/]")
        for g in top_ghosts:
            tc = ANCHOR_COLORS.get(g.anchor_type, "dim")
            console.print(f"    [{REFUTE_COLOR}]✗[/] [{tc}]{g.anchor}[/]  [dim]type={g.anchor_type}[/]")
        console.print()

    # ── Polarization Index ──
    flash_text("POLARIZATION INDEX (SHEAF H¹)", style="bold white on dark_blue")
    console.print()

    console.print(Panel(
        f"[dim]H¹ ≈ min(bullish, bearish) — the irreducible contradiction.[/]\n"
        f"[dim]Signals that can't be resolved by majority vote.[/]\n\n"
        f"  [dim]H¹ = 0  →[/]  trivial consensus (boring or ignored)\n"
        f"  [dim]H¹ > 0  →[/]  fractured identity or strategic polarization\n"
        f"  [dim]H¹ high →[/]  brand crisis [bold]or[/] cultural phenomenon",
        title="[bold]Sheaf Cohomology H¹[/]",
        border_style=ACCENT, box=box.ROUNDED, width=62,
    ))
    console.print()

    # Polarization table
    tbl = Table(
        title="[bold]Polarization by Anchor[/]  [dim](H¹ = min(bull, bear))[/]",
        border_style=ACCENT, box=box.ROUNDED,
        show_header=True, header_style="bold",
    )
    tbl.add_column("Anchor", width=20, style="bold white")
    tbl.add_column("Bull", width=6, justify="right", style=SUPPORT_COLOR)
    tbl.add_column("Bear", width=6, justify="right", style=REFUTE_COLOR)
    tbl.add_column("H¹", width=6, justify="right")
    tbl.add_column("Direction", width=12)
    tbl.add_column("", width=22)

    for p in polarization[:10]:
        max_count = max(p.total_count, 1)
        bull_w = int(p.bull_count / max_count * 10)
        bear_w = int(p.bear_count / max_count * 10)
        bar = f"[{SUPPORT_COLOR}]{'█' * bull_w}[/][{REFUTE_COLOR}]{'█' * bear_w}[/]{'░' * max(0, 10 - bull_w - bear_w)}"
        dc = (SUPPORT_COLOR if p.direction == "positive"
              else REFUTE_COLOR if p.direction == "negative"
              else UNCERTAIN_COLOR if p.direction == "balanced"
              else "dim")
        tbl.add_row(
            p.anchor,
            str(p.bull_count), str(p.bear_count),
            f"[bold {'bright_red' if p.h1 > 2 else UNCERTAIN_COLOR if p.h1 > 0 else 'dim'}]{p.h1}[/]",
            f"[{dc}]{p.direction}[/]",
            bar,
        )
    console.print(tbl)
    console.print()


# ═══════════════════════════════════════════════════════════════════════
# STEP 11: NARRATIVE LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════

def render_narrative(sa_results):
    console.print()
    console.rule(f"[bold white]STEP 11[/]  [{ACCENT}]NARRATIVE LIFECYCLE[/]", style=ACCENT)
    console.print("  [dim]Persistent homology on temporal chains — distinguish hype from structural shift[/]\n")

    narrative = sa_results["narrative"]

    console.print(Panel(
        f"[bold]β₀[/] [dim](connected components)[/]  fragmentation of the conversation\n"
        f"    [dim]High β₀ = scattered speculation  ·  Low β₀ = consensus forming[/]\n\n"
        f"[bold]β₁[/] [dim](loops / cycles)[/]          recurring narrative patterns\n"
        f"    [dim]Long-lived β₁ = permanent market feature  ·  Short β₁ = fad[/]\n\n"
        f"[bold]Phase Detection[/]  [dim]from signal density curve[/]\n"
        f"    [{SUPPORT_COLOR}]emerging[/] → [{ACCENT}]ramping[/] → [bold white]mature[/] → [{UNCERTAIN_COLOR}]decaying[/] → [{REFUTE_COLOR}]dead[/]",
        title="[bold]Persistent Homology on NEXT Chains[/]",
        border_style=ACCENT, box=box.ROUNDED, width=68,
    ))
    console.print()

    flash_text("NARRATIVE ANALYSIS", style="bold white on dark_green")
    console.print()

    # Main table
    tbl = Table(
        title="[bold]Narrative Lifecycle by Anchor[/]",
        border_style=ACCENT, box=box.ROUNDED,
        show_header=True, header_style="bold",
    )
    tbl.add_column("Anchor", width=22, style="bold white")
    tbl.add_column("Chain", width=7, justify="right")
    tbl.add_column("β₀", width=6, justify="right")
    tbl.add_column("β₁", width=6, justify="right")
    tbl.add_column("Phase", width=12)
    tbl.add_column("Velocity", width=9, justify="right")
    tbl.add_column("Gap Distribution", width=24)

    PHASE_COLORS = {
        "emerging": SUPPORT_COLOR,
        "ramping": ACCENT,
        "mature": "bold white",
        "decaying": UNCERTAIN_COLOR,
        "dead": REFUTE_COLOR,
        "unknown": "dim",
    }

    for n in narrative[:10]:
        pc = PHASE_COLORS.get(n.phase, "dim")
        # Mini gap histogram
        gap_parts = []
        for bucket, count in n.gap_distribution.items():
            if count > 0:
                gap_parts.append(f"[dim]{bucket}:[/]{count}")
        gap_str = " ".join(gap_parts) if gap_parts else "[dim]—[/]"

        tbl.add_row(
            n.anchor,
            str(n.chain_length),
            f"[{UNCERTAIN_COLOR}]{n.beta0:.1f}[/]",
            f"[{'magenta' if n.beta1 > 1 else 'dim'}]{n.beta1:.1f}[/]",
            f"[{pc}]{n.phase}[/]",
            f"[{ACCENT}]{n.velocity}[/]/mo" if n.velocity > 0 else "[dim]—[/]",
            gap_str,
        )
    console.print(tbl)
    console.print()

    # Phase distribution
    from collections import Counter
    phases = Counter(n.phase for n in narrative)
    parts = [f"[{PHASE_COLORS.get(p, 'dim')}]{p}: {c}[/]" for p, c in phases.most_common()]
    console.print(f"  [dim]Phase distribution:[/]  {' · '.join(parts)}")
    console.print()


# ═══════════════════════════════════════════════════════════════════════
# STEP 12: CONTAGION + KAN EXTENSION
# ═══════════════════════════════════════════════════════════════════════

def render_contagion_kan(sa_results, cog, all_infons):
    console.print()
    console.rule(f"[bold white]STEP 12[/]  [{ACCENT}]CONTAGION + KAN EXTENSION[/]", style=ACCENT)
    console.print("  [dim]Risk propagation fan-out and market-transfer estimates[/]\n")

    contagion = sa_results["contagion"]

    # ── Contagion ──
    flash_text("CONTAGION ANALYSIS", style="bold white on dark_red")
    console.print()

    console.print(Panel(
        f"[dim]When an anchor emits bearish signals (polarity=0), which[/]\n"
        f"[dim]competitors benefit via shared predicates?[/]\n\n"
        f"  [bold white]Actor[/] ──[{REFUTE_COLOR}]BEARISH[/]──▶ [bold white]Hit Anchor[/] "
        f"──[dim]shared pred[/]──▶ [{SUPPORT_COLOR}]Competitor[/]",
        title="[bold]4-Hop Contagion Path[/]",
        border_style=ACCENT, box=box.ROUNDED, width=62,
    ))
    console.print()

    if contagion:
        tbl = Table(
            title="[bold]Contagion Exposure[/]",
            border_style=ACCENT, box=box.ROUNDED,
            show_header=True, header_style="bold",
        )
        tbl.add_column("Anchor", width=20, style="bold white")
        tbl.add_column("Bearish", width=9, justify="right", style=REFUTE_COLOR)
        tbl.add_column("Fan-out", width=9, justify="right")
        tbl.add_column("Score", width=10, justify="right")
        tbl.add_column("Beneficiaries", width=30)

        for c in contagion[:8]:
            bens = ", ".join(c.beneficiaries[:4])
            if len(c.beneficiaries) > 4:
                bens += f" +{len(c.beneficiaries) - 4}"
            tbl.add_row(
                c.anchor,
                str(c.bearish_signals),
                str(c.fan_out_degree),
                f"[{REFUTE_COLOR}]{c.contagion_score:.2f}[/]",
                f"[{SUPPORT_COLOR}]{bens}[/]" if bens else "[dim]none[/]",
            )
        console.print(tbl)
    else:
        console.print(f"  [dim]No contagion detected (no bearish signals with shared predicates)[/]")
    console.print()

    # ── Kan Extension ──
    flash_text("KAN EXTENSION — MARKET TRANSFER", style="bold white on dark_blue")
    console.print()

    console.print(Panel(
        f"[bold]Left Kan[/]  [dim](optimistic)[/]  — project source signals into target gaps\n"
        f"[bold]Right Kan[/] [dim](conservative)[/] — assume zero where no evidence transfers\n\n"
        f"[dim]Gap between left and right = your[/] [bold]uncertainty budget[/]\n"
        f"[dim]for entering a new phase / market / domain.[/]",
        title="[bold]Kan Extension: Best Completion of Partial Data[/]",
        border_style=ACCENT, box=box.ROUNDED, width=62,
    ))
    console.print()

    # Split infons by temporal phase
    cutoff = DOMAIN["kan_source_cutoff"]
    kan = KanExtension()
    result = kan.extend(
        all_infons, cog.schema,
        source_filter=lambda i: (i.timestamp or "") <= cutoff,
        target_filter=lambda i: (i.timestamp or "") > cutoff,
        source_label=DOMAIN["kan_source_label"],
        target_label=DOMAIN["kan_target_label"],
    )

    tbl = Table(
        title=f"[bold]{result.source_context}  →  {result.target_context}[/]",
        border_style=ACCENT, box=box.ROUNDED,
        show_header=True, header_style="bold",
    )
    tbl.add_column("", width=22, style="dim", justify="right")
    tbl.add_column("Value", width=12, justify="center")

    tbl.add_row("Shared anchors", f"[bold]{len(result.shared_anchors)}[/]")
    tbl.add_row("Source-only", f"[{UNCERTAIN_COLOR}]{len(result.source_only)}[/]")
    tbl.add_row("Target-only", f"[{SUPPORT_COLOR}]{len(result.target_only)}[/]")

    console.print(tbl)
    console.print()

    # Top uncertainty items
    top_unc = list(result.uncertainty_budget.items())[:8]
    if top_unc:
        console.print(f"  [bold {ACCENT}]Uncertainty Budget[/]  [dim](left − right Kan, top gaps)[/]")
        max_unc = max(abs(v) for _, v in top_unc) if top_unc else 1.0
        for anchor, gap in top_unc:
            if gap > 0.001:
                animate_bar(anchor, gap, max_val=max(max_unc, 0.01), color=UNCERTAIN_COLOR)
    console.print()


# ═══════════════════════════════════════════════════════════════════════
# STEP 13: DRIVER TREE
# ═══════════════════════════════════════════════════════════════════════

def _render_driver_node(tree_widget, node, depth=0):
    """Recursively add DriverNode to a Rich Tree widget."""
    val_color = (SUPPORT_COLOR if node.value > 0.1
                 else REFUTE_COLOR if node.value < -0.1
                 else "dim")
    engine_tag = f"  [dim]\\[{node.source_engine}][/]" if node.source_engine else ""
    label = Text.from_markup(
        f"[bold white]{node.name}[/]  [{val_color}]{node.value:+.3f}[/] "
        f"[dim]{node.unit}[/]{engine_tag}"
    )
    branch = tree_widget.add(label)
    for child in node.children:
        _render_driver_node(branch, child, depth + 1)


def render_driver_tree(sa_results, cog, all_infons):
    console.print()
    console.rule(f"[bold white]STEP 13[/]  [{ACCENT}]DRIVER TREE[/]", style=ACCENT)
    console.print("  [dim]McKinsey-style decomposition — every leaf computed from graph topology[/]\n")

    console.print(Panel(
        f"[bold white]Portfolio Value[/]\n"
        f"├── [bold]Revenue[/]      [dim]= demand × conversion × pricing power[/]\n"
        f"│   ├── Demand       [dim]← signal count (organic vs paid proxy)[/]\n"
        f"│   ├── Conversion   [dim]← Kano must-be coverage %[/]\n"
        f"│   └── Pricing      [dim]← attractive feature count × exclusivity[/]\n"
        f"├── [bold]Cost[/]         [dim]= engineering waste + marketing burden + drag[/]\n"
        f"│   ├── Engineering   [dim]← silent features % (Feature Gap Functor)[/]\n"
        f"│   ├── Marketing     [dim]← H¹ × polarization direction[/]\n"
        f"│   └── Portfolio     [dim]← ghost count × carrying cost proxy[/]\n"
        f"└── [bold]Risk[/]         [dim]= contagion + concentration + narrative decay[/]\n"
        f"    ├── Contagion     [dim]← bearish fan-out degree[/]\n"
        f"    ├── Concentration [dim]← inverse subject diversity[/]\n"
        f"    └── Narrative     [dim]← decaying phase ratio[/]",
        title="[bold]Driver Tree Structure[/]  [dim]— no financial data needed[/]",
        border_style=ACCENT, box=box.ROUNDED, width=70,
    ))
    console.print()

    flash_text("COMPUTING DRIVER TREE", style="bold white on dark_green")
    console.print()

    sa = StructuralAnalyzer(cog.schema)
    driver = sa.driver_tree(all_infons, sa_results)

    # Render as Rich Tree
    root_color = SUPPORT_COLOR if driver.value > 0 else REFUTE_COLOR
    root_tree = Tree(
        Text.from_markup(
            f"[bold white]{driver.name}[/]  [{root_color} bold]{driver.value:+.4f}[/]"
        ),
        guide_style=ACCENT,
    )
    for child in driver.children:
        _render_driver_node(root_tree, child)

    console.print(Panel(
        root_tree,
        title="[bold]Computed Driver Tree[/]  [dim]— all values from structural engines[/]",
        border_style=ACCENT, box=box.ROUNDED, width=78,
    ))
    console.print()

    # Animated summary bars
    for child in driver.children:
        color = SUPPORT_COLOR if child.name == "Revenue" else REFUTE_COLOR if child.name == "Cost" else UNCERTAIN_COLOR
        animate_bar(child.name, abs(child.value), max_val=max(abs(c.value) for c in driver.children) or 1, color=color)
    console.print()

    # Unit economics callout
    console.print(Panel(
        f"[dim]Every node above is computable from graph topology alone:[/]\n\n"
        f"  [{REFUTE_COLOR}]Ghost anchor[/]        →  CAC → ∞ [dim](no organic discovery)[/]\n"
        f"  [{REFUTE_COLOR}]Silent feature[/]       →  R&D spend with zero market return\n"
        f"  [{SUPPORT_COLOR}]High β₁[/]             →  amortize over 5+ years [dim](structural permanence)[/]\n"
        f"  [{UNCERTAIN_COLOR}]Short β₁[/]            →  expense it, don't capitalize [dim](fad)[/]\n"
        f"  [{REFUTE_COLOR}]H¹ (bearish-skewed)[/] →  reputation repair cost\n"
        f"  [{SUPPORT_COLOR}]H¹ (strategic)[/]      →  earned media offsets paid [dim](low CAC)[/]\n"
        f"  [{REFUTE_COLOR}]Contagion fan-out[/]    →  competitor's CAC drops free\n\n"
        f"  [dim italic]Plug in actual costs per unit to get dollar values.[/]\n"
        f"  [dim italic]The shape and relative weights come from the topology.[/]",
        title="[bold]Unit Economics Assumptions from Topology[/]",
        border_style="yellow", box=box.ROUNDED, width=70,
    ))
    console.print()


# ═══════════════════════════════════════════════════════════════════════
# RECAP
# ═══════════════════════════════════════════════════════════════════════

def render_recap():
    recap = Table.grid(padding=(0, 2))
    recap.add_column(width=3, style="bold white", justify="right")
    recap.add_column(width=24, style=f"bold {ACCENT}")
    recap.add_column(style="dim", width=46)
    recap.add_row("1", "CHANGE OF BASIS", "SPLADE → AnchorProjector → your typed concepts")
    recap.add_row("", "", "No training, no embeddings — just a JSON schema.")
    recap.add_row("", "", "")
    recap.add_row("2", "EXTRACTIVE INFONS", "Every triple grounded to character spans")
    recap.add_row("", "", "with support type. Always trace back to source.")
    recap.add_row("", "", "")
    recap.add_row("3", "TEMPORAL GRAPH", "NEXT edges link facts per anchor across time.")
    recap.add_row("", "", "The graph knows sequence: first, second, third.")
    recap.add_row("", "", "")
    recap.add_row("4", "HIERARCHY EXPANSION", "Parent queries expand to descendants at")
    recap.add_row("", "", "query time. Broad queries find leaf evidence.")
    recap.add_row("", "", "")
    recap.add_row("5", "DEMPSTER-SHAFER", "Conflicting evidence fused, not averaged.")
    recap.add_row("", "", "Four mass functions per infon, Dempster's rule.")
    recap.add_row("", "", "")
    recap.add_row("6", "GRAPH MCTS", "AlphaGo-style tree search discovers multi-hop")
    recap.add_row("", "", "chains flat retrieval misses.")
    recap.add_row("", "", "")
    recap.add_row("7", "CONTRARY VIEW", "Invert the evidential lens at query time.")
    recap.add_row("", "", "Same data, opposite thesis — red-team any claim.")
    recap.add_row("", "", "")
    recap.add_row("8", "KANO + CONJOINT", "Classify anchors: must-be / linear / delighter.")
    recap.add_row("", "", "Conjoint part-worth utility from graph signals.")
    recap.add_row("", "", "")
    recap.add_row("9", "FEATURE GAP FUNCTOR", "F: Schema → Discourse. Silent value (defined,")
    recap.add_row("", "", "never discussed) vs hallucinated (discussed, never")
    recap.add_row("", "", "defined). Measures ontology-corpus alignment.")
    recap.add_row("", "", "")
    recap.add_row("10", "GHOSTS + POLARIZATION", "Ghost anchors: zero signals, CAC → ∞.")
    recap.add_row("", "", "H¹ polarization: irreducible contradiction.")
    recap.add_row("", "", "")
    recap.add_row("11", "NARRATIVE LIFECYCLE", "β₀ (fragmentation) and β₁ (loops) from")
    recap.add_row("", "", "persistent homology on NEXT chains. Phase")
    recap.add_row("", "", "detection: emerging → mature → dead.")
    recap.add_row("", "", "")
    recap.add_row("12", "CONTAGION + KAN", "Bearish fan-out via shared predicates.")
    recap.add_row("", "", "Kan extension: left (optimistic) / right")
    recap.add_row("", "", "(conservative) market-transfer estimates.")
    recap.add_row("", "", "")
    recap.add_row("13", "DRIVER TREE", "McKinsey Revenue / Cost / Risk decomposition.")
    recap.add_row("", "", "Every leaf from graph topology — no financials.")
    recap.add_row("", "", "Plug in unit costs for dollar values.")

    console.print(Panel(
        recap,
        title="[bold white]RECAP — Thirteen Innovations[/]",
        border_style=ACCENT,
        box=box.DOUBLE_EDGE,
        padding=(1, 2),
    ))


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    render_title()
    pause()

    config = CognitionConfig(
        schema_path=DOMAIN["schema_path"],
        db_path=":memory:",
        activation_threshold=0.15,
        min_confidence=0.02,
        top_k_per_role=5,
        default_top_k=50,
        consolidation_interval=0,
    )

    with Progress(
        SpinnerColumn("dots12", style=ACCENT),
        TextColumn(f"[{ACCENT}]Loading SPLADE-tiny (17MB bundled, no GPU required)...[/]"),
        console=console, transient=True,
    ) as progress:
        progress.add_task("", total=None)
        cog = Cognition(config)

    console.print(f"  [{SUPPORT_COLOR}]Ready.[/] Schema: [bold]{len(cog.schema.names)}[/] anchors | "
                  f"Device: [bold]{cog.encoder.device}[/]\n")

    docs = DOMAIN["documents"]

    # ── Step 1 ─────────────────────────────────────────────────────────
    render_change_of_basis(cog)
    pause()

    # ── Step 2 ─────────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn("dots12", style=ACCENT),
        TextColumn(f"[{ACCENT}]Ingesting {len(docs)} documents...[/]"),
        console=console, transient=True,
    ) as progress:
        progress.add_task("", total=None)
        t0 = time.time()
        n_infons = cog.ingest(docs)
        elapsed = time.time() - t0

    all_infons = cog.store.query_infons(limit=200)
    render_extraction(cog, all_infons)

    console.print(f"\n  [{SUPPORT_COLOR}]Extracted:[/] [bold]{n_infons}[/] infons from "
                  f"[bold]{len(docs)}[/] documents in [bold]{elapsed:.1f}s[/]")
    pause()

    # ── Step 3 ─────────────────────────────────────────────────────────
    render_temporal_graph(cog, all_infons)
    pause()

    # ── Step 4 ─────────────────────────────────────────────────────────
    render_hierarchy_expansion(cog)
    pause()

    # ── Step 5 ─────────────────────────────────────────────────────────
    render_dempster_shafer(cog, all_infons)
    pause()

    # ── Step 6 ─────────────────────────────────────────────────────────
    mcts_result, query = render_mcts(cog, all_infons)
    render_comparison(cog, mcts_result, query)
    pause()

    # ── Step 7 ─────────────────────────────────────────────────────────
    all_infons = cog.store.query_infons(limit=200)
    render_contrary_view(cog, all_infons)
    pause()

    # ── Structural Analysis ───────────────────────────────────────────
    console.print()
    with Progress(
        SpinnerColumn("dots12", style=ACCENT),
        TextColumn(f"[{ACCENT}]Running structural analysis (9 engines)...[/]"),
        console=console, transient=True,
    ) as progress:
        progress.add_task("", total=None)
        sa_results = cog.analyze(enrich=True)
        all_infons = cog.store.query_infons(limit=200)

    # ── Step 8 ─────────────────────────────────────────────────────────
    render_kano_conjoint(sa_results)
    pause()

    # ── Step 9 ─────────────────────────────────────────────────────────
    render_feature_gap(sa_results, cog)
    pause()

    # ── Step 10 ────────────────────────────────────────────────────────
    render_ghost_polarization(sa_results, cog)
    pause()

    # ── Step 11 ────────────────────────────────────────────────────────
    render_narrative(sa_results)
    pause()

    # ── Step 12 ────────────────────────────────────────────────────────
    render_contagion_kan(sa_results, cog, all_infons)
    pause()

    # ── Step 13 ────────────────────────────────────────────────────────
    render_driver_tree(sa_results, cog, all_infons)
    pause()

    # ── Recap ──────────────────────────────────────────────────────────
    render_recap()

    cog.close()
    console.print(f"\n  [dim]Demo complete.[/]\n")


if __name__ == "__main__":
    main()
