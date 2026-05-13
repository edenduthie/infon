#!/usr/bin/env python3
"""Cognition TUI — Midnight Commander style.

Two-panel knowledge graph navigator with F-key bar.
Left panel: ontology grid / sheaf. Right panel: infon results / timeline.
Bottom: command input. F-keys for actions.

.. deprecated::
    This TUI targets the pre-v2 retrieval-only API (`cog.query()`).
    It does not expose the calibrated reasoner, mass functions, θ
    ignorance, `cog.expand()`, `cog.refresh()`, or the AutoML helpers
    (`sweep`, `cross_val_score`, `ensemble_top_k`, `analyze_corpus`).

    For the current showcase use `demo_customer.py` (rich-terminal
    walkthrough of the full v3 stack) or `ask.py` (CLI with reasoner
    + retrieval + diagnostics).

    This file is kept in tree as a research artifact. It still runs
    against the current runtime but is not being updated to match
    the v3 feature set.

Usage:
    python cognition_tui.py
    python cognition_tui.py --schema schema.json --db data/my.db
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Static, Input, Footer, Label
from rich.text import Text

from cognition import (
    Cognition, CognitionConfig, AnchorSchema, Encoder,
    extract_infons, split_sentences,
    SheafCoherence,
)
from cognition.consolidate import build_next_edges


# ── Demo data ───────────────────────────────────────────────────────────

DEMO_SCHEMA = {
    "us":            {"type": "actor",    "tokens": ["us", "united states", "america", "washington"]},
    "china":         {"type": "actor",    "tokens": ["china", "chinese", "beijing"]},
    "russia":        {"type": "actor",    "tokens": ["russia", "russian", "moscow", "kremlin"]},
    "iran":          {"type": "actor",    "tokens": ["iran", "iranian", "tehran"]},
    "israel":        {"type": "actor",    "tokens": ["israel", "israeli", "jerusalem"]},
    "nato":          {"type": "actor",    "tokens": ["nato", "alliance", "atlantic"]},
    "eu":            {"type": "actor",    "tokens": ["eu", "european union", "brussels"]},
    "un":            {"type": "actor",    "tokens": ["un", "united nations", "security council"]},
    "india":         {"type": "actor",    "tokens": ["india", "indian", "delhi"]},
    "japan":         {"type": "actor",    "tokens": ["japan", "japanese", "tokyo"]},
    "palestine":     {"type": "actor",    "tokens": ["palestine", "palestinian", "gaza", "hamas"]},
    "african_union": {"type": "actor",    "tokens": ["african union", "au", "african"]},
    "sanction":      {"type": "relation", "tokens": ["sanction", "sanctions", "embargo", "restrict"]},
    "negotiate":     {"type": "relation", "tokens": ["negotiate", "negotiation", "talks", "diplomacy", "diplomatic"]},
    "deploy":        {"type": "relation", "tokens": ["deploy", "deployment", "send", "station", "troops"]},
    "attack":        {"type": "relation", "tokens": ["attack", "strike", "bomb", "assault", "offensive"]},
    "trade":         {"type": "relation", "tokens": ["trade", "export", "import", "tariff", "commerce"]},
    "invest":        {"type": "relation", "tokens": ["invest", "investment", "fund", "finance"]},
    "condemn":       {"type": "relation", "tokens": ["condemn", "denounce", "criticize", "oppose"]},
    "cooperate":     {"type": "relation", "tokens": ["cooperate", "cooperation", "collaborate", "joint", "partnership"]},
    "nuclear":       {"type": "feature",  "tokens": ["nuclear", "uranium", "enrichment", "atomic"]},
    "military":      {"type": "feature",  "tokens": ["military", "army", "defense", "weapon"]},
    "technology":    {"type": "feature",  "tokens": ["technology", "cyber", "ai", "digital"]},
    "territory":     {"type": "feature",  "tokens": ["territory", "border", "land", "sovereignty"]},
    "humanitarian":  {"type": "feature",  "tokens": ["humanitarian", "aid", "refugee", "crisis"]},
    "maritime":      {"type": "feature",  "tokens": ["maritime", "naval", "sea", "strait", "shipping"]},
    "middle_east":   {"type": "market",   "tokens": ["middle east", "gulf", "levant"]},
    "east_asia":     {"type": "market",   "tokens": ["east asia", "pacific", "asia-pacific", "indo-pacific"]},
    "europe":        {"type": "market",   "tokens": ["europe", "european", "continent"]},
    "africa":        {"type": "market",   "tokens": ["africa", "sahel", "horn of africa", "sub-saharan"]},
}

DEMO_DOCS = [
    {"id": "geo-001", "timestamp": "2004-08-02", "text": "African Union deployed peacekeeping troops to the Darfur region amid escalating humanitarian crisis and military attacks on civilian populations."},
    {"id": "geo-002", "timestamp": "2006-12-23", "text": "The UN Security Council imposed sanctions on Iran over its nuclear enrichment program, demanding suspension of uranium processing activities."},
    {"id": "geo-003", "timestamp": "2008-03-26", "text": "NATO expanded its military presence in Afghanistan while the African Union struggled to maintain peacekeeping operations in Somalia."},
    {"id": "geo-004", "timestamp": "2010-06-09", "text": "China and Japan engaged in maritime territorial disputes in the East China Sea, with both nations deploying naval vessels near contested islands."},
    {"id": "geo-005", "timestamp": "2012-11-14", "text": "Israel launched military operations in Gaza while the international community condemned the escalating violence and humanitarian crisis."},
    {"id": "geo-006", "timestamp": "2014-03-18", "text": "Russia annexed Crimea, prompting the EU and US to impose sweeping economic sanctions and NATO to deploy additional forces to Eastern Europe."},
    {"id": "geo-007", "timestamp": "2015-07-05", "text": "Iran nuclear deal signed after years of diplomatic negotiations between Iran and world powers, lifting trade sanctions in exchange for nuclear restrictions."},
    {"id": "geo-008", "timestamp": "2016-07-12", "text": "China rejected the international tribunal ruling on South China Sea territorial claims, deploying military assets to artificial islands in the disputed maritime region."},
    {"id": "geo-009", "timestamp": "2017-09-03", "text": "India and Japan announced a joint investment in infrastructure development across Southeast Asia to counter Chinese economic influence in the region."},
    {"id": "geo-010", "timestamp": "2018-06-12", "text": "US-China trade war escalated with new tariffs on technology exports, affecting global supply chains and East Asian markets."},
    {"id": "geo-011", "timestamp": "2019-10-09", "text": "Turkey launched military operations in northern Syria, drawing condemnation from the EU and complicating NATO alliance dynamics."},
    {"id": "geo-012", "timestamp": "2020-09-10", "text": "Israel signed the Abraham Accords, normalizing diplomatic relations with UAE and Bahrain in a historic Middle East peace agreement."},
    {"id": "geo-013", "timestamp": "2021-08-15", "text": "NATO forces withdrew from Afghanistan as the Taliban seized control, creating a humanitarian crisis and refugee emergency."},
    {"id": "geo-014", "timestamp": "2022-02-24", "text": "Russia invaded Ukraine, triggering the largest military conflict in Europe since WWII. NATO deployed rapid response forces and the EU imposed unprecedented economic sanctions."},
    {"id": "geo-015", "timestamp": "2022-08-02", "text": "US Speaker Pelosi visited Taiwan, escalating tensions between China and the US. China deployed military forces around Taiwan in response."},
    {"id": "geo-016", "timestamp": "2023-01-15", "text": "Japan announced a historic increase in military spending and cooperation with NATO, signaling a shift in East Asian security dynamics."},
    {"id": "geo-017", "timestamp": "2023-04-20", "text": "India emerged as a key diplomatic mediator, negotiating with both Russia and the West while expanding trade partnerships across Africa and the Middle East."},
    {"id": "geo-018", "timestamp": "2023-10-07", "text": "Hamas launched a major attack on Israel from Gaza, triggering Israeli military operations and international calls for humanitarian corridors."},
    {"id": "geo-019", "timestamp": "2024-02-14", "text": "EU imposed new sanctions on Iranian drone technology transfers to Russia, linking Middle East and European security concerns."},
    {"id": "geo-020", "timestamp": "2024-06-10", "text": "China and Russia conducted joint naval exercises in the Pacific, while Japan and the US strengthened their maritime defense cooperation."},
    {"id": "geo-021", "timestamp": "2024-09-01", "text": "African Union launched a continental technology investment initiative, partnering with India and Japan on digital infrastructure across Sub-Saharan Africa."},
    {"id": "geo-022", "timestamp": "2025-01-20", "text": "UN-mediated negotiations on the Ukraine conflict stalled as Russia rejected territorial concessions and NATO expanded its eastern border presence."},
    {"id": "geo-023", "timestamp": "2025-06-15", "text": "Iran announced a new nuclear cooperation agreement with China, drawing condemnation from Israel and the US while complicating Middle East diplomacy."},
    {"id": "geo-024", "timestamp": "2026-01-10", "text": "The EU and African Union signed a comprehensive trade and humanitarian partnership, investing in technology and infrastructure development."},
]


# ── Style ───────────────────────────────────────────────────────────────

C = "#00bbcc"       # cyan (MC blue equivalent)
CD = "#007788"      # dim cyan
CB = "#003344"      # border
W = "#cccccc"       # text
WB = "#ffffff"      # bright text
BG = "#0a1014"      # background
PB = "#0d1520"      # panel bg
HI = "#00ffcc"      # highlight
MAG = "#cc44ff"     # relation/predicate
GRN = "#44ff88"     # feature
AMB = "#ffaa22"     # market
RED = "#ff4466"     # negative valence
SEL = "#1a3040"     # selected row bg

TC = {"actor": C, "relation": MAG, "feature": GRN, "market": AMB}
TG = {"actor": "\u25c6", "relation": "\u2192", "feature": "\u2605", "market": "\u25cf"}


# ═══════════════════════════════════════════════════════════════════════
# WIDGETS
# ═══════════════════════════════════════════════════════════════════════

class PanelTitle(Static):
    """MC-style panel title bar."""

    def __init__(self, title: str, **kw):
        super().__init__(**kw)
        self._title = title

    def render(self):
        t = Text()
        t.append("\u2500\u2500\u2524 ", style=CD)
        t.append(self._title, style=f"bold {C}")
        t.append(" \u251c", style=CD)
        # fill rest with line chars
        t.append("\u2500" * 60, style=CD)
        return t


class CmdInput(Input):
    """Bottom command line with history."""

    history: list[str] = []
    history_idx: int = -1
    _stashed: str = ""

    def on_key(self, event) -> None:
        if event.key == "up" and self.history:
            if self.history_idx == -1:
                self._stashed = self.value
                self.history_idx = len(self.history) - 1
            elif self.history_idx > 0:
                self.history_idx -= 1
            self.value = self.history[self.history_idx]
            event.prevent_default()
        elif event.key == "down":
            if self.history_idx >= 0:
                self.history_idx += 1
                if self.history_idx >= len(self.history):
                    self.history_idx = -1
                    self.value = self._stashed
                else:
                    self.value = self.history[self.history_idx]
                event.prevent_default()


class LeftPanel(Static):
    """Left panel: ontology grid + sheaf."""

    content: reactive[object] = reactive(None, always_update=True)

    def render(self):
        if isinstance(self.content, Text):
            return self.content
        return Text(" Loading engine...", style=CD)


class RightPanel(Static):
    """Right panel: infon results / timeline / constraints."""

    content: reactive[object] = reactive(None, always_update=True)

    def render(self):
        if isinstance(self.content, Text):
            return self.content
        return Text(" Type a query and press Enter", style=CD)


class FKeyBar(Static):
    """MC-style F-key function bar at the very bottom."""

    persona: reactive[str] = reactive("analyst", always_update=True)

    def render(self):
        t = Text()
        keys = [
            ("1", "Help"),
            ("2", "Sheaf"),
            ("3", "Timeline"),
            ("4", "Constr"),
            ("5", "Anchors"),
            ("6", self.persona.title()),
            ("7", "Stats"),
            ("8", "Clear"),
            ("9", "Ingest"),
            ("10", "Quit"),
        ]
        for num, label in keys:
            t.append(f" {num}", style="bold #000000 on #00bbcc")
            t.append(f"{label:<8s}", style="#cccccc on #003344")
        return t


# ═══════════════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════════════

class CognitionApp(App):
    """Midnight Commander style knowledge graph navigator."""

    TITLE = "COGNITION"

    CSS = """
    Screen {
        background: #0a1014;
    }

    #title-bar {
        dock: top;
        height: 1;
        background: #003344;
    }

    #panels {
        height: 1fr;
    }

    #left-box {
        width: 1fr;
        border: solid #007788;
        background: #0d1520;
    }

    #right-box {
        width: 1fr;
        border: solid #007788;
        background: #0d1520;
    }

    LeftPanel {
        padding: 0 1;
    }

    RightPanel {
        padding: 0 1;
    }

    #cmd-row {
        dock: bottom;
        height: auto;
        max-height: 4;
    }

    #fkey-bar {
        dock: bottom;
        height: 1;
        background: #003344;
    }

    #prompt-label {
        width: 2;
        height: 1;
        color: #00bbcc;
        background: #0a1014;
        text-style: bold;
        padding: 0;
    }

    CmdInput {
        background: #0a1014;
        color: #00ffcc;
        border: none;
        height: 1;
        width: 1fr;
    }

    CmdInput:focus {
        border: none;
    }

    #status-line {
        dock: bottom;
        height: 1;
        background: #0d1520;
    }
    """

    BINDINGS = [
        Binding("f1", "show_help", "Help", show=False),
        Binding("f2", "show_sheaf", "Sheaf", show=False),
        Binding("f3", "show_timeline", "Timeline", show=False),
        Binding("f4", "show_constraints", "Constraints", show=False),
        Binding("f5", "show_anchors", "Anchors", show=False),
        Binding("f6", "cycle_persona", "Persona", show=False),
        Binding("f7", "show_stats", "Stats", show=False),
        Binding("f8", "clear", "Clear", show=False),
        Binding("f9", "ingest_demo", "Ingest", show=False),
        Binding("f10", "quit", "Quit", show=False),
        Binding("escape", "focus_cmd", priority=True, show=False),
        Binding("tab", "toggle_panel", show=False),
    ]

    # State
    schema: AnchorSchema | None = None
    cog: Cognition | None = None
    sheaf: SheafCoherence | None = None
    personas = ["analyst", "investor", "engineer", "executive", "regulator"]
    persona_idx: int = 0
    current_persona: str = "analyst"
    left_mode: str = "anchors"     # anchors, sheaf, help
    right_mode: str = "results"    # results, timeline, constraints, stats
    last_result = None

    def __init__(self, schema_path: str | None = None, db_path: str = "data/cognition_mc.db"):
        super().__init__()
        self.schema_path = schema_path
        self.db_path = db_path

    def compose(self) -> ComposeResult:
        yield Static(id="title-bar")
        with Horizontal(id="panels"):
            with VerticalScroll(id="left-box"):
                yield LeftPanel()
            with VerticalScroll(id="right-box"):
                yield RightPanel()
        yield Static(id="status-line")
        with Horizontal(id="cmd-row"):
            yield Label("> ", id="prompt-label")
            yield CmdInput(placeholder="query / command", id="cmd")
        yield FKeyBar(id="fkey-bar")

    def on_mount(self) -> None:
        self._render_title()
        self._render_status("Loading engine...")
        self.query_one("#cmd", CmdInput).focus()
        self.load_engine()

    def _render_title(self) -> None:
        bar = self.query_one("#title-bar", Static)
        t = Text()
        t.append(" COGNITION", style=f"bold {HI}")
        t.append("  Infon Engine  ", style=W)
        t.append("\u2502 ", style=CD)
        t.append(f"{self.current_persona.upper()}", style=f"bold {MAG}")
        t.append("  \u2502 ", style=CD)
        t.append("SPLADE-tiny 4.4M  30,522 vocab", style=CD)
        bar.update(t)

    def _render_status(self, msg: str) -> None:
        bar = self.query_one("#status-line", Static)
        t = Text()
        t.append(f" {msg}", style=CD)
        bar.update(t)

    def _render_status_counts(self) -> None:
        if not self.cog:
            return
        stats = self.cog.stats()
        bar = self.query_one("#status-line", Static)
        t = Text()
        t.append(f" {stats['infon_count']} infons", style=f"bold {GRN}")
        t.append("  ", style=CD)
        t.append(f"{stats['constraint_count']} constraints", style=f"bold {AMB}")
        t.append("  ", style=CD)
        n_anchors = len(self.schema.names) if self.schema else 0
        t.append(f"{n_anchors} anchors", style=f"bold {C}")
        if self.sheaf:
            t.append("  ", style=CD)
            t.append(f"Fiedler={self.sheaf.fiedler_value:.3f}", style=f"bold {AMB}")
        bar.update(t)

    # ── Engine init ─────────────────────────────────────────────────────

    @work(thread=True, exclusive=True, group="init")
    def load_engine(self) -> None:
        if self.schema_path and Path(self.schema_path).exists():
            schema_file = self.schema_path
        else:
            schema_file = "data/mc_schema.json"
            Path("data").mkdir(exist_ok=True)
            Path(schema_file).write_text(json.dumps(DEMO_SCHEMA, indent=2))

        config = CognitionConfig(schema_path=schema_file, db_path=self.db_path)
        self.cog = Cognition(config)
        self.schema = self.cog.schema

        if self.cog.store.count_infons() == 0:
            self.app.call_from_thread(self._render_status, "Ingesting demo corpus...")
            self.cog.ingest(DEMO_DOCS, consolidate_now=True)

        # Build sheaf
        sentences = []
        for doc in DEMO_DOCS:
            sentences.extend(split_sentences(doc["text"]))
        act_matrix = self.cog.encoder.encode(sentences)
        self.sheaf = SheafCoherence(self.schema.names)
        self.sheaf.observe(act_matrix, threshold=0.3)
        self.sheaf.fit()

        all_infons = self.cog.store.query_infons(limit=50000)
        for inf in all_infons:
            inf.coherence = self.sheaf.score_infon(inf)
        self.cog.store.put_infons(all_infons)

        self.app.call_from_thread(self._render_status_counts)
        self.app.call_from_thread(self._render_title)
        self.app.call_from_thread(self._show_anchors)
        self.app.call_from_thread(self._show_help_right)

    # ── Command input ───────────────────────────────────────────────────

    @on(Input.Submitted, "#cmd")
    def on_cmd(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        if not raw:
            return

        cmd_input = self.query_one("#cmd", CmdInput)
        if raw and (not cmd_input.history or cmd_input.history[-1] != raw):
            cmd_input.history.append(raw)
        cmd_input.history_idx = -1

        # Parse commands
        low = raw.lower()
        if low in ("quit", "exit", "q"):
            self.exit()
        elif low in ("help", "?"):
            self.action_show_help()
        elif low in ("stats", "stat"):
            self.action_show_stats()
        elif low in ("sheaf", "coherence"):
            self.action_show_sheaf()
        elif low in ("anchors", "grid", "ontology"):
            self.action_show_anchors()
        elif low in ("timeline", "time", "temporal"):
            self._show_timeline()
        elif low in ("constraints", "constr"):
            self._show_constraints()
        elif low in ("clear", "cls"):
            self.action_clear()
        elif low.startswith("persona "):
            p = low.split(None, 1)[1]
            if p in self.personas:
                self.current_persona = p
                self.persona_idx = self.personas.index(p)
                self._render_title()
                self.query_one("#fkey-bar", FKeyBar).persona = p
                self._render_status(f"Persona: {p}")
            else:
                self._render_status(f"Unknown persona: {p}. Options: {', '.join(self.personas)}")
        elif low.startswith("ingest "):
            path = raw.split(None, 1)[1]
            self._do_ingest(path)
        else:
            # It's a query
            self.run_query(raw)

        cmd_input.value = ""

    # ── Query ───────────────────────────────────────────────────────────

    @work(thread=True, exclusive=True, group="query")
    def run_query(self, text: str) -> None:
        if not self.cog:
            self.app.call_from_thread(self._render_status, "Engine not loaded yet")
            return

        self.app.call_from_thread(self._render_status, f"Querying: {text}")

        t0 = time.time()
        result = self.cog.query(
            text, persona=self.current_persona, top_k=50, include_chains=True,
        )
        dt = (time.time() - t0) * 1000
        self.last_result = result

        # Update left: show activated anchors
        activations = self.cog.encoder.encode_single(text)
        self.app.call_from_thread(self._show_anchors, activations)

        # Update right: show infon results
        self.app.call_from_thread(self._show_results, result, dt)
        self.app.call_from_thread(
            self._render_status,
            f"{len(result.infons)} infons  {len(result.constraints)} constraints  "
            f"{len(result.edges)} edges  {dt:.0f}ms"
        )

    def _show_results(self, result, dt: float) -> None:
        rp = self.query_one(RightPanel)
        t = Text()
        t.append(f" Query: ", style=CD)
        t.append(f"{result.query}\n", style=f"bold {WB}")
        t.append(f" Persona: ", style=CD)
        t.append(f"{result.persona}", style=f"bold {MAG}")
        t.append(f"  {len(result.infons)} results  {dt:.0f}ms\n", style=CD)
        t.append(f" \u2500" * 56 + "\n", style=CB)

        # Top anchor activations
        sorted_acts = sorted(result.anchors_activated.items(), key=lambda x: -x[1])[:6]
        if sorted_acts:
            t.append(f"\n Activated: ", style=CD)
            for name, score in sorted_acts:
                atype = self.schema.types.get(name, "") if self.schema else ""
                color = TC.get(atype, C)
                t.append(f"{name}", style=color)
                t.append(f"={score:.2f} ", style=CD)
            t.append("\n")

        t.append("\n")

        for i, inf in enumerate(result.infons[:25]):
            v = result.valence.get(inf.infon_id, 0)
            if v > 0.1:
                arrow, vc = "\u25b2", GRN
            elif v < -0.1:
                arrow, vc = "\u25bc", RED
            else:
                arrow, vc = "\u2500", CD

            sc = TC.get(self.schema.types.get(inf.subject, ""), C) if self.schema else C
            pc = TC.get(self.schema.types.get(inf.predicate, ""), MAG) if self.schema else MAG
            oc = TC.get(self.schema.types.get(inf.object, ""), GRN) if self.schema else GRN

            t.append(f" {arrow} ", style=vc)
            t.append(inf.predicate, style=f"bold {pc}")
            t.append(", ", style=CD)
            t.append(inf.subject, style=f"bold {sc}")
            t.append(", ", style=CD)
            t.append(inf.object, style=f"bold {oc}")

            coh_bar = "\u2588" * max(1, int(inf.coherence * 6))
            t.append(f"  {inf.confidence:.2f}", style=W)
            t.append(f" {coh_bar}", style=AMB)
            t.append(f" {v:+.2f}\n", style=vc)

            if inf.sentence:
                sent = inf.sentence[:68] + "..." if len(inf.sentence) > 68 else inf.sentence
                t.append(f"   ", style=CD)
                if inf.timestamp:
                    t.append(f"{inf.timestamp[:10]} ", style=CD)
                t.append(f"{sent}\n", style=f"dim {W}")

        if len(result.infons) > 25:
            t.append(f"\n ... +{len(result.infons) - 25} more (F3=timeline F4=constraints)\n", style=CD)

        self.right_mode = "results"
        rp.content = t

    # ── Left panel views ────────────────────────────────────────────────

    def _show_anchors(self, activations: dict | None = None) -> None:
        lp = self.query_one(LeftPanel)
        t = Text()
        acts = activations or {}

        t.append(" ONTOLOGY GRID\n", style=f"bold {C}")
        t.append(" \u2500" * 34 + "\n", style=CB)

        if not self.schema:
            t.append(" Loading...\n", style=CD)
            lp.content = t
            return

        by_type = defaultdict(list)
        for name in self.schema.names:
            by_type[self.schema.types.get(name, "feature")].append(name)

        for atype in ["actor", "relation", "feature", "market"]:
            color = TC.get(atype, W)
            glyph = TG.get(atype, "\u25cf")
            t.append(f"\n {glyph} {atype.upper()}\n", style=f"bold {color}")

            for name in sorted(by_type.get(atype, [])):
                act = acts.get(name, 0.0)
                if act > 0.5:
                    bar = "\u2588" * min(8, int(act * 4))
                    t.append(f"  {name:16s}", style=f"bold {color}")
                    t.append(f" {bar} {act:.2f}\n", style=color)
                elif act > 0.1:
                    bar = "\u2591" * max(1, int(act * 4))
                    t.append(f"  {name:16s}", style=color)
                    t.append(f" {bar} {act:.2f}\n", style=f"dim {color}")
                else:
                    t.append(f"  {name:16s}", style=f"dim {CD}")
                    t.append(f" \u00b7\n", style=CB)

        self.left_mode = "anchors"
        lp.content = t

    def _show_sheaf_left(self) -> None:
        lp = self.query_one(LeftPanel)
        t = Text()
        t.append(" SHEAF COHERENCE\n", style=f"bold {C}")
        t.append(" \u2500" * 34 + "\n\n", style=CB)

        if not self.sheaf:
            t.append(" Not computed yet.\n", style=CD)
            lp.content = t
            return

        fv = self.sheaf.fiedler_value
        t.append(" Fiedler value: ", style=CD)
        t.append(f"{fv:.4f}\n", style=f"bold {AMB}")

        if fv > 0.5:
            t.append(" Graph: ", style=CD)
            t.append("CONNECTED\n", style=f"bold {GRN}")
        elif fv > 0:
            t.append(" Graph: ", style=CD)
            t.append("WEAKLY CONNECTED\n", style=f"bold {AMB}")
        else:
            t.append(" Graph: ", style=CD)
            t.append("DISCONNECTED\n", style=f"bold {RED}")

        comps = self.sheaf.component_structure()
        t.append(f" Components: {len(comps)}\n\n", style=CD)

        # Hub anchors
        centrality = self.sheaf.anchor_centrality()
        top = sorted(centrality.items(), key=lambda x: -x[1])[:12]
        t.append(" HUB ANCHORS\n", style=f"bold {C}")
        t.append(" \u2500" * 34 + "\n", style=CB)
        for name, score in top:
            atype = self.schema.types.get(name, "?") if self.schema else "?"
            color = TC.get(atype, W)
            glyph = TG.get(atype, "\u25cf")
            bar = "\u2588" * max(1, int(score * 10))
            t.append(f" {glyph} ", style=color)
            t.append(f"{name:14s}", style=f"bold {color}")
            t.append(f" {bar} ", style=color)
            t.append(f"{score:.2f}\n", style=CD)

        # Eigenvalue spectrum
        if self.sheaf.laplacian is not None:
            eigenvalues = np.linalg.eigvalsh(self.sheaf.laplacian)
            eigenvalues.sort()
            t.append(f"\n LAPLACIAN SPECTRUM\n", style=f"bold {C}")
            t.append(" \u2500" * 34 + "\n", style=CB)
            for i, ev in enumerate(eigenvalues[:10]):
                label = "\u03bb\u2080" if i == 0 else "\u03bb\u2081*" if i == 1 else f"\u03bb{i}"
                bar = "\u2588" * max(1, int(ev * 3))
                t.append(f" {label:5s}", style=CD)
                t.append(f" {bar} ", style=C)
                t.append(f"{ev:.3f}\n", style=CD)

        self.left_mode = "sheaf"
        lp.content = t

    def _show_help_left(self) -> None:
        lp = self.query_one(LeftPanel)
        t = Text()
        t.append(" HELP\n", style=f"bold {C}")
        t.append(" \u2500" * 34 + "\n\n", style=CB)

        cmds = [
            ("Type a query", "natural language search"),
            ("help, ?", "this screen"),
            ("stats", "knowledge graph stats"),
            ("sheaf", "sheaf coherence (left)"),
            ("anchors", "ontology grid (left)"),
            ("timeline", "temporal chains (right)"),
            ("constraints", "aggregated claims (right)"),
            ("persona <name>", "set persona"),
            ("ingest <path>", "ingest JSON documents"),
            ("clear, cls", "clear panels"),
            ("quit, exit, q", "exit"),
        ]

        t.append(" COMMANDS\n", style=f"bold {C}")
        for cmd, desc in cmds:
            t.append(f"  {cmd:20s}", style=f"bold {HI}")
            t.append(f" {desc}\n", style=W)

        t.append(f"\n F-KEYS\n", style=f"bold {C}")
        fkeys = [
            ("F1", "Help"), ("F2", "Sheaf"), ("F3", "Timeline"),
            ("F4", "Constraints"), ("F5", "Anchors"), ("F6", "Persona"),
            ("F7", "Stats"), ("F8", "Clear"), ("F9", "Ingest demo"),
            ("F10", "Quit"), ("Tab", "Switch panels"),
            ("Up/Down", "Command history"), ("Esc", "Focus command line"),
        ]
        for key, desc in fkeys:
            t.append(f"  {key:10s}", style=f"bold {C}")
            t.append(f" {desc}\n", style=W)

        t.append(f"\n PERSONAS\n", style=f"bold {C}")
        for p in self.personas:
            if p == self.current_persona:
                t.append(f"  \u25b6 {p}\n", style=f"bold {MAG}")
            else:
                t.append(f"    {p}\n", style=W)

        self.left_mode = "help"
        lp.content = t

    # ── Right panel views ───────────────────────────────────────────────

    def _show_help_right(self) -> None:
        rp = self.query_one(RightPanel)
        t = Text()
        t.append(" COGNITION\n", style=f"bold {HI}")
        t.append(" \u2500" * 50 + "\n\n", style=CB)
        t.append(" Document \u2192 Infon \u2192 Temporal Knowledge Graph\n\n", style=W)
        t.append(" Type a natural language query below.\n", style=CD)
        t.append(" The engine encodes through SPLADE, projects\n", style=CD)
        t.append(" onto the ontology grid, and retrieves grounded\n", style=CD)
        t.append(" infons ranked by persona-relative valence.\n\n", style=CD)

        t.append(" Quick queries to try:\n\n", style=f"bold {C}")
        samples = [
            "What is Russia doing in Europe?",
            "How has the conflict between Russia and NATO evolved?",
            "What are China's maritime activities?",
            "Who is investing in Africa?",
            "What nuclear threats exist in the Middle East?",
        ]
        for s in samples:
            t.append(f"   {s}\n", style=HI)

        t.append(f"\n Press F1 for help, F6 to cycle persona.\n", style=CD)
        rp.content = t

    def _show_timeline(self) -> None:
        rp = self.query_one(RightPanel)
        t = Text()
        t.append(" TEMPORAL CHAINS\n", style=f"bold {C}")
        t.append(" \u2500" * 50 + "\n", style=CB)

        if not self.last_result or not self.last_result.timeline:
            t.append("\n No timeline data. Run a query first.\n", style=CD)
            rp.content = t
            return

        result = self.last_result
        t.append(f" Query: {result.query}\n", style=CD)
        t.append(f" {len(result.timeline)} events, {len(result.edges)} NEXT edges\n\n", style=CD)

        prev_year = None
        for inf in result.timeline[:40]:
            year = inf.timestamp[:4] if inf.timestamp else "????"
            if year != prev_year:
                t.append(f"\n {year}\n", style=f"bold {C}")
                prev_year = year

            v = result.valence.get(inf.infon_id, 0)
            if v > 0.1:
                arrow, vc = "\u25b2", GRN
            elif v < -0.1:
                arrow, vc = "\u25bc", RED
            else:
                arrow, vc = "\u2500", CD

            ts = inf.timestamp[5:10] if inf.timestamp else "??-??"
            pc = TC.get(self.schema.types.get(inf.predicate, ""), MAG) if self.schema else MAG
            sc = TC.get(self.schema.types.get(inf.subject, ""), C) if self.schema else C
            oc = TC.get(self.schema.types.get(inf.object, ""), GRN) if self.schema else GRN

            t.append(f"  {ts} ", style=CD)
            t.append(f"{arrow} ", style=vc)
            t.append(inf.predicate, style=pc)
            t.append(", ", style=CD)
            t.append(inf.subject, style=sc)
            t.append(", ", style=CD)
            t.append(inf.object, style=oc)
            coh_bar = "\u2588" * max(1, int(inf.coherence * 5))
            t.append(f"  {coh_bar}\n", style=AMB)

        if len(result.timeline) > 40:
            t.append(f"\n  ... +{len(result.timeline) - 40} more\n", style=CD)

        self.right_mode = "timeline"
        rp.content = t

    def _show_constraints(self) -> None:
        rp = self.query_one(RightPanel)
        t = Text()
        t.append(" CONSTRAINTS\n", style=f"bold {C}")
        t.append(" \u2500" * 50 + "\n", style=CB)

        if not self.last_result or not self.last_result.constraints:
            t.append("\n No constraints. Run a query first.\n", style=CD)
            rp.content = t
            return

        result = self.last_result
        t.append(f" Query: {result.query}\n", style=CD)
        t.append(f" {len(result.constraints)} aggregated claims\n\n", style=CD)

        for c in result.constraints[:20]:
            sc = TC.get(self.schema.types.get(c.subject, ""), C) if self.schema else C
            pc = TC.get(self.schema.types.get(c.predicate, ""), MAG) if self.schema else MAG
            oc = TC.get(self.schema.types.get(c.object, ""), GRN) if self.schema else GRN

            t.append(" (", style=CD)
            t.append(c.subject, style=sc)
            t.append(", ", style=CD)
            t.append(c.predicate, style=pc)
            t.append(", ", style=CD)
            t.append(c.object, style=oc)
            t.append(")\n", style=CD)

            bar = "\u2588" * max(1, int(c.score * 10))
            t.append(f"   ev={c.evidence} docs={c.doc_count} ", style=f"dim {W}")
            t.append(f"str={c.strength:.2f} ", style=W)
            t.append(f"{bar}\n", style=AMB)

        self.right_mode = "constraints"
        rp.content = t

    def _show_stats_right(self) -> None:
        rp = self.query_one(RightPanel)
        t = Text()
        t.append(" KNOWLEDGE GRAPH STATS\n", style=f"bold {C}")
        t.append(" \u2500" * 50 + "\n\n", style=CB)

        if not self.cog:
            t.append(" Engine not loaded.\n", style=CD)
            rp.content = t
            return

        stats = self.cog.stats()
        rows = [
            ("Infons", str(stats["infon_count"]), GRN),
            ("Constraints", str(stats["constraint_count"]), AMB),
            ("NEXT chains", "Yes" if stats["has_sequences"] else "No", C),
            ("Anchors", str(stats["anchors"]), C),
            ("Backend", stats["backend"], W),
            ("Model", stats["model"], W),
        ]
        for label, val, color in rows:
            t.append(f" {label:20s}", style=CD)
            t.append(f" {val}\n", style=f"bold {color}")

        if self.schema:
            by_type = Counter(self.schema.types.get(n, "?") for n in self.schema.names)
            t.append(f"\n ANCHOR TYPES\n", style=f"bold {C}")
            t.append(" \u2500" * 30 + "\n", style=CB)
            for atype, count in sorted(by_type.items()):
                color = TC.get(atype, W)
                glyph = TG.get(atype, "\u25cf")
                t.append(f" {glyph} {atype:12s}", style=color)
                t.append(f" {count}\n", style=f"bold {color}")

        if self.sheaf:
            t.append(f"\n SHEAF DIAGNOSTICS\n", style=f"bold {C}")
            t.append(" \u2500" * 30 + "\n", style=CB)
            t.append(f" Fiedler value:  ", style=CD)
            t.append(f"{self.sheaf.fiedler_value:.4f}\n", style=f"bold {AMB}")
            comps = self.sheaf.component_structure()
            t.append(f" Components:     ", style=CD)
            t.append(f"{len(comps)}\n", style=f"bold {W}")

            all_infons = self.cog.store.query_infons(limit=50000)
            coherences = [inf.coherence for inf in all_infons if inf.coherence > 0]
            if coherences:
                t.append(f" Mean coherence: ", style=CD)
                t.append(f"{np.mean(coherences):.3f}\n", style=f"bold {W}")

        self.right_mode = "stats"
        rp.content = t

    # ── Actions ─────────────────────────────────────────────────────────

    def action_show_help(self) -> None:
        self._show_help_left()

    def action_show_sheaf(self) -> None:
        self._show_sheaf_left()

    def action_show_timeline(self) -> None:
        self._show_timeline()

    def action_show_constraints(self) -> None:
        self._show_constraints()

    def action_show_anchors(self) -> None:
        self._show_anchors()

    def action_cycle_persona(self) -> None:
        self.persona_idx = (self.persona_idx + 1) % len(self.personas)
        self.current_persona = self.personas[self.persona_idx]
        self._render_title()
        self.query_one("#fkey-bar", FKeyBar).persona = self.current_persona
        self._render_status(f"Persona: {self.current_persona}")
        # Re-run last query if exists
        cmd = self.query_one("#cmd", CmdInput)
        if cmd.history:
            self.run_query(cmd.history[-1])

    def action_show_stats(self) -> None:
        self._show_stats_right()

    def action_clear(self) -> None:
        self.last_result = None
        self._show_anchors()
        self._show_help_right()
        self._render_status_counts()

    def action_ingest_demo(self) -> None:
        self._do_ingest(None)

    @work(thread=True, exclusive=True, group="ingest")
    def _do_ingest(self, path: str | None) -> None:
        if not self.cog:
            return
        self.app.call_from_thread(self._render_status, "Ingesting...")
        if path:
            with open(path) as f:
                docs = json.load(f)
        else:
            docs = DEMO_DOCS
        n = self.cog.ingest(docs, consolidate_now=True)
        self.app.call_from_thread(self._render_status, f"Ingested {n} infons from {len(docs)} docs")
        self.app.call_from_thread(self._render_status_counts)

    def action_focus_cmd(self) -> None:
        self.query_one("#cmd", CmdInput).focus()

    def action_toggle_panel(self) -> None:
        self.query_one("#cmd", CmdInput).focus()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cognition TUI")
    parser.add_argument("--schema", help="Path to schema JSON file")
    parser.add_argument("--db", default="data/cognition_mc.db", help="SQLite database path")
    args = parser.parse_args()
    CognitionApp(schema_path=args.schema, db_path=args.db).run()


if __name__ == "__main__":
    main()
