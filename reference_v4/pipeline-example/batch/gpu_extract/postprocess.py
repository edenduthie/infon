"""
Postprocess NuExtract-2.0 raw JSON output into step2-compatible v2 format.

Converts the enhanced multi-action NuExtract extraction (action-centric with
structured actors, influence patterns, and inter-action edges) into the
directed-hyperedge event structure expected by step2_ingest_lancedb/run.py's
_flatten_v2() function, and applies quality scoring.

New template structure:
  {
    "actions": [
      {
        "action_description", "l1_domain", "l2_subdomain", "l3_code",
        "escalation_classification", "influence_patterns", "scope", "claim_type",
        "contestation_dynamics",
        "initiators": [{"name","country_code","organisation","organisation_type","role","individual_name"}],
        "targets": [...],
        "locations", "date_raw", "date_iso", "date_precision", "assets"
      }
    ],
    "edges": [
      {"source_action_index", "target_action_index", "relation_type", "strength", "description"}
    ]
  }

Step2 v2 format expects:
  l1_domain, l2_subdomain, escalation_classification, scope, claim_type,
  dominant_action_type, initiators[], targets[], actors_flat[], actors{typed},
  actions[], actions_flat[], locations_flat[], dates[], edges[], edge_count,
  assets[], quality, _source{}
"""


# L1 code -> full name mapping for step2 compatibility
_L1_CODE_TO_NAME = {
    "D": "DIPLOMATIC",
    "I": "INFORMATIONAL",
    "M": "MILITARY",
    "E": "ECONOMIC",
    "F": "FINANCIAL",
    "IN": "INTELLIGENCE",
    "L": "LAW_ENFORCEMENT",
    "EN": "ENVIRONMENTAL",
    "DV": "DEVELOPMENT",
}

# Organisation type -> step2 actor_type mapping
_ORG_TYPE_TO_ACTOR_TYPE = {
    "government": "government_bodies",
    "military": "military_orgs",
    "intelligence": "military_orgs",
    "law-enforcement": "government_bodies",
    "legislature": "government_bodies",
    "judiciary": "government_bodies",
    "diplomatic": "government_bodies",
    "state-media": "government_bodies",
    "media": "actor_collectives",
    "civil-society": "actor_collectives",
    "private-sector": "actor_collectives",
    "armed-group": "military_orgs",
    "individual": "government_officials",
}


COST_MAG_ORDER = {"negligible": 0, "low": 1, "moderate": 2, "significant": 3, "major": 4}
RISK_ORDER = {"none": 0, "low": 1, "moderate": 2, "high": 3, "extreme": 4}
REV_ORDER = {"easily-reversible": 0, "reversible": 1, "difficult-to-reverse": 2, "irreversible": 3}

def _max_ordinal(values, order):
    """Return the maximum value by ordinal ranking."""
    filtered = [v for v in values if v in order]
    if not filtered:
        return ""
    return max(filtered, key=lambda x: order[x])

def _min_ordinal(values, order):
    """Return the minimum value by ordinal ranking (least reversible)."""
    filtered = [v for v in values if v in order]
    if not filtered:
        return ""
    return max(filtered, key=lambda x: order[x])  # max because higher = less reversible


def postprocess_nuextract_output(raw, url="", crawl_date="", title=""):
    """Convert NuExtract raw JSON into a step2-compatible v2 event dict.

    Args:
        raw: Parsed JSON dict from NuExtract model output
        url: Source URL of the WARC record
        crawl_date: Crawl date string
        title: Extracted HTML title

    Returns:
        dict matching the v2 event format expected by _flatten_v2(), or None
        if no actions were extracted (document not relevant).
    """
    if not isinstance(raw, dict):
        return None

    # Early exit: model judged article as not grayzone-relevant
    relevance = raw.get("grayzone_relevance", "")
    if relevance == "none":
        return None

    actions_raw = raw.get("actions", [])
    if not isinstance(actions_raw, list) or not actions_raw:
        return None

    # Filter to actions with actual content
    valid_actions = [a for a in actions_raw if isinstance(a, dict) and a.get("action_description")]
    if not valid_actions:
        return None

    # Use the primary-salience action as "dominant" for event-level classifications
    # Falls back to first action if none tagged primary
    dominant = next(
        (a for a in valid_actions if a.get("action_salience") == "primary"),
        valid_actions[0],
    )

    # Derive L1/L2/L3 code from L3 name via lookup table
    from nuextract_template import L3_NAME_TO_CODE
    l3_name = dominant.get("l3_code", "")
    l3_code = L3_NAME_TO_CODE.get(l3_name, "")
    l3_parts = l3_code.split(".")
    l1_code = l3_parts[0] if l3_parts else ""
    l1_domain = _L1_CODE_TO_NAME.get(l1_code, l1_code)
    l2_subdomain = ".".join(l3_parts[:2]) if len(l3_parts) >= 2 else ""
    escalation = dominant.get("escalation_classification", "ambiguous")
    scope = dominant.get("scope", "bilateral")
    claim_type = dominant.get("claim_type", "no-claims")

    # Aggregate across all actions
    all_initiators = []       # flat name list (deduped)
    all_targets = []          # flat name list (deduped)
    actors_typed = {          # grouped by step2 actor type
        "nation_states": [],
        "military_orgs": [],
        "government_bodies": [],
        "government_officials": [],
        "actor_roles": [],
        "actor_collectives": [],
    }
    actors_flat_set = set()
    actions_typed = []
    actions_flat = []
    locations_flat_set = set()
    locations_structured = []  # Full structured location objects
    dates = []
    assets_set = set()
    influence_patterns_all = []
    contestation_all = []

    initiator_names_seen = set()
    target_names_seen = set()

    # Type system aggregation accumulators
    narrative_framings_all = []
    situation_signals_agg = {
        "saving_face_indicators": [],
        "justice_penalty_indicators": [],
        "historical_references": [],
    }
    asymmetry_signals_all = []

    # New type system aggregation accumulators
    actor_postures = []
    regional_tensions = []
    state_descriptions = []
    resource_types_all = []
    cost_magnitudes = []
    risk_levels_all = []
    reversibilities = []
    surprise_elements = []
    ambiguity_levels = []
    cross_domain_linkages = []
    constancy_indicators = []
    situation_framings_list = []
    initiator_roles_all = []
    target_roles_all = []

    # Counterfactual / Implicit / Trade-off accumulators
    decision_contexts_agg = {
        "alternatives_mentioned": [],
        "constraints_cited": [],
        "preconditions": [],
        "enabling_factors": [],
        "conditional_threats": [],
        "explicit_counterfactuals": [],
    }
    implicit_signals_agg = {
        "stated_justifications": [],
        "unstated_objective_hints": [],
        "denials": [],
        "audience_targets": [],
        "capabilities_revealed": [],
        "precedent_implications": [],
    }
    trade_off_signals_agg = {
        "benefits_claimed": [],
        "costs_acknowledged": [],
        "costs_omitted_hints": [],
        "time_horizons": [],
        "dependencies_created": [],
        "gain_durabilities": [],
    }

    for idx, action in enumerate(valid_actions):
        desc = action.get("action_description", "")
        act_type = action.get("escalation_classification", "neutral")

        # Map escalation to a dominant_action_type-compatible value
        esc_to_action = {
            "escalation": "confrontational",
            "de-escalation": "cooperative",
            "cooperation-building": "cooperative",
            "cooperation-maintaining": "cooperative",
            "neutral": "diplomatic",
            "ambiguous": "unknown",
        }
        action_type_label = esc_to_action.get(act_type, "unknown")

        actions_typed.append({
            "type": action_type_label,
            "text": desc,
            "temporal_order": action.get("temporal_order", idx),
            "l3_code": L3_NAME_TO_CODE.get(action.get("l3_code", ""), ""),
            "l3_name": action.get("l3_code", ""),
            "action_salience": action.get("action_salience", ""),
            "narrative_role": action.get("narrative_role", ""),
            "retaliation_signal": action.get("retaliation_signal", "none"),
            "coordination_signal": action.get("coordination_signal", "none"),
            "influence_patterns": action.get("influence_patterns", []),
            "contestation_dynamics": action.get("contestation_dynamics", ""),
            "action_verb": action.get("action_verb", ""),
            "action_intensity": action.get("action_intensity", ""),
            "state_impact": action.get("state_impact", ""),
        })

        # Aggregate narrative framings
        for nf in action.get("narrative_framing", []):
            if nf and isinstance(nf, str):
                narrative_framings_all.append(nf)

        # Aggregate situation signals
        ss = action.get("situation_signals", {})
        if isinstance(ss, dict):
            for key in situation_signals_agg:
                for indicator in ss.get(key, []):
                    if indicator and isinstance(indicator, str) and indicator.strip():
                        situation_signals_agg[key].append(indicator.strip())

        # Collect asymmetry signals
        asym = action.get("asymmetry_signals", {})
        if isinstance(asym, dict) and any(asym.values()):
            asymmetry_signals_all.append(asym)

        # New type system field aggregation
        state_ind = action.get("state_indicators", {})
        if isinstance(state_ind, dict):
            ap = state_ind.get("actor_posture", "")
            if ap and isinstance(ap, str):
                actor_postures.append(ap)
            rt = state_ind.get("regional_tension", "")
            if rt and isinstance(rt, str):
                regional_tensions.append(rt)
            prior = state_ind.get("prior_state_description", "")
            resulting = state_ind.get("resulting_state_description", "")
            if prior or resulting:
                state_descriptions.append({"prior": prior or "", "resulting": resulting or ""})

        cost_sig = action.get("cost_signals", {})
        if isinstance(cost_sig, dict):
            rtype = cost_sig.get("resource_type", "")
            if rtype and isinstance(rtype, str):
                resource_types_all.append(rtype)
            cmag = cost_sig.get("cost_magnitude", "")
            if cmag and isinstance(cmag, str):
                cost_magnitudes.append(cmag)
            rlvl = cost_sig.get("risk_level", "")
            if rlvl and isinstance(rlvl, str):
                risk_levels_all.append(rlvl)
            rev = cost_sig.get("reversibility", "")
            if rev and isinstance(rev, str):
                reversibilities.append(rev)

        asym_sig = action.get("asymmetry_signals", {})
        if isinstance(asym_sig, dict):
            se = asym_sig.get("surprise_element", "")
            if se and isinstance(se, str):
                surprise_elements.append(se)
            al = asym_sig.get("ambiguity_level", "")
            if al and isinstance(al, str):
                ambiguity_levels.append(al)
            cdl = asym_sig.get("cross_domain_linkage", "")
            if cdl and isinstance(cdl, str) and cdl.strip():
                cross_domain_linkages.append(cdl.strip())
            ci = asym_sig.get("constancy_indicator", "")
            if ci and isinstance(ci, str):
                constancy_indicators.append(ci)

        sf = action.get("situation_framing", "")
        if sf and isinstance(sf, str):
            situation_framings_list.append(sf)

        arf = action.get("actor_role_framing", {})
        if isinstance(arf, dict):
            ir = arf.get("initiator_role", "")
            if ir and isinstance(ir, str):
                initiator_roles_all.append(ir)
            tr = arf.get("target_role", "")
            if tr and isinstance(tr, str):
                target_roles_all.append(tr)

        # Decision context aggregation
        dc = action.get("decision_context", {})
        if isinstance(dc, dict):
            for key in decision_contexts_agg:
                for item in dc.get(key, []):
                    if item and isinstance(item, str) and item.strip():
                        decision_contexts_agg[key].append(item.strip())

        # Implicit signals aggregation
        imp = action.get("implicit_signals", {})
        if isinstance(imp, dict):
            sj = imp.get("stated_justification", "")
            if sj and isinstance(sj, str) and sj.strip():
                implicit_signals_agg["stated_justifications"].append(sj.strip())
            uoh = imp.get("unstated_objective_hint", "")
            if uoh and isinstance(uoh, str):
                implicit_signals_agg["unstated_objective_hints"].append(uoh)
            dd = imp.get("denial_or_deflection", "")
            if dd and isinstance(dd, str) and dd.strip():
                implicit_signals_agg["denials"].append(dd.strip())
            for at in imp.get("audience_targeting", []):
                if at and isinstance(at, str):
                    implicit_signals_agg["audience_targets"].append(at)
            cr = imp.get("capability_revealed", "")
            if cr and isinstance(cr, str) and cr.strip():
                implicit_signals_agg["capabilities_revealed"].append(cr.strip())
            pi = imp.get("precedent_implications", "")
            if pi and isinstance(pi, str) and pi.strip():
                implicit_signals_agg["precedent_implications"].append(pi.strip())

        # Trade-off signals aggregation
        tos = action.get("trade_off_signals", {})
        if isinstance(tos, dict):
            bc = tos.get("benefit_claimed", "")
            if bc and isinstance(bc, str) and bc.strip():
                trade_off_signals_agg["benefits_claimed"].append(bc.strip())
            ca = tos.get("cost_acknowledged", "")
            if ca and isinstance(ca, str) and ca.strip():
                trade_off_signals_agg["costs_acknowledged"].append(ca.strip())
            coh = tos.get("cost_omitted_hint", "")
            if coh and isinstance(coh, str):
                trade_off_signals_agg["costs_omitted_hints"].append(coh)
            th = tos.get("time_horizon", "")
            if th and isinstance(th, str):
                trade_off_signals_agg["time_horizons"].append(th)
            dep = tos.get("dependency_created", "")
            if dep and isinstance(dep, str) and dep.strip():
                trade_off_signals_agg["dependencies_created"].append(dep.strip())
            rog = tos.get("reversibility_of_gain", "")
            if rog and isinstance(rog, str):
                trade_off_signals_agg["gain_durabilities"].append(rog)

        if desc:
            actions_flat.append(desc)

        # Structured initiators
        for actor in action.get("initiators", []):
            if not isinstance(actor, dict):
                continue
            name = actor.get("name", "").strip()
            if not name:
                continue
            # Prefer canonical_name for deduplication and downstream use
            canonical = actor.get("canonical_name", "").strip() or name
            if canonical not in initiator_names_seen:
                initiator_names_seen.add(canonical)
                all_initiators.append(canonical)
            _accumulate_actor(actor, actors_typed, actors_flat_set)

        # Structured targets
        for actor in action.get("targets", []):
            if not isinstance(actor, dict):
                continue
            name = actor.get("name", "").strip()
            if not name:
                continue
            canonical = actor.get("canonical_name", "").strip() or name
            if canonical not in target_names_seen:
                target_names_seen.add(canonical)
                all_targets.append(canonical)
            _accumulate_actor(actor, actors_typed, actors_flat_set)

        # Locations — handle both structured objects and legacy flat strings
        for loc in action.get("locations", []):
            if isinstance(loc, dict):
                loc_name = loc.get("name", "").strip()
                loc_canonical = loc.get("canonical_name", "").strip() or loc_name
                if loc_canonical:
                    locations_flat_set.add(loc_canonical)
                    locations_structured.append({
                        "name": loc_name,
                        "canonical_name": loc_canonical,
                        "level": loc.get("level", ""),
                        "country_code": loc.get("country_code", ""),
                        "macro_region": loc.get("macro_region", ""),
                    })
            elif isinstance(loc, str) and loc.strip():
                locations_flat_set.add(loc.strip())

        # Dates
        date_raw = action.get("date_raw", "")
        date_iso = action.get("date_iso", "")
        if date_raw or date_iso:
            dates.append({"raw": date_raw or "", "iso": date_iso or ""})

        # Assets
        for asset in action.get("assets", []):
            if asset and isinstance(asset, str) and asset.strip():
                assets_set.add(asset.strip())

        # Influence patterns
        for pat in action.get("influence_patterns", []):
            if pat and isinstance(pat, str):
                influence_patterns_all.append(pat)

        # Contestation dynamics
        cd = action.get("contestation_dynamics", "")
        if cd:
            contestation_all.append(cd)

    # Deduplicate typed actor lists
    for key in actors_typed:
        actors_typed[key] = list(dict.fromkeys(actors_typed[key]))

    actors_flat = list(dict.fromkeys(
        actors_typed["nation_states"] + actors_typed["military_orgs"] +
        actors_typed["government_bodies"] + actors_typed["government_officials"] +
        actors_typed["actor_roles"] + actors_typed["actor_collectives"]
    ))

    # Convert inter-action edges to entity-level edges
    edges_raw = raw.get("edges", [])
    edges = _convert_edges(edges_raw, valid_actions)

    # Determine dominant action type from primary-salience action
    dominant_action_type = "unknown"
    if actions_typed:
        # Find the action_typed entry matching the dominant action
        primary = next(
            (a for a in actions_typed if a.get("action_salience") == "primary"),
            actions_typed[0],
        )
        dominant_action_type = primary.get("type", "unknown")

    # Build v2 event structure (matches _flatten_v2 expectations)
    event = {
        # Classifications
        "l1_domain": l1_domain,
        "l2_subdomain": l2_subdomain,
        "l3_code": l3_code,
        "l3_name": l3_name,
        "escalation_classification": escalation,
        "scope": scope,
        "claim_type": claim_type,

        # Directed hyperedge
        "initiators": all_initiators,
        "targets": all_targets,
        "dominant_action_type": dominant_action_type,

        # Actors
        "actors": actors_typed,
        "actors_flat": actors_flat,

        # Actions
        "actions": actions_typed,
        "actions_flat": actions_flat,

        # Grayzone relevance (from model judgment)
        "grayzone_relevance": relevance or "high",

        # Locations
        "locations_flat": list(locations_flat_set),
        "locations": _dedup_locations(locations_structured),

        # Temporal
        "dates": dates,

        # Edges
        "edges": edges,
        "edge_count": len(edges),

        # Assets
        "assets": list(assets_set),

        # Enhanced fields (preserved for downstream use)
        "influence_patterns": list(dict.fromkeys(influence_patterns_all)),
        "contestation_dynamics": contestation_all[0] if contestation_all else "",

        # Downstream enrichment signals (for clustering, denoising, pattern mining)
        "has_retaliation": any(
            a.get("retaliation_signal") in ("explicit", "implied") for a in actions_typed
        ),
        "has_coordination": any(
            a.get("coordination_signal") in ("explicit", "implied") for a in actions_typed
        ),
        "retaliation_strength": _max_ordinal(
            [a.get("retaliation_signal", "none") for a in actions_typed],
            {"none": 0, "implied": 1, "explicit": 2},
        ),
        "coordination_strength": _max_ordinal(
            [a.get("coordination_signal", "none") for a in actions_typed],
            {"none": 0, "implied": 1, "explicit": 2},
        ),

        # Type system fields
        "state_indicators": {
            "actor_postures": list(dict.fromkeys(actor_postures)),
            "regional_tension": regional_tensions[0] if regional_tensions else "",
            "state_descriptions": state_descriptions,
        },
        "cost_profile": {
            "resource_types": list(dict.fromkeys(resource_types_all)),
            "max_cost_magnitude": _max_ordinal(cost_magnitudes, COST_MAG_ORDER),
            "max_risk_level": _max_ordinal(risk_levels_all, RISK_ORDER),
            "reversibility": _min_ordinal(reversibilities, REV_ORDER),
        },
        "asymmetry_profile": {
            "surprise": _max_ordinal(surprise_elements, {"none": 0, "tactical": 1, "strategic": 2}),
            "ambiguity": _max_ordinal(ambiguity_levels, {"none": 0, "low": 1, "moderate": 2, "high": 3}),
            "cross_domain_linkages": list(dict.fromkeys(cross_domain_linkages)),
            "constancy": _max_ordinal(constancy_indicators, {"one-off": 0, "repeated": 1, "sustained-campaign": 2, "escalating-series": 3}),
        },
        "situation_framing": list(dict.fromkeys(situation_framings_list)),
        "actor_roles": {
            "initiator_roles": list(dict.fromkeys(initiator_roles_all)),
            "target_roles": list(dict.fromkeys(target_roles_all)),
        },

        # Counterfactual / Implicit / Trade-off analysis signals
        "decision_context": {
            k: list(dict.fromkeys(v)) for k, v in decision_contexts_agg.items()
        },
        "implicit_signals": {
            k: list(dict.fromkeys(v)) for k, v in implicit_signals_agg.items()
        },
        "trade_off_signals": {
            k: list(dict.fromkeys(v)) for k, v in trade_off_signals_agg.items()
        },

        # Legacy type system fields
        "action_verb": dominant.get("action_verb", ""),
        "action_intensity": dominant.get("action_intensity", ""),
        "state_impact": dominant.get("state_impact", ""),
        "asymmetry_signals": asymmetry_signals_all[0] if asymmetry_signals_all else {},
        "narrative_framing": list(dict.fromkeys(narrative_framings_all)),
        "situation_signals": {
            k: list(dict.fromkeys(v)) for k, v in situation_signals_agg.items()
        },

        # Source metadata
        "_source": {
            "url": url,
            "crawl_date": crawl_date,
            "title": title,
        },
    }

    # Quality scoring
    event["quality"] = assess_event_quality(event)

    return event


def split_to_per_action_events(aggregated_event):
    """Split an aggregated multi-action event into per-action events.

    Each action becomes its own event with action-specific classifications,
    actors, edges, and quality scoring. Shared metadata (source, assets,
    grayzone_relevance) is inherited from the parent event.

    If there is only one action, returns [aggregated_event] unchanged.
    """
    actions_typed = aggregated_event.get("actions", [])
    if len(actions_typed) <= 1:
        return [aggregated_event]

    from nuextract_template import L3_NAME_TO_CODE

    # Shared fields inherited by all per-action events
    source = aggregated_event.get("_source", {})
    grayzone_relevance = aggregated_event.get("grayzone_relevance", "high")

    # Build a lookup of action description -> edges from the aggregated event
    all_edges = aggregated_event.get("edges", [])

    per_action_events = []

    for idx, action in enumerate(actions_typed):
        action_desc = action.get("text", "")
        l3_name = action.get("l3_name", "")
        l3_code = L3_NAME_TO_CODE.get(l3_name, "")
        l3_parts = l3_code.split(".")
        l1_code = l3_parts[0] if l3_parts else ""
        l1_domain = _L1_CODE_TO_NAME.get(l1_code, l1_code)
        l2_subdomain = ".".join(l3_parts[:2]) if len(l3_parts) >= 2 else ""

        # Map escalation to action type
        esc_to_action = {
            "escalation": "confrontational",
            "de-escalation": "cooperative",
            "cooperation-building": "cooperative",
            "cooperation-maintaining": "cooperative",
            "neutral": "diplomatic",
            "ambiguous": "unknown",
        }
        dominant_action_type = esc_to_action.get(action.get("type", ""), "unknown")

        # Filter edges: keep edges that reference this action's description
        action_edges = [
            e for e in all_edges
            if e.get("source") == action_desc or e.get("target") == action_desc
        ]

        # Build per-action initiators/targets from edges
        initiators = []
        targets = []
        initiator_set = set()
        target_set = set()
        for edge in action_edges:
            if edge.get("relation_type") == "INITIATES_ACTION" and edge.get("source"):
                name = edge["source"]
                if name not in initiator_set:
                    initiator_set.add(name)
                    initiators.append(name)
            elif edge.get("relation_type") == "ACTION_TARGETS" and edge.get("target"):
                name = edge["target"]
                if name not in target_set:
                    target_set.add(name)
                    targets.append(name)

        # Fall back to aggregated initiators/targets if edges don't provide them
        if not initiators:
            initiators = list(aggregated_event.get("initiators", []))
        if not targets:
            targets = list(aggregated_event.get("targets", []))

        # Build actors_flat from initiators + targets
        actors_flat = list(dict.fromkeys(initiators + targets))

        # Build actors_typed — simplified from parent, scoped to this action's actors
        parent_actors = aggregated_event.get("actors", {})
        actors_typed = {}
        action_actor_set = set(actors_flat)
        for key, members in parent_actors.items():
            filtered = [m for m in members if m in action_actor_set]
            actors_typed[key] = filtered

        # Action-specific locations
        action_locations = []
        for edge in action_edges:
            if edge.get("relation_type") == "ACTION_AT_LOCATION" and edge.get("target"):
                action_locations.append(edge["target"])
        if not action_locations:
            action_locations = list(aggregated_event.get("locations_flat", []))

        # Action-specific dates
        action_dates = []
        for edge in action_edges:
            if edge.get("relation_type") == "OCCURRED_ON" and edge.get("target"):
                action_dates.append({"raw": "", "iso": edge["target"]})
        if not action_dates:
            action_dates = list(aggregated_event.get("dates", []))

        # Inherit assets from parent
        assets = list(aggregated_event.get("assets", []))

        evt = {
            # Classifications (action-specific)
            "l1_domain": l1_domain,
            "l2_subdomain": l2_subdomain,
            "l3_code": l3_code,
            "l3_name": l3_name,
            "escalation_classification": action.get("type", "unknown"),
            "scope": aggregated_event.get("scope", "bilateral"),
            "claim_type": aggregated_event.get("claim_type", "no-claims"),

            # Directed hyperedge
            "initiators": initiators,
            "targets": targets,
            "dominant_action_type": dominant_action_type,

            # Actors
            "actors": actors_typed,
            "actors_flat": actors_flat,

            # Actions (single action for this event)
            "actions": [action],
            "actions_flat": [action_desc] if action_desc else [],

            # Grayzone relevance
            "grayzone_relevance": grayzone_relevance,

            # Locations
            "locations_flat": list(dict.fromkeys(action_locations)),
            "locations": aggregated_event.get("locations", []),

            # Temporal
            "dates": action_dates,

            # Edges (only edges referencing this action)
            "edges": action_edges,
            "edge_count": len(action_edges),

            # Assets
            "assets": assets,

            # Enhanced fields (from action)
            "influence_patterns": action.get("influence_patterns", []),
            "contestation_dynamics": action.get("contestation_dynamics", ""),

            # Downstream enrichment signals
            "has_retaliation": action.get("retaliation_signal") in ("explicit", "implied"),
            "has_coordination": action.get("coordination_signal") in ("explicit", "implied"),
            "retaliation_strength": action.get("retaliation_signal", "none"),
            "coordination_strength": action.get("coordination_signal", "none"),

            # Type system fields (inherited from parent, action-level where available)
            "state_indicators": aggregated_event.get("state_indicators", {}),
            "cost_profile": aggregated_event.get("cost_profile", {}),
            "asymmetry_profile": aggregated_event.get("asymmetry_profile", {}),
            "situation_framing": aggregated_event.get("situation_framing", []),
            "actor_roles": aggregated_event.get("actor_roles", {}),

            # Counterfactual / Implicit / Trade-off signals (inherited)
            "decision_context": aggregated_event.get("decision_context", {}),
            "implicit_signals": aggregated_event.get("implicit_signals", {}),
            "trade_off_signals": aggregated_event.get("trade_off_signals", {}),

            # Legacy type system fields
            "action_verb": action.get("action_verb", ""),
            "action_intensity": action.get("action_intensity", ""),
            "state_impact": action.get("state_impact", ""),
            "asymmetry_signals": aggregated_event.get("asymmetry_signals", {}),
            "narrative_framing": action.get("narrative_framing", []) if isinstance(action.get("narrative_framing"), list) else aggregated_event.get("narrative_framing", []),
            "situation_signals": aggregated_event.get("situation_signals", {}),

            # Source metadata
            "_source": source,
            "_parent_event": True,
            "_action_index": idx,
        }

        # Re-score quality for this per-action event
        evt["quality"] = assess_event_quality(evt)
        per_action_events.append(evt)

    return per_action_events


def _dedup_locations(locations):
    """Deduplicate structured location objects by canonical_name."""
    seen = set()
    result = []
    for loc in locations:
        key = loc.get("canonical_name", "") or loc.get("name", "")
        if key and key not in seen:
            seen.add(key)
            result.append(loc)
    return result


def _accumulate_actor(actor, actors_typed, actors_flat_set):
    """Add a structured actor to the typed groups and flat set."""
    name = actor.get("name", "").strip()
    if not name:
        return

    # Prefer canonical_name for consistent deduplication
    canonical = actor.get("canonical_name", "").strip() or name

    org_type = actor.get("organisation_type", "")
    actor_type_key = _ORG_TYPE_TO_ACTOR_TYPE.get(org_type, "actor_roles")

    # For individuals, use individual_name if available, else org name
    individual = actor.get("individual_name", "").strip()
    if individual and org_type == "individual":
        if individual not in actors_typed.get("government_officials", []):
            actors_typed["government_officials"].append(individual)
        actors_flat_set.add(individual)

    # Use country_code to infer nation_state
    country = actor.get("country_code", "").strip()
    if country and len(country) >= 2:
        # Store as nation if it looks like a country reference
        if country not in actors_typed.get("nation_states", []):
            actors_typed["nation_states"].append(country)

    # Add canonical name to appropriate type bucket
    if canonical not in actors_typed.get(actor_type_key, []):
        actors_typed[actor_type_key].append(canonical)
    actors_flat_set.add(canonical)

    # Also add organisation name if different from canonical name
    org = actor.get("organisation", "").strip()
    if org and org != canonical:
        if org not in actors_typed.get(actor_type_key, []):
            actors_typed[actor_type_key].append(org)
        actors_flat_set.add(org)


def _convert_edges(edges_raw, valid_actions):
    """Convert action-indexed edges to entity-level edges for step2.

    New format edges reference source/target action indices.
    We convert these to edges that reference the action descriptions
    and their initiator/target entities, producing INITIATES_ACTION,
    ACTION_TARGETS, and inter-action relation edges.
    """
    edges = []

    # First, generate structural edges from each action's initiators/targets
    for action in valid_actions:
        desc = action.get("action_description", "")
        if not desc:
            continue

        for actor in action.get("initiators", []):
            if isinstance(actor, dict) and actor.get("name"):
                actor_label = (actor.get("canonical_name", "") or actor["name"]).strip()
                edges.append({
                    "source": actor_label,
                    "target": desc,
                    "relation_type": "INITIATES_ACTION",
                })

        for actor in action.get("targets", []):
            if isinstance(actor, dict) and actor.get("name"):
                actor_label = (actor.get("canonical_name", "") or actor["name"]).strip()
                edges.append({
                    "source": desc,
                    "target": actor_label,
                    "relation_type": "ACTION_TARGETS",
                })

        # Location edges — handle structured objects and legacy strings
        for loc in action.get("locations", []):
            loc_label = ""
            if isinstance(loc, dict):
                loc_label = (loc.get("canonical_name", "") or loc.get("name", "")).strip()
            elif isinstance(loc, str):
                loc_label = loc.strip()
            if loc_label:
                edges.append({
                    "source": desc,
                    "target": loc_label,
                    "relation_type": "ACTION_AT_LOCATION",
                })

        # Date edges
        date_raw = action.get("date_raw", "")
        date_iso = action.get("date_iso", "")
        if date_raw or date_iso:
            edges.append({
                "source": desc,
                "target": date_iso or date_raw,
                "relation_type": "OCCURRED_ON",
            })

    # Then, convert inter-action edges
    if isinstance(edges_raw, list):
        for e in edges_raw:
            if not isinstance(e, dict):
                continue
            src_idx = e.get("source_action_index")
            tgt_idx = e.get("target_action_index")
            rel_type = e.get("relation_type", "")

            if src_idx is None or tgt_idx is None:
                continue
            if not isinstance(src_idx, int) or not isinstance(tgt_idx, int):
                continue
            if src_idx < 0 or src_idx >= len(valid_actions):
                continue
            if tgt_idx < 0 or tgt_idx >= len(valid_actions):
                continue

            src_desc = valid_actions[src_idx].get("action_description", "")
            tgt_desc = valid_actions[tgt_idx].get("action_description", "")
            if src_desc and tgt_desc:
                # Map new relation types to uppercase step2-compatible names
                mapped_rel = rel_type.upper().replace("-", "_")
                edges.append({
                    "source": src_desc,
                    "target": tgt_desc,
                    "relation_type": mapped_rel,
                    "causal_link_type": e.get("causal_link_type", "unknown"),
                    "strength": e.get("strength", ""),
                    "description": e.get("description", ""),
                })

    return edges


def assess_event_quality(event):
    """Quality assessment for the enhanced multi-action format.

    Scoring (0-24 points):
      +2 pts: L1 domain identified
      +1 pt:  L2 subdomain identified
      +1 pt:  Escalation classification (not ambiguous)
      +2 pts: 2+ actors in actors_flat
      +1 pt:  Has identified initiators AND targets (directed hyperedge)
      +2 pts: 2+ specific actions extracted
      +2 pts: Has locations
      +1 pt:  Temporal information
      +1 pt:  Physical assets
      +3 pts: 2+ relation edges (directed graph structure)
      +2 pts: 1 relation edge
      +1 pt:  Has influence patterns
      +1 pt:  Has contestation dynamics
      +1 pt:  Has action verb (type system)
      +1 pt:  Has non-status-quo state impact (type system)
      +1 pt:  Has narrative framing (type system)
      +1 pt:  Has situation signals (type system)

    Quality levels: HIGH (11+), MEDIUM (6-10), LOW (<6)
    """
    score = 0

    # L1 domain
    l1 = event.get("l1_domain", "")
    if l1 and l1 != "NONE":
        score += 2

    # L2 subdomain
    if event.get("l2_subdomain"):
        score += 1

    # L3 classification depth
    if event.get("l3_code"):
        score += 1

    # Escalation classification
    esc = event.get("escalation_classification", "")
    if esc and esc != "ambiguous":
        score += 1

    # Actors
    actors_flat = event.get("actors_flat", [])
    if len(actors_flat) >= 2:
        score += 2
    elif len(actors_flat) == 1:
        score += 1

    # Directed hyperedge completeness
    if event.get("initiators") and event.get("targets"):
        score += 1

    # Actions
    actions_flat = event.get("actions_flat", [])
    if len(actions_flat) >= 2:
        score += 2
    elif len(actions_flat) == 1:
        score += 1

    # Locations
    if event.get("locations_flat"):
        score += 2

    # Temporal
    if event.get("dates"):
        score += 1

    # Assets
    if event.get("assets"):
        score += 1

    # Relation edges
    edge_count = event.get("edge_count", 0)
    if edge_count >= 2:
        score += 3
    elif edge_count == 1:
        score += 2

    # Influence patterns
    if event.get("influence_patterns"):
        score += 1

    # Contestation dynamics
    if event.get("contestation_dynamics"):
        score += 1

    # Type system fields
    if event.get("action_verb"):
        score += 1

    state_impact = event.get("state_impact", "")
    if state_impact and state_impact != "status_quo":
        score += 1

    if event.get("narrative_framing"):
        score += 1

    ss = event.get("situation_signals", {})
    if isinstance(ss, dict) and any(ss.get(k) for k in ss):
        score += 1

    # Type system fields
    # State indicators
    state_ind = event.get("state_indicators", {})
    if state_ind.get("actor_postures"):
        score += 1

    # Cost profile
    cost_prof = event.get("cost_profile", {})
    if cost_prof.get("resource_types") and cost_prof["resource_types"] != ["none-apparent"]:
        score += 1

    # Asymmetry profile
    asym = event.get("asymmetry_profile", {})
    if any(asym.get(k) and asym[k] not in ("", "none") for k in ["surprise", "ambiguity", "constancy"]):
        score += 1

    # Situation framing
    if event.get("situation_framing") and event["situation_framing"] != ["none-clear"]:
        score += 1

    # Actor roles
    roles = event.get("actor_roles", {})
    if roles.get("initiator_roles") or roles.get("target_roles"):
        score += 1

    # Decision context signals
    dc = event.get("decision_context", {})
    if isinstance(dc, dict) and any(dc.get(k) for k in dc):
        score += 1

    # Implicit signals
    imp = event.get("implicit_signals", {})
    if isinstance(imp, dict):
        if imp.get("unstated_objective_hints") and imp["unstated_objective_hints"] != ["none-apparent"]:
            score += 1

    # Trade-off signals
    tos = event.get("trade_off_signals", {})
    if isinstance(tos, dict) and (tos.get("benefits_claimed") or tos.get("costs_acknowledged")):
        score += 1

    # Downstream enrichment signals
    if event.get("has_retaliation"):
        score += 1
    if event.get("has_coordination"):
        score += 1

    if score >= 12:
        return "HIGH"
    elif score >= 7:
        return "MEDIUM"
    else:
        return "LOW"
