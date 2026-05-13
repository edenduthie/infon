# Module 08: LLM Agent Tools


> **Update for the cassette substrate.** The 8 @tool-decorated functions here are superseded by the 9-tool `Analyst` class in `cognition.cassette.analyst`. The Analyst wraps a Strands agent around a store with: `set_schema`, `ingest`, `reingest`, `extraction_report`, `ask`, `connect`, `any_of`, `record_finding`, `list_findings`. See notebook 06 for the end-to-end walkthrough.

## What You'll Learn

- How to expose concept-space operations to LLM agents as callable tools
- Designing tool interfaces that guide good query strategies
- Building a system prompt that teaches the agent your domain's concept basis

## Background

The change of basis gave us a knowledge graph in concept coordinates. Agent tools expose those coordinates as operations an LLM can invoke: browse the basis (`list_anchors`), query by concept coordinates (`find_infons`), walk temporal trajectories (`get_timeline`), or run a full change-of-basis query (`ask_cognition`). The agent navigates concept space by composing tool calls: "List anchors → find constraints for Toyota → get evidence → compare with BMW → summarize."

We use the `@tool` decorator pattern (compatible with Strands, LangChain, or any tool-use framework).

## Prompt 1: Build Agent Tools

---

```
Create cognition/src/cognition/agent_tools.py with @tool-decorated functions:

1. list_anchors(anchor_type) — browse the vocabulary
2. find_infons(subject, predicate, object, min_importance) — search infons
3. get_evidence(subject, predicate, object) — drill into evidence sentences
4. get_constraints(subject, predicate, object) — corpus-level assertions
5. get_timeline(anchor) — temporal evolution of an entity
6. compare_entities(entity_a, entity_b) — side-by-side constraint comparison
7. ask_cognition(question, persona) — full natural language query
8. get_stats() — knowledge graph overview

Each tool returns formatted text that's easy for an LLM to parse.

Also create create_tools(cognition_instance) that:
1. Sets up module-level state
2. Builds a domain-aware system prompt from the schema
3. Returns (tools_list, system_prompt)
```

---

## Prompt 2: Test with an Agent

---

```
Set up a quick test where an LLM agent uses these tools to answer:

"Compare Toyota and BMW's investment strategies in electric vehicles. 
Which company is better positioned for the EV transition?"

Show me:
1. What tools the agent calls and in what order
2. The key evidence it finds
3. Its final analysis

This tests the full chain: agent → tools → cognition → store → response.
```

---

## Tool Design Principles

1. **Start broad, drill deep**: `list_anchors` → `find_infons` → `get_evidence`
2. **Every tool returns text**: LLMs work with text, not data structures
3. **Limit results**: always cap at a reasonable `limit` to avoid context overflow
4. **Include scores**: show confidence/importance so the agent can assess quality
5. **Ground in evidence**: `get_evidence` shows actual sentences, not just triples

## Checkpoint

- [ ] All 8 tools return well-formatted text
- [ ] `create_tools()` returns a usable system prompt
- [ ] An agent can compose multi-step queries using the tools
- [ ] The system prompt accurately describes your domain's schema
