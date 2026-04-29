"""
FastMCP-based stdio MCP server for infon.

This module implements the MCP server exposing three tools and three resources:

Tools:
- search(query, limit): Semantic search over the knowledge base
- store_observation(text, source): Store agent observations as infons
- query_ast(symbol, relation, limit): Query AST-derived code relationships

Resources:
- infon://stats: Store statistics
- infon://schema: Active anchor schema
- infon://recent: 20 most recent infons
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from infon.consolidate import consolidate
from infon.encoder import encode
from infon.extract import extract_text
from infon.infon import Infon
from infon.schema import AnchorSchema
from infon.store import InfonStore


# Initialize FastMCP server
mcp = FastMCP("infon", version="0.1.0")


# Global state for store and schema (initialized in run_server)
_store: InfonStore | None = None
_schema: AnchorSchema | None = None


def _init_server(db_path: Path) -> tuple[InfonStore, AnchorSchema]:
    """
    Initialize the MCP server with store and schema.
    
    Args:
        db_path: Path to the DuckDB database file
        
    Returns:
        Tuple of (store, schema)
    """
    # Load schema from adjacent schema.json
    schema_path = db_path.parent / "schema.json"
    
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found at {schema_path}")
    
    with open(schema_path, "r") as f:
        schema_data = json.load(f)
    
    schema = AnchorSchema(**schema_data)
    
    # Open store
    store = InfonStore(db_path)
    
    return store, schema


def _infon_to_dict(infon: Infon, score: float | None = None) -> dict[str, Any]:
    """
    Convert an Infon to a JSON-serializable dict for MCP responses.
    
    Args:
        infon: The Infon to convert
        score: Optional relevance score
        
    Returns:
        Dictionary representation
    """
    grounding_dict = {}
    if infon.grounding.root.grounding_type == "ast":
        grounding_dict = {
            "type": "ast",
            "file_path": infon.grounding.root.file_path,
            "line_number": infon.grounding.root.line_number,
            "node_type": infon.grounding.root.node_type,
        }
    else:  # text
        grounding_dict = {
            "type": "text",
            "doc_id": infon.grounding.root.doc_id,
            "sent_id": infon.grounding.root.sent_id,
            "char_start": infon.grounding.root.char_start,
            "char_end": infon.grounding.root.char_end,
            "sentence_text": infon.grounding.root.sentence_text,
        }
    
    result = {
        "subject": infon.subject,
        "predicate": infon.predicate,
        "object": infon.object,
        "polarity": infon.polarity,
        "confidence": infon.confidence,
        "grounding": grounding_dict,
        "context": [],  # TODO: Add NEXT-edge context in Phase 8
    }
    
    if score is not None:
        result["score"] = score
    
    return result


def _simple_retrieve(query: str, limit: int = 10) -> list[tuple[Infon, float]]:
    """
    Simple retrieval function (placeholder for Phase 8 full implementation).
    
    This is a minimal version that:
    1. Encodes the query to anchor space
    2. Finds infons matching any activated anchor
    3. Scores by activation strength + reinforcement
    4. Returns top-K ranked results
    
    Args:
        query: Natural language query
        limit: Maximum number of results
        
    Returns:
        List of (Infon, score) tuples sorted by score descending
    """
    if _schema is None or _store is None:
        return []
    
    # Encode query to anchor space
    anchor_scores = encode(query, _schema)
    
    if not anchor_scores:
        return []
    
    # Get top-K activated anchors
    top_anchors = sorted(anchor_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Query store for infons containing these anchors
    results: list[tuple[Infon, float]] = []
    
    for anchor, activation in top_anchors:
        # Query by subject
        subject_matches = _store.query(subject=anchor, limit=limit)
        for infon in subject_matches:
            # Score = activation * (1 + reinforcement_count/10)
            score = activation * (1.0 + infon.reinforcement_count / 10.0)
            results.append((infon, score))
        
        # Query by object
        object_matches = _store.query(object=anchor, limit=limit)
        for infon in object_matches:
            score = activation * (1.0 + infon.reinforcement_count / 10.0)
            results.append((infon, score))
    
    # Deduplicate and sort by score
    seen_ids = set()
    unique_results = []
    for infon, score in results:
        if infon.id not in seen_ids:
            seen_ids.add(infon.id)
            unique_results.append((infon, score))
    
    unique_results.sort(key=lambda x: x[1], reverse=True)
    
    return unique_results[:limit]


@mcp.tool()
def search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Search the knowledge base with semantic ranking.
    
    Encodes the query to anchor space and returns ranked infons.
    
    Args:
        query: Natural language search query
        limit: Maximum number of results (default 10)
        
    Returns:
        List of ranked infon dicts with subject, predicate, object, score, grounding
    """
    try:
        results = _simple_retrieve(query, limit)
        return [_infon_to_dict(infon, score) for infon, score in results]
    except Exception as e:
        # Return error as dict (graceful handling)
        return [{"error": str(e), "query": query}]


@mcp.tool()
def store_observation(text: str, source: str = "agent") -> dict[str, int]:
    """
    Store an agent observation as infons.
    
    Extracts triples from the text, persists them, and runs consolidation.
    
    Args:
        text: The observation text to extract and store
        source: Source identifier (default "agent")
        
    Returns:
        Summary with infons_added and infons_reinforced counts
    """
    try:
        if _schema is None or _store is None:
            return {"error": "Server not initialized", "infons_added": 0, "infons_reinforced": 0}
        
        # Construct doc_id as "<source>:<timestamp>"
        timestamp = datetime.now(timezone.utc).isoformat()
        doc_id = f"{source}:{timestamp}"
        
        # Extract infons from text
        infons = extract_text(text, doc_id, _schema)
        
        # Track added vs reinforced
        infons_added = 0
        infons_reinforced = 0
        
        # Upsert each infon
        for infon in infons:
            # Check if this triple already exists
            existing = _store.query(
                subject=infon.subject,
                predicate=infon.predicate,
                object=infon.object,
                limit=1,
            )
            
            if existing and existing[0].polarity == infon.polarity:
                infons_reinforced += 1
            else:
                infons_added += 1
            
            _store.upsert(infon)
        
        # Run consolidation
        consolidate(_store, _schema)
        
        return {
            "infons_added": infons_added,
            "infons_reinforced": infons_reinforced,
        }
    except Exception as e:
        return {"error": str(e), "infons_added": 0, "infons_reinforced": 0}


@mcp.tool()
def query_ast(symbol: str, relation: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """
    Query AST-derived code relationships by symbol.
    
    Finds infons where the symbol appears as subject or object,
    optionally filtered by relation type.
    
    Args:
        symbol: The symbol/anchor to query for
        relation: Optional predicate filter (e.g., "calls", "imports")
        limit: Maximum number of results (default 20)
        
    Returns:
        List of infons sorted by reinforcement_count descending
    """
    try:
        if _store is None:
            return [{"error": "Server not initialized"}]
        
        # Query for infons with this symbol
        subject_matches = _store.query(subject=symbol, predicate=relation, limit=limit)
        object_matches = _store.query(object=symbol, predicate=relation, limit=limit)
        
        # Combine and deduplicate
        all_matches = subject_matches + object_matches
        seen_ids = set()
        unique_matches = []
        for infon in all_matches:
            if infon.id not in seen_ids:
                seen_ids.add(infon.id)
                unique_matches.append(infon)
        
        # Sort by reinforcement_count descending
        unique_matches.sort(key=lambda i: i.reinforcement_count, reverse=True)
        
        # Convert to dicts
        return [_infon_to_dict(infon) for infon in unique_matches[:limit]]
    except Exception as e:
        return [{"error": str(e), "symbol": symbol}]


@mcp.resource("infon://stats")
def get_stats() -> str:
    """
    Get store statistics as Markdown.
    
    Returns:
        Markdown-formatted statistics
    """
    if _store is None:
        return "Error: Store not initialized"
    
    try:
        stats = _store.stats()
        
        md = f"""# Infon Knowledge Base Statistics

- **Infons**: {stats.infon_count}
- **Edges**: {stats.edge_count}
- **Constraints**: {stats.constraint_count}
- **Documents**: {stats.document_count}

## Top Anchors

"""
        for anchor, count in stats.top_anchors:
            md += f"- `{anchor}`: {count}\n"
        
        return md
    except Exception as e:
        return f"Error: {e}"


@mcp.resource("infon://schema")
def get_schema() -> str:
    """
    Get the active schema as Markdown.
    
    Returns:
        Markdown-formatted anchor schema grouped by type
    """
    if _schema is None:
        return "Error: Schema not initialized"
    
    try:
        md = f"""# Anchor Schema

**Version**: {_schema.version}  
**Language**: {_schema.language}

## Actors

"""
        for anchor in _schema.actors:
            md += f"- **{anchor.key}**: {anchor.description}\n"
        
        md += "\n## Relations\n\n"
        for anchor in _schema.relations:
            md += f"- **{anchor.key}**: {anchor.description}\n"
        
        md += "\n## Features\n\n"
        for anchor in _schema.features:
            md += f"- **{anchor.key}**: {anchor.description}\n"
        
        return md
    except Exception as e:
        return f"Error: {e}"


@mcp.resource("infon://recent")
def get_recent() -> str:
    """
    Get the 20 most recent infons as a Markdown table.
    
    Returns:
        Markdown table of recent infons
    """
    if _store is None:
        return "Error: Store not initialized"
    
    try:
        # Query recent infons (DuckDB doesn't have ORDER BY in query method, so we do it manually)
        conn = _store._conn
        rows = conn.execute(
            """
            SELECT id, subject, predicate, object, polarity,
                   grounding_type, grounding_json,
                   confidence, timestamp, importance_json, kind,
                   reinforcement_count
            FROM infons
            ORDER BY created_at DESC
            LIMIT 20
            """
        ).fetchall()
        
        infons = [_store._row_to_infon(row) for row in rows]
        
        md = """# Recent Infons

| Subject | Predicate | Object | Polarity | Confidence |
|---------|-----------|--------|----------|------------|
"""
        for infon in infons:
            polarity_str = "✓" if infon.polarity else "✗"
            md += f"| {infon.subject} | {infon.predicate} | {infon.object} | {polarity_str} | {infon.confidence:.2f} |\n"
        
        return md
    except Exception as e:
        return f"Error: {e}"


def run_server(db_path: str | None = None) -> None:
    """
    Run the FastMCP stdio server.
    
    Args:
        db_path: Path to DuckDB database (default: .infon/kb.ddb)
    """
    global _store, _schema
    
    # Determine database path
    if db_path is None:
        db_path_obj = Path.cwd() / ".infon" / "kb.ddb"
    else:
        db_path_obj = Path(db_path)
    
    # Initialize server
    _store, _schema = _init_server(db_path_obj)
    
    try:
        # Run FastMCP stdio server
        mcp.run()
    finally:
        # Clean up on shutdown
        if _store is not None:
            _store.close()


def main():
    """Main entry point for command-line invocation."""
    parser = argparse.ArgumentParser(description="Infon MCP Server")
    parser.add_argument("--db", type=str, help="Path to DuckDB database")
    
    args = parser.parse_args()
    
    run_server(args.db)


if __name__ == "__main__":
    main()
