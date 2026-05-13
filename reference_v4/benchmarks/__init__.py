"""NeurIPS 2026 benchmarks: Cognition vs RAG on FEVER and HoVer.

Three experimental conditions:
  1. RAG (baseline) — dense retrieval, text snippets, no structure
  2. Cognition-fixed — hand-crafted Wikipedia-general schema
  3. Cognition-discovered — Kan extension (schema-free discovery)

Two benchmarks:
  A. FEVER — atomic evidence retrieval (infons vs text snippets)
  B. HoVer — multi-hop reasoning (hyperedges vs document links)
"""
