# Benchmark — cognition pipeline at scale

Wall-clock timings on a single laptop CPU (macOS, M-series). All numbers reflect the quality-filtered extraction (quality_threshold=0.04, max_triples_per_sentence=2).

| Scale (target) | Actual infons | Nodes | Edges | Init (ms) | Ingest (ms) | Rate (inf/s) | Graph build (ms) | GNN fit (ms) | Query p50 (ms) | Query p99 (ms) | ΔRSS (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 500 | 90 | 109 | 540 | 1086 | 546 | 165 | 705 | 837 | 573 | 604 | 316 |
| 1500 | 294 | 313 | 1764 | 55 | 1114 | 264 | 1825 | 2811 | 3443 | 3862 | 229 |
| 5000 | 928 | 947 | 5568 | 68 | 4027 | 230 | 9180 | 17012 | 6368 | 6544 | 179 |

## How to read this table

- **Scale (target)** is the number of infons we asked the synthetic generator to produce. **Actual** is what survived quality filtering — typically 15–25% of target.
- **Ingest rate** is the sustained throughput of the SPLADE encoder + extractor + SQLite store.
- **Graph build** is the cost to materialize the hypergraph tensor from the store for a given query.
- **GNN fit** is 15 epochs of transductive training.
- **Query p50/p99** is the per-query latency distribution (rebuilds the graph + runs reasoner each time — an upper bound; cached-graph latency is much lower).
