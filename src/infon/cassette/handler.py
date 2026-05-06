"""AWS Lambda handler for cassette ingestion.

This file is the entrypoint ship as the Lambda function body; it's tiny
by design. All real work happens in the layer (infon package +
torch-cpu + transformers + the 17MB SPLADE model).

Event shape (what LambdaExecutor sends):
    {
      "schema_s3": "s3://bucket/schemas/my_schema.json",
      "schema_ref": "abc123",
      "out_root":   "s3://bucket/data",
      "jobs": [
          {"doc": {"id": "...", "text": "...", "timestamp": "..."},
           "cassette_id": "..."},
          ...
      ]
    }

Response shape:
    {
      "summaries": [
          {"cassette_id": "...", "cassette_path": "s3://...", ...},
          ...
      ]
    }

Cold start: ~400-800ms (load SPLADE from the layer).
Warm invocation: ~100ms per doc.
"""

from __future__ import annotations

import json


# Worker globals — Lambda reuses containers between invocations, so
# loading SPLADE once per container amortizes across many invocations.
_ENCODER = None
_SCHEMA = None
_SCHEMA_PATH_CACHED: str | None = None


def _load_schema_from_s3(s3_uri: str) -> str:
    """Download a schema JSON to /tmp and return the local path."""
    import fsspec
    local_path = "/tmp/schema.json"
    with fsspec.open(s3_uri, "r") as src, open(local_path, "w") as dst:
        dst.write(src.read())
    return local_path


def handler(event: dict, _context) -> dict:
    """Lambda entrypoint."""
    global _ENCODER, _SCHEMA, _SCHEMA_PATH_CACHED

    from infon.schema import AnchorSchema
    from infon.encoder import Encoder
    from infon.config import InfonConfig
    from infon.extract import extract_infons
    from infon.cassette.format import CassetteWriter
    from infon.cassette.index import build_indexes

    schema_s3 = event["schema_s3"]
    schema_ref = event["schema_ref"]
    out_root = event["out_root"]
    jobs = event["jobs"]

    # Load schema + encoder — cached across warm invocations.
    if _SCHEMA_PATH_CACHED != schema_s3:
        schema_path = _load_schema_from_s3(schema_s3)
        _SCHEMA = AnchorSchema.from_file(schema_path)
        _SCHEMA_PATH_CACHED = schema_s3
        _ENCODER = None

    if _ENCODER is None:
        _ENCODER = Encoder(schema=_SCHEMA)

    config = InfonConfig(schema_path="/tmp/schema.json")

    summaries = []
    for job in jobs:
        doc = job["doc"]
        cassette_id = job["cassette_id"]

        infons, _ = extract_infons([doc], _ENCODER, _SCHEMA, config)

        import fsspec
        cass_path = f"{out_root}/cassettes/{cassette_id}.inf"
        with fsspec.open(cass_path, "wb") as f:
            w = CassetteWriter(f, cassette_id=cassette_id,
                                schema_ref=schema_ref)
            for inf in infons:
                w.add(inf)
            footer = w.close()

        index_paths = build_indexes(footer, f"{out_root}/index")

        summaries.append({
            "cassette_id": cassette_id,
            "cassette_path": cass_path,
            "footer_json": footer.to_json().decode(),
            "index_paths": index_paths,
            "n_infons": footer.n_records,
            "doc_id": doc.get("id", ""),
        })

    return {"summaries": summaries}
