"""Lambda handler: Export LanceDB tables to Excel workbook on S3.

Reads all five LanceDB tables, enriches with entity names, and writes
a multi-sheet Excel workbook to S3.
"""

import json
import os
import tempfile
from datetime import datetime

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
s3_client = boto3.client("s3", region_name=REGION)


def handler(event, context):
    """Lambda entry point.

    Input:
        lancedb_uri: LanceDB S3 URI
        s3_bucket: Output S3 bucket
        s3_key_prefix: S3 key prefix for the workbook (default: exports/)

    Output:
        workbook_s3_uri: S3 URI of the generated Excel workbook
    """
    import lancedb
    import pandas as pd

    lancedb_uri = event.get("lancedb_uri", os.environ.get(
        "LANCEDB_URI", "s3://sirius-dimefiled-results/hypergraph/"
    ))
    s3_bucket = event.get("s3_bucket", os.environ.get(
        "S3_BUCKET", "sirius-dimefiled-results"
    ))
    s3_key_prefix = event.get("s3_key_prefix", "exports/")

    conn = lancedb.connect(lancedb_uri)

    # Load all tables
    tables_data = {}
    for table_name in ["raw_events", "entities", "relations", "clusters", "hyperedges"]:
        try:
            table = conn.open_table(table_name)
            df = table.to_pandas()
            tables_data[table_name] = df
            print(f"  {table_name}: {len(df)} rows")
        except Exception as e:
            print(f"  {table_name}: not found ({e})")
            tables_data[table_name] = pd.DataFrame()

    # Build entity lookup
    entity_lookup = {}
    if not tables_data["entities"].empty:
        for _, row in tables_data["entities"].iterrows():
            entity_lookup[row["entity_id"]] = row.get(
                "short_name", row.get("canonical_name", "")
            )

    # Enrich relations
    relations_df = tables_data["relations"]
    if not relations_df.empty and entity_lookup:
        relations_df = relations_df.copy()
        relations_df["source_name"] = relations_df["source_entity_id"].map(
            lambda eid: entity_lookup.get(eid, eid)
        )
        relations_df["target_name"] = relations_df["target_entity_id"].map(
            lambda eid: entity_lookup.get(eid, eid)
        )
        tables_data["relations"] = relations_df

    # Enrich hyperedges
    hyperedges_df = tables_data["hyperedges"]
    if not hyperedges_df.empty and entity_lookup:
        hyperedges_df = hyperedges_df.copy()
        hyperedges_df["initiator_names"] = hyperedges_df["initiators"].apply(
            lambda ids: ", ".join(entity_lookup.get(i, i) for i in ids)
            if isinstance(ids, list) else ""
        )
        hyperedges_df["target_names"] = hyperedges_df["targets"].apply(
            lambda ids: ", ".join(entity_lookup.get(i, i) for i in ids)
            if isinstance(ids, list) else ""
        )
        tables_data["hyperedges"] = hyperedges_df

    # Drop vector columns
    for name in ["entities", "hyperedges"]:
        df = tables_data[name]
        if not df.empty and "vector" in df.columns:
            tables_data[name] = df.drop(columns=["vector"])

    # Write to Excel in /tmp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    local_path = os.path.join(tempfile.gettempdir(), f"SIRIUS_Pipeline_Output_{timestamp}.xlsx")

    with pd.ExcelWriter(local_path, engine="openpyxl") as writer:
        summary_data = {
            "Table": list(tables_data.keys()),
            "Row Count": [len(df) for df in tables_data.values()],
        }
        pd.DataFrame(summary_data).to_excel(
            writer, sheet_name="Summary", index=False
        )

        sheet_names = {
            "raw_events": "Raw Events",
            "entities": "Entities",
            "relations": "Relations",
            "clusters": "Clusters",
            "hyperedges": "Hyperedges",
        }
        for name, sheet_name in sheet_names.items():
            df = tables_data[name]
            if not df.empty:
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str[:32000]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        # DIMEFILED_TAXONOMY reference sheet
        try:
            from nuextract_template import (
                L3_NAME_TO_CODE, L4_SUBDOMAINS,
            )
            _L1_NAMES = {
                "D": "DIPLOMATIC", "I": "INFORMATIONAL", "M": "MILITARY",
                "E": "ECONOMIC", "F": "FINANCIAL", "IN": "INTELLIGENCE",
                "L": "LAW_ENFORCEMENT", "EN": "ENVIRONMENTAL",
            }
            _HARDNESS = {
                "D": 1, "I": 2, "M": 6, "E": 3,
                "F": 3, "IN": 5, "L": 4, "EN": 1,
            }
            _l3_c2n = {v: k for k, v in L3_NAME_TO_CODE.items()}
            tax_rows = []
            for entry in L4_SUBDOMAINS:
                parts = entry.split(" - ", 1)
                l4code = parts[0].strip()
                l4name = parts[1].strip() if len(parts) > 1 else ""
                cp = l4code.split(".")
                if len(cp) < 4:
                    continue
                l1c = cp[0]
                l2c = f"{cp[0]}.{cp[1]}"
                l3c = f"{cp[0]}.{cp[1]}.{cp[2]}"
                tax_rows.append({
                    "l1_domain": _L1_NAMES.get(l1c, l1c),
                    "l2_subdomain": l2c,
                    "l3_code": l3c,
                    "l3_name": _l3_c2n.get(l3c, ""),
                    "l4_code": l4code,
                    "l4_name": l4name,
                    "escalation_default": _HARDNESS.get(l1c, 0),
                    "description": f"{_L1_NAMES.get(l1c, l1c)} > "
                                   f"{_l3_c2n.get(l3c, '')} > {l4name}",
                })
            if tax_rows:
                pd.DataFrame(tax_rows).to_excel(
                    writer, sheet_name="DIMEFILED_TAXONOMY", index=False
                )
                print(f"  DIMEFILED_TAXONOMY: {len(tax_rows)} rows")
        except ImportError:
            print("  DIMEFILED_TAXONOMY: skipped (nuextract_template not available)")

    # Upload to S3
    s3_key = f"{s3_key_prefix}SIRIUS_Pipeline_Output_{timestamp}.xlsx"
    s3_client.upload_file(local_path, s3_bucket, s3_key)
    workbook_uri = f"s3://{s3_bucket}/{s3_key}"

    print(f"Exported workbook to {workbook_uri}")

    return {
        "workbook_s3_uri": workbook_uri,
    }
