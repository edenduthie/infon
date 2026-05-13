"""CloudStore: DynamoDB + S3 backend for serverless cognition.

Single-table design with adjacency list pattern per INFON_SPEC.md section 10.
Requires `pip install cognition[aws]` for boto3.

PK/SK layout:
    INFON#{id}            | META               → infon fields
    ANCHOR#{name}         | INFON#{id}         → role, confidence
    INFON#{id_a}          | NEXT#{id_b}        → anchor, gap_days
    INFON#{id_a}          | ENTAILS#{id_b}     → confidence
    CONSTRAINT#{s}#{p}#{o}| META               → evidence, strength, ...
    CONSTRAINT#{s}#{p}#{o}| INFON#{id}         → confidence
    DOC#{doc_id}          | INFON#{id}         → sent_id

GSI-1: SK as partition key (reverse lookups)
GSI-2: subject + timestamp (temporal queries)
"""

from __future__ import annotations

import json
from typing import Any

from ..infon import Infon, Edge, Constraint


def _require_boto3():
    try:
        import boto3
        return boto3
    except ImportError:
        raise ImportError(
            "CloudStore requires boto3. Install with: pip install cognition[aws]"
        )


class CloudStore:
    """DynamoDB + S3 store backend for AWS deployment."""

    def __init__(self, table_name: str, bucket: str = "",
                 region: str = "us-east-1"):
        self.table_name = table_name
        self.bucket = bucket
        self.region = region
        self._table = None
        self._s3 = None

    @property
    def table(self):
        if self._table is None:
            boto3 = _require_boto3()
            ddb = boto3.resource("dynamodb", region_name=self.region)
            self._table = ddb.Table(self.table_name)
        return self._table

    @property
    def s3(self):
        if self._s3 is None:
            boto3 = _require_boto3()
            self._s3 = boto3.client("s3", region_name=self.region)
        return self._s3

    def init(self) -> None:
        """Create DynamoDB table if it doesn't exist."""
        boto3 = _require_boto3()
        client = boto3.client("dynamodb", region_name=self.region)
        try:
            client.describe_table(TableName=self.table_name)
        except client.exceptions.ResourceNotFoundException:
            client.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {"AttributeName": "PK", "KeyType": "HASH"},
                    {"AttributeName": "SK", "KeyType": "RANGE"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "PK", "AttributeType": "S"},
                    {"AttributeName": "SK", "AttributeType": "S"},
                    {"AttributeName": "subject", "AttributeType": "S"},
                    {"AttributeName": "timestamp", "AttributeType": "S"},
                ],
                GlobalSecondaryIndexes=[
                    {
                        "IndexName": "GSI1",
                        "KeySchema": [
                            {"AttributeName": "SK", "KeyType": "HASH"},
                            {"AttributeName": "PK", "KeyType": "RANGE"},
                        ],
                        "Projection": {"ProjectionType": "ALL"},
                    },
                    {
                        "IndexName": "GSI2-subject-time",
                        "KeySchema": [
                            {"AttributeName": "subject", "KeyType": "HASH"},
                            {"AttributeName": "timestamp", "KeyType": "RANGE"},
                        ],
                        "Projection": {"ProjectionType": "ALL"},
                    },
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            waiter = client.get_waiter("table_exists")
            waiter.wait(TableName=self.table_name)

    # ── Infons ──────────────────────────────────────────────────────────

    def put_infon(self, infon: Infon) -> None:
        item = self._infon_to_item(infon)
        self.table.put_item(Item=item)

        # Anchor index entries
        for role, anchor in [("subject", infon.subject),
                              ("predicate", infon.predicate),
                              ("object", infon.object)]:
            self.table.put_item(Item={
                "PK": f"ANCHOR#{anchor}",
                "SK": f"INFON#{infon.infon_id}",
                "role": role,
                "confidence": str(infon.confidence),
            })

        # Doc index
        self.table.put_item(Item={
            "PK": f"DOC#{infon.doc_id}",
            "SK": f"INFON#{infon.infon_id}",
            "sent_id": infon.sent_id,
        })

    def put_infons(self, infons: list[Infon]) -> None:
        with self.table.batch_writer() as batch:
            for infon in infons:
                batch.put_item(Item=self._infon_to_item(infon))
                for role, anchor in [("subject", infon.subject),
                                      ("predicate", infon.predicate),
                                      ("object", infon.object)]:
                    batch.put_item(Item={
                        "PK": f"ANCHOR#{anchor}",
                        "SK": f"INFON#{infon.infon_id}",
                        "role": role,
                        "confidence": str(infon.confidence),
                    })
                batch.put_item(Item={
                    "PK": f"DOC#{infon.doc_id}",
                    "SK": f"INFON#{infon.infon_id}",
                    "sent_id": infon.sent_id,
                })

    def get_infon(self, infon_id: str) -> Infon | None:
        resp = self.table.get_item(Key={
            "PK": f"INFON#{infon_id}", "SK": "META",
        })
        item = resp.get("Item")
        if not item:
            return None
        return self._item_to_infon(item)

    def query_infons(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        doc_id: str | None = None,
        min_importance: float = 0.0,
        limit: int = 100,
    ) -> list[Infon]:
        from boto3.dynamodb.conditions import Key, Attr

        # Use GSI2 for subject+timestamp queries
        if subject and not predicate and not object:
            resp = self.table.query(
                IndexName="GSI2-subject-time",
                KeyConditionExpression=Key("subject").eq(subject),
                Limit=limit,
                ScanIndexForward=False,
            )
            infons = [self._item_to_infon(i) for i in resp.get("Items", [])]
            if min_importance > 0:
                infons = [inf for inf in infons if inf.importance >= min_importance]
            return infons[:limit]

        # Use anchor index for single-anchor queries
        anchor = subject or predicate or object
        if anchor:
            resp = self.table.query(
                KeyConditionExpression=Key("PK").eq(f"ANCHOR#{anchor}"),
                Limit=limit * 2,
            )
            infon_ids = [
                item["SK"].replace("INFON#", "")
                for item in resp.get("Items", [])
            ]
            infons = []
            for iid in infon_ids[:limit]:
                inf = self.get_infon(iid)
                if inf and inf.importance >= min_importance:
                    infons.append(inf)
            return infons

        # Doc index
        if doc_id:
            resp = self.table.query(
                KeyConditionExpression=Key("PK").eq(f"DOC#{doc_id}"),
                Limit=limit,
            )
            infon_ids = [
                item["SK"].replace("INFON#", "")
                for item in resp.get("Items", [])
            ]
            return [inf for iid in infon_ids
                    if (inf := self.get_infon(iid)) and inf.importance >= min_importance]

        # Fallback: scan (expensive, avoid in production)
        resp = self.table.scan(
            FilterExpression=Attr("SK").eq("META") & Attr("PK").begins_with("INFON#"),
            Limit=limit,
        )
        return [self._item_to_infon(i) for i in resp.get("Items", [])
                if float(i.get("importance", 0)) >= min_importance]

    def get_infons_for_anchor(self, anchor: str, role: str | None = None,
                               limit: int = 100) -> list[Infon]:
        from boto3.dynamodb.conditions import Key, Attr

        kce = Key("PK").eq(f"ANCHOR#{anchor}")
        fe = Attr("role").eq(role) if role else None

        kwargs: dict[str, Any] = {
            "KeyConditionExpression": kce,
            "Limit": limit,
        }
        if fe:
            kwargs["FilterExpression"] = fe

        resp = self.table.query(**kwargs)
        infon_ids = [
            item["SK"].replace("INFON#", "")
            for item in resp.get("Items", [])
        ]
        return [inf for iid in infon_ids if (inf := self.get_infon(iid))]

    def count_infons(self) -> int:
        resp = self.table.scan(
            Select="COUNT",
            FilterExpression=__import__("boto3").dynamodb.conditions.Attr("SK").eq("META")
            & __import__("boto3").dynamodb.conditions.Attr("PK").begins_with("INFON#"),
        )
        return resp.get("Count", 0)

    # ── Edges ───────────────────────────────────────────────────────────

    def put_edge(self, edge: Edge) -> None:
        sk_type = edge.edge_type
        self.table.put_item(Item={
            "PK": f"INFON#{edge.source}",
            "SK": f"{sk_type}#{edge.target}",
            "edge_type": edge.edge_type,
            "weight": str(edge.weight),
            "metadata": json.dumps(edge.metadata) if edge.metadata else "{}",
        })

    def put_edges(self, edges: list[Edge]) -> None:
        with self.table.batch_writer() as batch:
            for edge in edges:
                batch.put_item(Item={
                    "PK": f"INFON#{edge.source}" if not edge.source.startswith("INFON#") else edge.source,
                    "SK": f"{edge.edge_type}#{edge.target}",
                    "edge_type": edge.edge_type,
                    "weight": str(edge.weight),
                    "metadata": json.dumps(edge.metadata) if edge.metadata else "{}",
                })

    def get_edges(self, source: str | None = None, target: str | None = None,
                  edge_type: str | None = None, limit: int = 100) -> list[Edge]:
        from boto3.dynamodb.conditions import Key

        if source:
            kce = Key("PK").eq(f"INFON#{source}")
            if edge_type:
                kce = kce & Key("SK").begins_with(f"{edge_type}#")
            resp = self.table.query(
                KeyConditionExpression=kce, Limit=limit,
            )
        elif target:
            # Reverse lookup via GSI1
            kce = Key("SK").begins_with(f"{edge_type}#{target}" if edge_type else target)
            resp = self.table.query(
                IndexName="GSI1",
                KeyConditionExpression=Key("SK").eq(
                    f"{edge_type}#{target}" if edge_type else target
                ),
                Limit=limit,
            )
        else:
            return []

        result = []
        for item in resp.get("Items", []):
            meta = json.loads(item.get("metadata", "{}"))
            src = item["PK"].replace("INFON#", "")
            tgt = item["SK"].split("#", 1)[1] if "#" in item["SK"] else item["SK"]
            result.append(Edge(
                source=src, target=tgt,
                edge_type=item.get("edge_type", ""),
                weight=float(item.get("weight", 1.0)),
                metadata=meta,
            ))
        return result

    def get_next_chain(self, infon_id: str, anchor: str,
                       limit: int = 50) -> list[Edge]:
        chain = []
        current = infon_id
        seen = set()
        while len(chain) < limit and current not in seen:
            seen.add(current)
            edges = self.get_edges(source=current, edge_type="NEXT", limit=10)
            found = None
            for e in edges:
                if e.metadata.get("anchor") == anchor:
                    found = e
                    break
            if not found:
                break
            chain.append(found)
            current = found.target
        return chain

    # ── Constraints ─────────────────────────────────────────────────────

    def put_constraint(self, constraint: Constraint) -> None:
        key = f"{constraint.subject}#{constraint.predicate}#{constraint.object}"
        self.table.put_item(Item={
            "PK": f"CONSTRAINT#{key}",
            "SK": "META",
            "subject": constraint.subject,
            "predicate": constraint.predicate,
            "object": constraint.object,
            "evidence": constraint.evidence,
            "doc_count": constraint.doc_count,
            "strength": str(constraint.strength),
            "persistence": constraint.persistence,
            "score": str(constraint.score),
            "infon_ids": json.dumps(constraint.infon_ids),
        })

    def get_constraints(self, subject: str | None = None,
                        predicate: str | None = None,
                        object: str | None = None,
                        min_score: float = 0.0,
                        limit: int = 100) -> list[Constraint]:
        from boto3.dynamodb.conditions import Key, Attr

        # If all three specified, direct get
        if subject and predicate and object:
            key = f"{subject}#{predicate}#{object}"
            resp = self.table.get_item(Key={
                "PK": f"CONSTRAINT#{key}", "SK": "META",
            })
            item = resp.get("Item")
            if not item:
                return []
            c = self._item_to_constraint(item)
            return [c] if c.score >= min_score else []

        # Scan constraints (filtered)
        fe = Attr("SK").eq("META") & Attr("PK").begins_with("CONSTRAINT#")
        if subject:
            fe = fe & Attr("subject").eq(subject)
        if predicate:
            fe = fe & Attr("predicate").eq(predicate)
        if object:
            fe = fe & Attr("object").eq(object)

        resp = self.table.scan(FilterExpression=fe, Limit=limit)
        constraints = [self._item_to_constraint(i) for i in resp.get("Items", [])]
        if min_score > 0:
            constraints = [c for c in constraints if c.score >= min_score]
        constraints.sort(key=lambda c: -c.score)
        return constraints[:limit]

    # ── Maintenance ─────────────────────────────────────────────────────

    def prune(self, threshold: float) -> int:
        # DynamoDB doesn't support conditional batch updates well —
        # scan and delete items below threshold
        from boto3.dynamodb.conditions import Attr
        resp = self.table.scan(
            FilterExpression=(
                Attr("SK").eq("META")
                & Attr("PK").begins_with("INFON#")
                & Attr("importance").lt(str(threshold))
            ),
        )
        count = 0
        with self.table.batch_writer() as batch:
            for item in resp.get("Items", []):
                batch.delete_item(Key={"PK": item["PK"], "SK": item["SK"]})
                count += 1
        return count

    def vacuum(self) -> None:
        pass  # DynamoDB manages storage automatically

    # ── Internal helpers ────────────────────────────────────────────────

    def _infon_to_item(self, infon: Infon) -> dict:
        d = infon.to_dict()
        item = {
            "PK": f"INFON#{infon.infon_id}",
            "SK": "META",
        }
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, (dict, list)):
                item[k] = json.dumps(v)
            elif isinstance(v, float):
                item[k] = str(v)
            else:
                item[k] = v
        return item

    def _item_to_infon(self, item: dict) -> Infon:
        d = {}
        for k, v in item.items():
            if k in ("PK", "SK"):
                continue
            if isinstance(v, str):
                # Try to parse JSON fields
                if k in ("spans", "support", "subject_meta", "predicate_meta",
                          "object_meta", "locations", "temporal_refs"):
                    try:
                        d[k] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        d[k] = v
                else:
                    d[k] = v
            else:
                d[k] = v

        # Convert string floats back
        for f in ("confidence", "activation", "coherence", "specificity",
                   "novelty", "importance", "decay_rate"):
            if f in d and isinstance(d[f], str):
                d[f] = float(d[f])
        for f in ("polarity", "reinforcement_count"):
            if f in d and isinstance(d[f], str):
                d[f] = int(d[f])

        return Infon.from_dict(d)

    def _item_to_constraint(self, item: dict) -> Constraint:
        ids = json.loads(item.get("infon_ids", "[]"))
        return Constraint(
            subject=item.get("subject", ""),
            predicate=item.get("predicate", ""),
            object=item.get("object", ""),
            evidence=int(item.get("evidence", 0)),
            doc_count=int(item.get("doc_count", 0)),
            strength=float(item.get("strength", 0)),
            persistence=int(item.get("persistence", 0)),
            score=float(item.get("score", 0)),
            infon_ids=ids,
        )
