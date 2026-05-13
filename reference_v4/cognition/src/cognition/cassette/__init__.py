"""Cassette: immutable, range-addressable infon files for S3-style stores.

Format:
    [MAGIC 8B "INFC0001"]
    [HEADER_LEN u32 LE][HEADER JSON]
    [ RECORD: [LEN u32 LE][GZIP(JSON infon)] ]*
    [FOOTER_LEN u32 LE][FOOTER JSON]
    [FOOTER_OFFSET u64 LE][MAGIC 8B]

Readers bootstrap by range-fetching the last 16 bytes.
"""

from .format import (
    MAGIC, CASSETTE_VERSION,
    CassetteWriter, CassetteReader,
    RecordLoc, Footer,
)
from .index import (
    build_indexes, Manifest,
    query_triple, query_time_range, query_anchor,
)
from .reader import RangeFetcher, LocalFetcher, FsspecFetcher, hydrate_locs
from .dsl import (
    Query, run_any, run_all,
    first_seen, last_seen, timeline, count_by,
    trajectory_hits, next_edges, constraint,
    NextEdge, Constraint,
)
from .reason import reason, reason_many, Verdict
from .reason_path import (
    reason_connectivity, reason_any_target, chain_mass,
    infer_connective_predicates,
)
from .executor import Executor, SyncExecutor, ProcessExecutor, Result
from .store import InfonStore
from .diagnostic import ExtractionReport, compute_report
from .findings import Finding
from .migrate import (
    SchemaFunctor, MigrationReport,
    plan_migration, migrate_many, migrate_store,
)


def _lazy_analyst(*a, **k):
    """Deferred import so strands-agents stays optional at package load."""
    from .analyst import Analyst
    return Analyst(*a, **k)


Analyst = _lazy_analyst


def _lazy_lambda_executor(*a, **k):
    """Deferred import so boto3 stays optional."""
    from .executor_lambda import LambdaExecutor
    return LambdaExecutor(*a, **k)


LambdaExecutor = _lazy_lambda_executor

__all__ = [
    "MAGIC", "CASSETTE_VERSION",
    "CassetteWriter", "CassetteReader",
    "RecordLoc", "Footer",
    "build_indexes", "Manifest",
    "query_triple", "query_time_range", "query_anchor",
    "RangeFetcher", "LocalFetcher", "FsspecFetcher", "hydrate_locs",
    "Query", "run_any", "run_all",
    "first_seen", "last_seen", "timeline", "count_by",
    "trajectory_hits", "next_edges", "constraint",
    "NextEdge", "Constraint",
    "reason", "reason_many", "Verdict",
    "reason_connectivity", "reason_any_target", "chain_mass",
    "infer_connective_predicates",
    "Executor", "SyncExecutor", "ProcessExecutor", "Result",
    "LambdaExecutor",
    "InfonStore",
    "ExtractionReport", "compute_report",
    "Finding",
    "SchemaFunctor", "MigrationReport",
    "plan_migration", "migrate_many", "migrate_store",
    "Analyst",
]
