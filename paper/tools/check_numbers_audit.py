"""
check_numbers_audit.py

Reads paper/numbers_audit.md and verifies that every number cited in the paper
matches the corresponding value in the referenced JSON file.

Table format (markdown):
    | value | section | location | json_path | json_key | tolerance | description |

- json_path is relative to the repository root.
- value may be a percentage (e.g., "87.0") while the JSON stores a decimal (0.87).
  The checker tries direct comparison first, then percentage (value/100) comparison.
- Exits 0 if all rows pass OR if the audit table has no data rows.
- Exits non-zero on any mismatch, printing the offending row plus delta.

Usage:
    python paper/tools/check_numbers_audit.py                       # from repo root
    python tools/check_numbers_audit.py                             # from paper/ directory
    python paper/tools/check_numbers_audit.py --repo-root /path/to  # explicit root
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _find_repo_root_auto() -> Path:
    """
    Find the repository root by:
    1. Walking up from cwd until we find a directory containing 'paper/'
    2. Falling back to walking up from this script's location
    """
    # Try cwd first (works when invoked from the repo root or a subdirectory)
    candidate = Path.cwd()
    while candidate != candidate.parent:
        if (candidate / "paper").is_dir():
            return candidate
        candidate = candidate.parent

    # Fallback: walk up from this script's location
    candidate = Path(__file__).resolve().parent
    while candidate != candidate.parent:
        if (candidate / "paper").is_dir():
            return candidate
        candidate = candidate.parent

    raise FileNotFoundError(
        "Cannot find repository root (no parent directory contains 'paper/')"
    )


def _find_audit_file(repo_root: Path) -> Path:
    return repo_root / "paper" / "numbers_audit.md"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _is_separator_row(row: str) -> bool:
    """Return True for markdown table separator rows like |---|---|...|."""
    return all(c in "-| " for c in row.strip())


def _parse_audit_table(text: str) -> list[dict]:
    """
    Parse a markdown table with header row:
        | value | section | location | json_path | json_key | tolerance | description |

    Returns a list of dicts (one per data row). Returns empty list if no data rows.
    """
    EXPECTED_HEADERS = ["value", "section", "location", "json_path", "json_key", "tolerance", "description"]
    rows = []
    header_found = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if _is_separator_row(stripped):
            continue

        cells = [c.strip() for c in stripped.strip("|").split("|")]

        if not header_found:
            # Normalise header names for comparison
            lower_cells = [c.lower() for c in cells]
            if lower_cells[:len(EXPECTED_HEADERS)] == EXPECTED_HEADERS:
                header_found = True
            continue  # skip header row itself

        if len(cells) < len(EXPECTED_HEADERS):
            continue  # malformed row, skip

        rows.append({
            "value":       cells[0],
            "section":     cells[1],
            "location":    cells[2],
            "json_path":   cells[3],
            "json_key":    cells[4],
            "tolerance":   cells[5],
            "description": cells[6] if len(cells) > 6 else "",
        })

    return rows


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------

class NumbersAuditError(Exception):
    """Raised when a number in the audit table does not match the JSON source."""


def _lookup_json_value(json_path: Path, json_key: str) -> float:
    """Load JSON file and retrieve the value at json_key (dot-separated path)."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with json_path.open() as fh:
        data = json.load(fh)

    # Support dot-separated key paths: "results.accuracy"
    parts = json_key.split(".")
    node = data
    for part in parts:
        if isinstance(node, dict):
            if part not in node:
                raise KeyError(f"Key '{part}' not found in {json_path} (full key: '{json_key}')")
            node = node[part]
        else:
            raise KeyError(
                f"Cannot traverse key '{part}' — node is not a dict (full key: '{json_key}')"
            )
    return float(node)


def _check_row(row: dict, repo_root: Path) -> None:
    """
    Verify one audit row. Raises NumbersAuditError on mismatch.

    Comparison strategy:
      1. Try direct comparison: abs(value - json_value) <= tolerance
      2. If that fails and value looks like a percentage (>1.0 typically),
         try percentage comparison: abs(value/100 - json_value) <= tolerance/100
    """
    try:
        stated_value = float(row["value"])
        tolerance = float(row["tolerance"])
    except ValueError as exc:
        raise NumbersAuditError(f"Cannot parse value/tolerance in row {row}: {exc}") from exc

    json_path = repo_root / row["json_path"]
    json_value = _lookup_json_value(json_path, row["json_key"])

    # Strategy 1: direct comparison
    direct_delta = abs(stated_value - json_value)
    if direct_delta <= tolerance:
        return

    # Strategy 2: percentage comparison (stated value is %, JSON is 0-1 decimal)
    pct_delta = abs(stated_value / 100.0 - json_value)
    if pct_delta <= tolerance / 100.0:
        return

    # Both strategies failed — report the direct delta (more intuitive)
    raise NumbersAuditError(
        f"MISMATCH in numbers_audit.md\n"
        f"  Row:          {row}\n"
        f"  Stated value: {stated_value}\n"
        f"  JSON value:   {json_value}  (from {row['json_path']} key '{row['json_key']}')\n"
        f"  delta:        {direct_delta:.6g}  (tolerance: {tolerance})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Check in-paper numbers against JSON sources.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to the repository root (default: auto-detect from cwd or script location)",
    )
    args = parser.parse_args()

    repo_root: Path
    if args.repo_root is not None:
        repo_root = args.repo_root.resolve()
    else:
        repo_root = _find_repo_root_auto()

    audit_file = _find_audit_file(repo_root)

    if not audit_file.exists():
        print(f"WARNING: numbers_audit.md not found at {audit_file} — skipping audit")
        return 0

    text = audit_file.read_text(encoding="utf-8")
    rows = _parse_audit_table(text)

    if not rows:
        print("numbers_audit.md: no data rows found — audit passes vacuously.")
        return 0

    errors: list[str] = []
    for row in rows:
        try:
            _check_row(row, repo_root)
        except (NumbersAuditError, FileNotFoundError, KeyError, ValueError) as exc:
            errors.append(str(exc))

    if errors:
        print(f"numbers_audit: {len(errors)} check(s) FAILED:\n", file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)
            print(file=sys.stderr)
        return 1

    print(f"numbers_audit: all {len(rows)} row(s) PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
