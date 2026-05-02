"""CLI entry point — launches the Cognition TUI."""

import sys


def main():
    # The TUI lives at repo root, not inside the package
    # Import inline to avoid heavy deps on `cognition --help`
    try:
        from cognition_tui import CognitionApp
    except ImportError:
        # Fall back: maybe running from repo root
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        try:
            from cognition_tui import CognitionApp
        except ImportError:
            print("cognition_tui.py not found. Run from the ontology-workshop directory.")
            sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(prog="cognition", description="Cognition TUI")
    parser.add_argument("--schema", help="Path to schema JSON file")
    parser.add_argument("--db", default="data/cognition_mc.db", help="SQLite database path")
    args = parser.parse_args()
    CognitionApp(schema_path=args.schema, db_path=args.db).run()
