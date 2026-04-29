"""
Main entry point for running the MCP server as a module.

Usage:
    python -m infon.mcp.server --db /path/to/kb.ddb
"""

from infon.mcp.server import main

if __name__ == "__main__":
    main()
