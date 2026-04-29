"""Database connection and query execution.

This module demonstrates function calls, state mutations, and return values.
"""

import os
from typing import Optional

from models import ConnectionError, QueryError


class Database:
    """Database connection manager."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.connection = None
        self._is_connected = False

    def connect(self) -> bool:
        """Establish database connection."""
        if self._is_connected:
            return True

        try:
            # Mutate state
            self._is_connected = True
            self.connection = {"host": self.host, "port": self.port}
            return True
        except Exception as e:
            raise ConnectionError(f"Failed to connect: {e}")

    def disconnect(self) -> None:
        """Close database connection."""
        # Mutate state
        self._is_connected = False
        self.connection = None

    def execute_query(self, query: str) -> list:
        """Execute SQL query and return results."""
        if not self._is_connected:
            raise QueryError("Not connected to database")

        # Simulate query execution
        results = []
        return results

    def get_connection_info(self) -> dict:
        """Return connection information."""
        return {
            "host": self.host,
            "port": self.port,
            "connected": self._is_connected,
        }


def create_database(config: dict) -> Database:
    """Factory function to create database instance."""
    host = config.get("host", "localhost")
    port = config.get("port", 5432)
    db = Database(host, port)
    return db


def execute_with_retry(db: Database, query: str, max_retries: int = 3) -> Optional[list]:
    """Execute query with retry logic."""
    for attempt in range(max_retries):
        try:
            result = db.execute_query(query)
            return result
        except QueryError:
            if attempt == max_retries - 1:
                raise
            continue
    return None
