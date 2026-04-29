"""Main application entry point.

This module demonstrates all eight relation types working together.
"""

import sys

from auth import (
    AuthenticationError,
    AuthorizationError,
    check_permission,
    login,
    logout,
    register_user,
)
from db import Database, create_database, execute_with_retry
from models import ConnectionError, QueryError, Session, User


def initialize_database() -> Database:
    """Initialize database connection."""
    config = {
        "host": "localhost",
        "port": 5432,
    }
    db = create_database(config)
    db.connect()
    return db


def run_query(db: Database, query: str) -> list | None:
    """Run database query with error handling."""
    try:
        results = execute_with_retry(db, query)
        return results
    except QueryError as e:
        print(f"Query failed: {e}")
        raise
    except ConnectionError as e:
        print(f"Connection failed: {e}")
        raise


def handle_user_registration(email: str, username: str, password: str) -> User | None:
    """Handle new user registration flow."""
    db = initialize_database()

    try:
        user = register_user(email, username, password, db)
        return user
    except AuthenticationError as e:
        print(f"Registration failed: {e}")
        return None
    finally:
        db.disconnect()


def handle_user_login(email: str, password: str) -> Session | None:
    """Handle user login flow."""
    db = initialize_database()

    try:
        session = login(email, password, db)
        return session
    except AuthenticationError as e:
        print(f"Login failed: {e}")
        return None
    finally:
        db.disconnect()


def main() -> int:
    """Main application entry point."""
    # Register a user
    user = handle_user_registration("user@example.com", "testuser", "password123")

    if user:
        # Login
        session = handle_user_login("user@example.com", "password123")

        if session:
            # Check permissions
            try:
                has_access = check_permission(user, "admin_panel")
                print(f"Access granted: {has_access}")
            except AuthorizationError as e:
                print(f"Authorization error: {e}")

            # Logout
            logout(session)

    return 0


if __name__ == "__main__":
    sys.exit(main())
