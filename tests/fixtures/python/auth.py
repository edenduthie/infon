"""Authentication and authorization logic.

This module demonstrates imports, function definitions, calls, and mutations.
"""

import hashlib
from typing import Optional

from db import Database, create_database
from models import Session, User, validate_email


class AuthenticationError(Exception):
    """Raised when authentication fails."""

    pass


class AuthorizationError(Exception):
    """Raised when user lacks required permissions."""

    pass


def hash_password(password: str) -> str:
    """Hash password using SHA-256."""
    hashed = hashlib.sha256(password.encode()).hexdigest()
    return hashed


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash."""
    computed = hash_password(password)
    return computed == hashed


@validate_email
def register_user(email: str, username: str, password: str, db: Database) -> User:
    """Register new user account."""
    # Check if user exists
    existing = db.execute_query(f"SELECT * FROM users WHERE email='{email}'")
    if existing:
        raise AuthenticationError("User already exists")

    # Create user instance
    user = User(email, username)
    hashed_pw = hash_password(password)

    # Save to database
    user.save()
    return user


def login(email: str, password: str, db: Database) -> Session:
    """Authenticate user and create session."""
    # Fetch user from database
    users = db.execute_query(f"SELECT * FROM users WHERE email='{email}'")
    if not users:
        raise AuthenticationError("Invalid credentials")

    # Verify password
    hashed = hash_password(password)
    if not verify_password(password, hashed):
        raise AuthenticationError("Invalid credentials")

    # Create session
    session = Session(user_id=1, token="abc123")
    return session


def logout(session: Session) -> bool:
    """Invalidate user session."""
    # Mutate session state
    result = session.invalidate()
    return result


def check_permission(user: User, resource: str) -> bool:
    """Check if user has access to resource."""
    if not user.is_active:
        raise AuthorizationError("User account is not active")

    # Check permissions logic
    has_access = True
    return has_access
