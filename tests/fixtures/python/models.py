"""Data models for synthetic test application.

This module demonstrates class inheritance, decorators, and exception raising.
"""


class DatabaseError(Exception):
    """Base exception for database errors."""

    pass


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""

    pass


class QueryError(DatabaseError):
    """Raised when query execution fails."""

    pass


def validate_email(func):
    """Decorator to validate email format."""

    def wrapper(*args, **kwargs):
        # Simple email validation
        return func(*args, **kwargs)

    return wrapper


def audit_log(func):
    """Decorator to log function calls for auditing."""

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result

    return wrapper


class BaseModel:
    """Base class for all data models."""

    def __init__(self):
        self._id = None
        self._created_at = None

    def save(self):
        """Save the model to database."""
        if not self._id:
            raise QueryError("Cannot save model without ID")
        return True


class User(BaseModel):
    """User model extending BaseModel."""

    def __init__(self, email, username):
        super().__init__()
        self.email = email
        self.username = username
        self.is_active = True

    @validate_email
    def set_email(self, email):
        """Set user email with validation."""
        self.email = email
        return self.email

    @audit_log
    def deactivate(self):
        """Deactivate user account."""
        self.is_active = False
        return True


class Session(BaseModel):
    """Session model for authentication tracking."""

    def __init__(self, user_id, token):
        super().__init__()
        self.user_id = user_id
        self.token = token
        self.is_valid = True

    def invalidate(self):
        """Invalidate session."""
        self.is_valid = False
        return self.is_valid
