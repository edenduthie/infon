"""Shared modules for the Sirius pipeline."""

import os
import sys

# Ensure the sirius/ root is on sys.path so steps can import `from shared import db`
_sirius_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sirius_root not in sys.path:
    sys.path.insert(0, _sirius_root)
