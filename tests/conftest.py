"""
conftest.py — pytest shared fixtures and configuration.

Adds project root to sys.path so all test imports work from the tests/ directory.
"""

import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
