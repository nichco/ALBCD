"""pytest configuration for qn/: puts qn/ on sys.path and registers the "slow" marker (pyproject.toml untouched)."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: tests that solve the jax examples (minutes)")
