"""Shared test fixtures."""
import os
import sys
from unittest.mock import patch

import pytest

# Ensure imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def skip_model_loading():
    """Prevent model loading during tests that don't need it."""
    with patch("api.app.load_models"):
        yield
