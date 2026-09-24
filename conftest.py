"""
Pytest configuration file for multispectral dataloader tests.

This file registers custom command-line options and fixtures for pytest.
"""

import os
import pytest

def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--data-dir",
        action="store",
        default=None,
        help="Directory containing multispectral TIFF files for testing"
    )

@pytest.fixture
def data_dir(request):
    """Fixture to provide the data directory path to tests."""
    data_dir = request.config.getoption("--data-dir")
    if data_dir is None:
        pytest.skip("--data-dir not specified")
    if not os.path.exists(data_dir):
        pytest.skip(f"Data directory {data_dir} does not exist")
    return data_dir 