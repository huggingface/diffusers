import gc

import pytest
import torch

from ..testing_utils import backend_empty_cache, torch_device


@pytest.fixture(autouse=True)
def cleanup():
    """Free VRAM and reset the dynamo cache before/after each test."""
    torch.compiler.reset()
    gc.collect()
    backend_empty_cache(torch_device)
    yield
    torch.compiler.reset()
    gc.collect()
    backend_empty_cache(torch_device)
