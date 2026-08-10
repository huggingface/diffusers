import pytest
import torch


def backend_memory_allocated(device: str) -> int:
    """
    Bytes currently allocated on `device`. `tests/testing_utils.py` only exposes the *peak* allocation, which cannot
    show memory being released. Skips on backends that do not implement `memory_allocated()` (e.g. mps).
    """
    device_module = getattr(torch, torch.device(device).type)
    if not hasattr(device_module, "memory_allocated"):
        pytest.skip(f"`memory_allocated()` is not implemented for {device}.")
    return device_module.memory_allocated()
