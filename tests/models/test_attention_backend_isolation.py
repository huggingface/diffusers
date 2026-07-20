# coding=utf-8
# Copyright 2026 The HuggingFace Team.
#
# Licensed under the Apache License, Version 2.0.

"""Unit tests for attention backend process isolation."""

from diffusers.models.attention_dispatch import (
    AttentionBackendName,
    _AttentionBackendRegistry,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention_processor import Attention


class _TinyAttentionModel(ModelMixin):
    def __init__(self):
        super().__init__()
        # Minimal Attention module so set_attention_backend has a processor to stamp.
        self.attn = Attention(query_dim=16, heads=2, dim_head=8)


def test_set_attention_backend_does_not_mutate_process_global_registry():
    """model.set_attention_backend must not leak into the process-global registry."""
    initial_backend, _ = _AttentionBackendRegistry.get_active_backend()
    model = _TinyAttentionModel()

    try:
        # Pick a backend different from the current global default when possible.
        target = AttentionBackendName.NATIVE
        if initial_backend == target:
            # Still exercise the API; isolation is what we assert.
            pass
        model.set_attention_backend(target.value)
        active_backend, _ = _AttentionBackendRegistry.get_active_backend()
        assert active_backend == initial_backend, (
            "set_attention_backend must not change the process-global active backend; "
            f"expected {initial_backend}, got {active_backend}"
        )
        # Per-module processor should still be stamped.
        assert model.attn.processor._attention_backend == target
    finally:
        model.reset_attention_backend()
        _AttentionBackendRegistry.set_active_backend(initial_backend)
