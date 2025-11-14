# coding=utf-8
# Copyright 2025 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_processor import (
    AttnProcessor,
)

from ...testing_utils import is_attention, require_accelerator, torch_device


@is_attention
@require_accelerator
class AttentionTesterMixin:
    """
    Mixin class for testing attention processor and module functionality on models.

    Tests functionality from AttentionModuleMixin including:
        - Attention processor management (set/get)
        - QKV projection fusion/unfusion
        - Attention backends (XFormers, NPU, etc.)

    Expected class attributes to be set by subclasses:
        - model_class: The model class to test
        - base_precision: Tolerance for floating point comparisons (default: 1e-3)
        - uses_custom_attn_processor: Whether model uses custom attention processors (default: False)

    Expected methods to be implemented by subclasses:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Pytest mark: attention
        Use `pytest -m "not attention"` to skip these tests
    """

    base_precision = 1e-3

    def test_fuse_unfuse_qkv_projections(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        if not hasattr(model, "fuse_qkv_projections"):
            pytest.skip("Model does not support QKV projection fusion.")

        # Get output before fusion
        with torch.no_grad():
            output_before_fusion = model(**inputs_dict)
            if isinstance(output_before_fusion, dict):
                output_before_fusion = output_before_fusion.to_tuple()[0]

        # Fuse projections
        model.fuse_qkv_projections()

        # Verify fusion occurred by checking for fused attributes
        has_fused_projections = False
        for module in model.modules():
            if isinstance(module, AttentionModuleMixin):
                if hasattr(module, "to_qkv") or hasattr(module, "to_kv"):
                    has_fused_projections = True
                    assert module.fused_projections, "fused_projections flag should be True"
                    break

        if has_fused_projections:
            # Get output after fusion
            with torch.no_grad():
                output_after_fusion = model(**inputs_dict)
                if isinstance(output_after_fusion, dict):
                    output_after_fusion = output_after_fusion.to_tuple()[0]

            # Verify outputs match
            assert torch.allclose(
                output_before_fusion, output_after_fusion, atol=self.base_precision
            ), "Output should not change after fusing projections"

            # Unfuse projections
            model.unfuse_qkv_projections()

            # Verify unfusion occurred
            for module in model.modules():
                if isinstance(module, AttentionModuleMixin):
                    assert not hasattr(module, "to_qkv"), "to_qkv should be removed after unfusing"
                    assert not hasattr(module, "to_kv"), "to_kv should be removed after unfusing"
                    assert not module.fused_projections, "fused_projections flag should be False"

            # Get output after unfusion
            with torch.no_grad():
                output_after_unfusion = model(**inputs_dict)
                if isinstance(output_after_unfusion, dict):
                    output_after_unfusion = output_after_unfusion.to_tuple()[0]

            # Verify outputs still match
            assert torch.allclose(
                output_before_fusion, output_after_unfusion, atol=self.base_precision
            ), "Output should match original after unfusing projections"

    def test_get_set_processor(self):
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict)
        model.to(torch_device)

        # Check if model has attention processors
        if not hasattr(model, "attn_processors"):
            pytest.skip("Model does not have attention processors.")

        # Test getting processors
        processors = model.attn_processors
        assert isinstance(processors, dict), "attn_processors should return a dict"
        assert len(processors) > 0, "Model should have at least one attention processor"

        # Test that all processors can be retrieved via get_processor
        for module in model.modules():
            if isinstance(module, AttentionModuleMixin):
                processor = module.get_processor()
                assert processor is not None, "get_processor should return a processor"

                # Test setting a new processor
                new_processor = AttnProcessor()
                module.set_processor(new_processor)
                retrieved_processor = module.get_processor()
                assert retrieved_processor is new_processor, "Retrieved processor should be the same as the one set"

    def test_attention_processor_dict(self):
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict)
        model.to(torch_device)

        if not hasattr(model, "set_attn_processor"):
            pytest.skip("Model does not support setting attention processors.")

        # Get current processors
        current_processors = model.attn_processors

        # Create a dict of new processors
        new_processors = {key: AttnProcessor() for key in current_processors.keys()}

        # Set processors using dict
        model.set_attn_processor(new_processors)

        # Verify all processors were set
        updated_processors = model.attn_processors
        for key in current_processors.keys():
            assert type(updated_processors[key]) == AttnProcessor, f"Processor {key} should be AttnProcessor"

    def test_attention_processor_count_mismatch_raises_error(self):
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict)
        model.to(torch_device)

        if not hasattr(model, "set_attn_processor"):
            pytest.skip("Model does not support setting attention processors.")

        # Get current processors
        current_processors = model.attn_processors

        # Create a dict with wrong number of processors
        wrong_processors = {list(current_processors.keys())[0]: AttnProcessor()}

        # Verify error is raised
        with pytest.raises(ValueError) as exc_info:
            model.set_attn_processor(wrong_processors)

        assert "number of processors" in str(exc_info.value).lower(), "Error should mention processor count mismatch"
