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

import os

import pytest
import torch
import torch.multiprocessing as mp

from diffusers.models._modeling_parallel import ContextParallelConfig
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_processor import (
    AttnProcessor,
)

from ...testing_utils import (
    assert_tensors_close,
    is_attention,
    is_context_parallel,
    require_torch_multi_accelerator,
    torch_device,
)


@is_attention
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
            assert_tensors_close(
                output_before_fusion,
                output_after_fusion,
                atol=self.base_precision,
                rtol=0,
                msg="Output should not change after fusing projections",
            )

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
            assert_tensors_close(
                output_before_fusion,
                output_after_unfusion,
                atol=self.base_precision,
                rtol=0,
                msg="Output should match original after unfusing projections",
            )

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


def _context_parallel_worker(rank, world_size, model_class, init_dict, cp_dict, inputs_dict, result_queue):
    try:
        # Setup distributed environment
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        model = model_class(**init_dict)
        model.to(device)
        model.eval()

        inputs_on_device = {}
        for key, value in inputs_dict.items():
            if isinstance(value, torch.Tensor):
                inputs_on_device[key] = value.to(device)
            else:
                inputs_on_device[key] = value

        cp_config = ContextParallelConfig(**cp_dict)
        model.enable_parallelism(config=cp_config)

        with torch.no_grad():
            output = model(**inputs_on_device)
            if isinstance(output, dict):
                output = output.to_tuple()[0]

        if rank == 0:
            result_queue.put(("success", output.shape))

    except Exception as e:
        if rank == 0:
            result_queue.put(("error", str(e)))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@is_context_parallel
@require_torch_multi_accelerator
class ContextParallelTesterMixin:
    base_precision = 1e-3

    @pytest.mark.parametrize("cp_type", ["ulysses_degree", "ring_degree"], ids=["ulysses", "ring"])
    def test_context_parallel_inference(self, cp_type):
        if not torch.distributed.is_available():
            pytest.skip("torch.distributed is not available.")

        if not hasattr(self.model_class, "_cp_plan") or self.model_class._cp_plan is None:
            pytest.skip("Model does not have a _cp_plan defined for context parallel inference.")

        world_size = 2
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        cp_dict = {cp_type: world_size}

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        mp.spawn(
            _context_parallel_worker,
            args=(world_size, self.model_class, init_dict, cp_dict, inputs_dict, result_queue),
            nprocs=world_size,
            join=True,
        )

        status, result = result_queue.get(timeout=60)
        assert status == "success", f"Context parallel inference failed: {result}"
