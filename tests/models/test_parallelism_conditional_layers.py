# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers.models._modeling_parallel import ContextParallelConfig, TensorParallelConfig
from diffusers.utils import logging

from ..testing_utils import (
    CaptureLogger,
    is_context_parallel,
    is_tensor_parallel,
    require_torch_multi_accelerator,
    torch_device,
)
from .testing_utils.parallelism import DEVICE_CONFIG, DummyConditionalPlanModel, _find_free_port


def _conditional_plan_worker(rank, world_size, master_port, checkpoint_dir, parallelism, inputs_dict, return_dict):
    """Load `DummyConditionalPlanModel` with a `parallel_config` and run a forward pass.

    Rank 0 reports its output, what the hooks logged while loading, and where the plan was applied: the parameters
    sharded into `DTensor`s for tensor parallelism, or the modules carrying a hook for context parallelism.
    """
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        device_config = DEVICE_CONFIG.get(torch_device, DEVICE_CONFIG["cuda"])
        dist.init_process_group(backend=device_config["backend"], rank=rank, world_size=world_size)
        device_config["module"].set_device(rank)
        device = torch.device(f"{torch_device}:{rank}")

        from torch.distributed.tensor import DTensor

        if parallelism == "tensor":
            parallel_config = TensorParallelConfig(tp_degree=world_size)
        else:
            parallel_config = ContextParallelConfig(ulysses_degree=world_size)

        # A skipped `Conditional` entry is logged at debug level.
        hooks_logger = logging.get_logger("diffusers.hooks")
        hooks_logger.setLevel(logging.DEBUG)
        with CaptureLogger(hooks_logger) as cap_logger:
            model = DummyConditionalPlanModel.from_pretrained(checkpoint_dir, parallel_config=parallel_config)
        if parallelism == "context":
            # A tensor-parallel load already places each rank's shard on its device.
            model.to(device)

        inputs_on_device = {k: v.to(device) for k, v in inputs_dict.items()}
        with torch.no_grad():
            output = model(**inputs_on_device)

        if rank == 0:
            if parallelism == "tensor":
                applied = [name for name, param in model.state_dict().items() if isinstance(param, DTensor)]
            else:
                applied = [name for name, module in model.named_modules() if hasattr(module, "_diffusers_hook")]

            return_dict["status"] = "success"
            return_dict["output"] = output.cpu().tolist()
            return_dict["logs"] = cap_logger.out
            return_dict["applied"] = applied

    except Exception as e:
        if rank == 0:
            return_dict["status"] = "error"
            return_dict["error"] = f"{type(e).__name__}: {e}"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@require_torch_multi_accelerator
class TestConditionalPlanEntries:
    """`Conditional` entries of a `_tp_plan` / `_cp_plan`, on a checkpoint with and without the conditional layers."""

    unmatched_message = "match no module of 'DummyConditionalPlanModel'"

    def _save_and_load_with_parallelism(self, tmp_path, parallelism, use_conditional_layers):
        """Save the model, load it on two ranks with parallelism, and check the output against a single device."""
        if not torch.distributed.is_available():
            pytest.skip("torch.distributed is not available.")

        torch.manual_seed(0)
        model = DummyConditionalPlanModel(use_conditional_layers=use_conditional_layers).eval()
        inputs_dict = {"hidden_states": torch.randn(1, 8, 8), "position_ids": torch.arange(8)}
        with torch.no_grad():
            reference = model(**inputs_dict)
        model.save_pretrained(tmp_path)

        world_size = 2
        return_dict = mp.Manager().dict()
        mp.spawn(
            _conditional_plan_worker,
            args=(world_size, _find_free_port(), str(tmp_path), parallelism, inputs_dict, return_dict),
            nprocs=world_size,
            join=True,
        )

        assert return_dict.get("status") == "success", (
            f"Loading with {parallelism} parallelism failed: {return_dict.get('error', 'Unknown error')}"
        )
        torch.testing.assert_close(reference, torch.tensor(return_dict["output"]), atol=1e-4, rtol=1e-4)
        return return_dict

    @is_tensor_parallel
    @pytest.mark.parametrize("use_conditional_layers", [True, False])
    def test_tensor_parallel(self, tmp_path, use_conditional_layers):
        result = self._save_and_load_with_parallelism(tmp_path, "tensor", use_conditional_layers)

        if use_conditional_layers:
            gate_params = {f"blocks.{i}.gate_proj.{name}" for i in range(2) for name in ("weight", "bias")}
            assert gate_params <= set(result["applied"])
            assert self.unmatched_message not in result["logs"]
        else:
            assert f"{self.unmatched_message}: ['blocks.*.gate_proj']" in result["logs"]

    @is_context_parallel
    @pytest.mark.parametrize("use_conditional_layers", [True, False])
    def test_context_parallel(self, tmp_path, use_conditional_layers):
        result = self._save_and_load_with_parallelism(tmp_path, "context", use_conditional_layers)

        if use_conditional_layers:
            assert "pos_embed" in result["applied"]
            assert self.unmatched_message not in result["logs"]
        else:
            assert f"{self.unmatched_message}: ['pos_embed']" in result["logs"]
