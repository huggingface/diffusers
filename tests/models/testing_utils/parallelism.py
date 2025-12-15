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

from ...testing_utils import (
    is_context_parallel,
    require_torch_multi_accelerator,
)


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
            output = model(**inputs_on_device, return_dict=False)[0]

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
