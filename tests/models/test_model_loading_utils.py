# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh

from diffusers.models._modeling_parallel import TensorParallelConfig
from diffusers.models.model_loading_utils import _load_shard_file_tp


@pytest.fixture
def cpu_tp_config(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("This loader test requires Gloo.")
    dist.init_process_group("gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
        config = TensorParallelConfig(mesh=mesh)
        config.setup(rank=0, world_size=1, device=torch.device("cpu"), mesh=mesh)
        yield config
    finally:
        dist.destroy_process_group()


class TestTensorParallelReplicatedLoading:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_preserves_requested_and_fp32_dtypes(self, tmp_path, cpu_tp_config, dtype):
        with torch.device("meta"):
            model = torch.nn.ModuleDict(
                {
                    "time_embedder": torch.nn.Linear(4, 4, dtype=torch.bfloat16),
                    "projection": torch.nn.Linear(4, 4, dtype=torch.bfloat16),
                }
            )
        model.register_buffer("indexes", torch.empty(4, device="meta", dtype=torch.int64))
        generator = torch.Generator().manual_seed(0)
        weights = {
            name: torch.randn(parameter.shape, generator=generator) for name, parameter in model.named_parameters()
        }
        weights["indexes"] = torch.arange(4)
        checkpoint = tmp_path / "weights.safetensors"
        save_file(weights, checkpoint)
        _load_shard_file_tp(
            str(checkpoint),
            model,
            model.state_dict(),
            {},
            cpu_tp_config,
            dtype=dtype,
            keep_in_fp32_modules=["time_embedder"],
            unexpected_keys=[],
        )
        for name, parameter in model.named_parameters():
            expected_dtype = torch.float32 if name.startswith("time_embedder.") else dtype
            assert parameter.dtype == expected_dtype
            torch.testing.assert_close(parameter, weights[name].to(expected_dtype), atol=0, rtol=0)
        torch.testing.assert_close(model.indexes, weights["indexes"], atol=0, rtol=0)
