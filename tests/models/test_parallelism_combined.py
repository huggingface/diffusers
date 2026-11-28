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

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers.models._modeling_parallel import ContextParallelConfig, ParallelConfig, TensorParallelConfig


def _check_combined_mesh(rank, world_size, rendezvous, include_tp):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        shape, names = ((2, 1, 2), ("tp", "ring", "ulysses")) if include_tp else ((1, 2), ("ring", "ulysses"))
        mesh = torch.distributed.device_mesh.init_device_mesh("cpu", shape, mesh_dim_names=names)
        config = ParallelConfig(
            context_parallel_config=ContextParallelConfig(ulysses_degree=2),
            tensor_parallel_config=TensorParallelConfig(tp_degree=2),
        )
        if not include_tp:
            with pytest.raises(ValueError, match=r"\['tp'\].*missing"):
                config.setup(rank, world_size, torch.device("cpu"), mesh=mesh)
            return

        config.setup(rank, world_size, torch.device("cpu"), mesh=mesh)
        tp_mesh = config.tensor_parallel_config._mesh
        cp_mesh = config.context_parallel_config._flattened_mesh
        cp_start = rank // 2 * 2
        assert dist.get_process_group_ranks(tp_mesh.get_group()) == [rank % 2, rank % 2 + 2]
        assert dist.get_process_group_ranks(cp_mesh.get_group()) == [cp_start, cp_start + 1]
        assert tp_mesh.get_local_rank() == rank // 2
        assert config.context_parallel_config._ulysses_local_rank == rank % 2
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="test requires Gloo")
class TestCombinedParallelConfig:
    @pytest.mark.parametrize(
        ("include_tp", "world_size"), [(True, 4), (False, 2)], ids=["separate-groups", "missing-tp"]
    )
    def test_combined_mesh(self, tmp_path, include_tp, world_size):
        rendezvous = (tmp_path / "rendezvous").as_uri()
        mp.spawn(_check_combined_mesh, args=(world_size, rendezvous, include_tp), nprocs=world_size)
