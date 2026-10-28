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

import os
import socket
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers import ModelMixin
from diffusers.models._modeling_parallel import ContextParallelConfig, ParallelConfig, TensorParallelConfig
from diffusers.utils.distributed_utils import sync_default_rng_seed


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _sync_seed_worker(rank, world_size, master_port, seeds, advances, return_dict):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(seeds[rank])
        torch.randn(advances[rank])
        sync_default_rng_seed(device=torch.device("cpu"))
        assert torch.initial_seed() == seeds[0]
        return_dict[rank] = torch.randn(4)
    finally:
        dist.destroy_process_group()


def _run(seeds, advances):
    world_size = len(seeds)
    return_dict = mp.Manager().dict()
    mp.spawn(
        _sync_seed_worker,
        args=(world_size, _find_free_port(), seeds, advances, return_dict),
        nprocs=world_size,
        join=True,
    )
    return [return_dict[rank] for rank in range(world_size)]


def _expected_draws(seed):
    torch.manual_seed(seed)
    return torch.randn(4)


def test_sync_default_rng_seed_aligns_different_seeds():
    draws = _run(seeds=[0, 1], advances=[1, 2])
    assert torch.equal(draws[0], _expected_draws(0))
    assert torch.equal(draws[1], _expected_draws(0))


def test_sync_default_rng_seed_aligns_shared_seed_at_different_offsets():
    # Same seed, but rank 1 has drawn more random numbers, so the streams are out of step until they are reset.
    draws = _run(seeds=[7, 7], advances=[1, 2])
    assert torch.equal(draws[0], _expected_draws(7))
    assert torch.equal(draws[1], _expected_draws(7))


@pytest.mark.parametrize("seed", [2**63, 2**64 - 1])
def test_sync_default_rng_seed_preserves_unsigned_seed(seed):
    draws = _run(seeds=[seed, 1], advances=[1, 2])
    for draw in draws:
        assert torch.equal(draw, _expected_draws(seed))


def test_sync_default_rng_seed_uses_explicit_device():
    device = torch.device("cuda:1")
    with (
        patch("torch.initial_seed", return_value=2**64 - 1),
        patch("torch.distributed.broadcast_object_list") as broadcast,
        patch("torch.manual_seed") as manual_seed,
    ):
        sync_default_rng_seed(device=device)

    broadcast.assert_called_once_with([2**64 - 1], src=0, group=None, device=device)
    manual_seed.assert_called_once_with(2**64 - 1)


def test_sync_default_rng_seed_uses_subgroup_source():
    group = Mock()
    device = torch.device("cpu")
    with (
        patch("torch.initial_seed", return_value=7),
        patch("torch.distributed.get_global_rank", return_value=2) as get_global_rank,
        patch("torch.distributed.broadcast_object_list") as broadcast,
        patch("torch.manual_seed"),
    ):
        sync_default_rng_seed(group=group, device=device)

    get_global_rank.assert_called_once_with(group, 0)
    broadcast.assert_called_once_with([7], src=2, group=group, device=device)


@pytest.mark.parametrize("parallel_type", ["tp", "cp", "empty"])
def test_resolve_parallel_config_syncs_world_only_for_parallel_models(parallel_type):
    mesh = MagicMock()
    mesh.size.return_value = 2
    mesh.ndim = 2
    mesh.mesh_dim_names = ("ring", "ulysses")
    if parallel_type == "tp":
        config = TensorParallelConfig(mesh=mesh)
    elif parallel_type == "cp":
        config = ContextParallelConfig(ring_degree=2, mesh=mesh)
    else:
        config = ParallelConfig()

    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=1),
        patch("torch.distributed.get_world_size", return_value=4),
        patch("torch._C._get_accelerator", return_value=torch.device("cuda")),
        patch("torch.accelerator.device_count", return_value=4),
        patch("diffusers.models.modeling_utils.sync_default_rng_seed") as sync_seed,
    ):
        resolved = ModelMixin()._resolve_parallel_config(config)

    if parallel_type == "empty":
        sync_seed.assert_not_called()
    else:
        sync_seed.assert_called_once_with(device=torch.device("cuda:1"))
    assert resolved._device == torch.device("cuda:1")
