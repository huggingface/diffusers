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

"""Tests for combining context parallelism with tensor parallelism on one device mesh.

These cover the wiring rather than the numerics: that `ParallelConfig` hands each parallelism its own submesh, and
that the resulting process groups are the intended factorisation of the world. They run on CPU over `gloo`, so they
need no accelerator — `gloo` has no `all_to_all`, so an end-to-end Ulysses forward pass cannot run here; that is
covered by `ContextAndTensorParallelTesterMixin` in `testing_utils/parallelism.py`, which needs four accelerators.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers.models._modeling_parallel import ContextParallelConfig, ParallelConfig, TensorParallelConfig


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def _mesh_factorization_worker(rank, world_size, master_port, ulysses_degree, tp_degree, return_dict):
    """Build a combined mesh, run `ParallelConfig.setup`, and report the process groups each parallelism landed on."""
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cpu", mesh_shape=(1, ulysses_degree, tp_degree), mesh_dim_names=("ring", "ulysses", "tp")
        )
        config = ParallelConfig(
            context_parallel_config=ContextParallelConfig(ulysses_degree=ulysses_degree),
            tensor_parallel_config=TensorParallelConfig(tp_degree=tp_degree),
        )
        config.setup(rank, world_size, torch.device("cpu"), mesh=mesh)

        cp_config = config.context_parallel_config
        tp_config = config.tensor_parallel_config
        return_dict[rank] = {
            "status": "success",
            "cp_ranks": dist.get_process_group_ranks(cp_config._flattened_mesh.get_group()),
            "ulysses_local_rank": cp_config._ulysses_local_rank,
            "tp_ranks": dist.get_process_group_ranks(tp_config._mesh.get_group()),
            "tp_local_rank": tp_config._mesh.get_local_rank(),
            "tp_degree": tp_config._tp_degree,
        }
    except Exception as e:  # noqa: BLE001 — surfaced as a test failure by the caller
        return_dict[rank] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _missing_tp_dim_worker(rank, world_size, master_port, return_dict):
    """A combined config handed a CP-only mesh must be rejected, naming the missing 'tp' dimension."""
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cpu", mesh_shape=(1, world_size), mesh_dim_names=("ring", "ulysses")
        )
        config = ParallelConfig(
            context_parallel_config=ContextParallelConfig(ulysses_degree=world_size),
            tensor_parallel_config=TensorParallelConfig(tp_degree=1),
        )
        try:
            config.setup(rank, world_size, torch.device("cpu"), mesh=mesh)
        except ValueError as e:
            return_dict[rank] = {"status": "raised", "message": str(e)}
        else:
            return_dict[rank] = {"status": "no_raise"}
    except Exception as e:  # noqa: BLE001
        return_dict[rank] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn(worker, world_size, *args):
    if not dist.is_available():
        pytest.skip("torch.distributed is not available.")
    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(worker, args=(world_size, _find_free_port(), *args, return_dict), nprocs=world_size, join=True)
    return return_dict


class TestCombinedParallelConfig:
    """`ParallelConfig` accepting both parallelisms at once, and splitting the mesh between them."""

    def test_both_configs_are_accepted(self):
        # Combining the two used to raise outright; the mesh dimension per parallelism is what makes it work.
        config = ParallelConfig(
            context_parallel_config=ContextParallelConfig(ulysses_degree=2),
            tensor_parallel_config=TensorParallelConfig(tp_degree=2),
        )
        assert config._is_combined
        assert config.context_parallel_config.ulysses_degree == 2
        assert config.tensor_parallel_config.tp_degree == 2

    def test_single_parallelism_is_not_combined(self):
        assert not ParallelConfig(context_parallel_config=ContextParallelConfig(ulysses_degree=2))._is_combined
        assert not ParallelConfig(tensor_parallel_config=TensorParallelConfig(tp_degree=2))._is_combined

    def test_mesh_is_factored_between_the_two(self):
        """Each parallelism must end up on its own process group: TP within a CP chunk, CP across the TP groups.

        With `mesh_shape=(ring=1, ulysses=2, tp=2)` over four ranks, "tp" is the fastest-varying dimension, so ranks
        {0,1} and {2,3} are the TP groups and {0,2} / {1,3} are the CP groups. A TP all-reduce must not reach a rank
        holding a different sequence chunk, and a CP collective must not reach a rank holding a different weight
        shard — this asserts exactly that split.
        """
        world_size = 4
        results = _spawn(_mesh_factorization_worker, world_size, 2, 2)

        for rank in range(world_size):
            assert results[rank]["status"] == "success", results[rank].get("error")

        assert [results[r]["tp_ranks"] for r in range(4)] == [[0, 1], [0, 1], [2, 3], [2, 3]]
        assert [results[r]["cp_ranks"] for r in range(4)] == [[0, 2], [1, 3], [0, 2], [1, 3]]
        assert [results[r]["ulysses_local_rank"] for r in range(4)] == [0, 0, 1, 1]
        # The TP shard index is the coordinate inside the TP group, which stops tracking the global rank as soon as
        # the mesh has more than one dimension. Weight sharding keys off this, so it is the value that matters.
        assert [results[r]["tp_local_rank"] for r in range(4)] == [0, 1, 0, 1]
        assert all(results[r]["tp_degree"] == 2 for r in range(4))

    def test_mesh_without_tp_dimension_is_rejected(self):
        world_size = 2
        results = _spawn(_missing_tp_dim_worker, world_size)

        for rank in range(world_size):
            assert results[rank]["status"] == "raised", (
                f"rank {rank}: expected a ValueError for a mesh with no 'tp' dimension, got {results[rank]}"
            )
            assert "tp" in results[rank]["message"]
