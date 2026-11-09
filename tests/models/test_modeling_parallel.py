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

import contextlib
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.models._modeling_parallel import (
    ContextParallelConfig,
    ContextParallelInput,
    ContextParallelOutput,
    TensorParallelConfig,
)
from diffusers.utils.torch_utils import randn_tensor

from ..testing_utils import (
    is_context_parallel,
    is_tensor_parallel,
    is_torch_tpu_available,
    require_torch_multi_accelerator,
    torch_device,
)
from .testing_utils.parallelism import DEVICE_CONFIG, _find_free_port


class _ParallelRNGDummyModel(ModelMixin, ConfigMixin):
    """A two-`Linear` MLP: tensor parallelism shards it column- then row-wise, context parallelism splits its sequence."""

    _tp_plan = {"proj_in": "colwise", "proj_out": "rowwise"}
    _cp_plan = {
        "": {"hidden_states": ContextParallelInput(split_dim=1, expected_dims=3)},
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    @register_to_config
    def __init__(self, dim: int = 8):
        super().__init__()
        self.proj_in = torch.nn.Linear(dim, 4 * dim)
        self.proj_out = torch.nn.Linear(4 * dim, dim)

    def forward(self, hidden_states):
        return self.proj_out(torch.nn.functional.gelu(self.proj_in(hidden_states)))


_PARALLEL_RNG_LATENTS_SHAPE = (1, 8, 8)


def _parallel_rng_device_type():
    """The accelerator type to run on. `torch_device` stays "cpu" on a TPU host unless `DIFFUSERS_TEST_DEVICE` is set."""
    device_type = torch.device(torch_device).type
    if device_type == "cpu" and is_torch_tpu_available():
        return "tpu"
    return device_type


def _parallel_rng_device(device_type, rank):
    if device_type == "tpu":
        # Each spawned process is bound to one chip. Avoid bf16 matmuls to keep the tolerance tight.
        torch.set_float32_matmul_precision("highest")
        return torch.device("tpu")
    DEVICE_CONFIG[device_type]["module"].set_device(rank)
    return torch.device(f"{device_type}:{rank}")


@contextlib.contextmanager
def _parallel_rng_spawn_env(device_type, world_size):
    """Prepare the environment `mp.spawn` needs to start `world_size` workers on `device_type`."""
    if device_type != "tpu":
        yield
        return

    from torch_tpu._internal.distributed.launchers.singlehost_wrapper import prepare_tpu_environment

    tpu_env = ("TORCH_TPU_TOPOLOGY", "TORCH_TPU_SLICEBUILDER_ADDRESSES")
    for key in tpu_env:
        os.environ.pop(key, None)
    prepare_tpu_environment(world_size)
    try:
        yield
    finally:
        for key in tpu_env:
            os.environ.pop(key, None)


def _parallel_rng_reference_worker(rank, device_type, checkpoint_dir, return_dict):
    """Single-device run seeded like rank 0 of the parallel run: the latents and output parallelism must reproduce."""
    try:
        device = _parallel_rng_device(device_type, 0)
        model = _ParallelRNGDummyModel.from_pretrained(checkpoint_dir).to(device)

        torch.manual_seed(0)
        latents = randn_tensor(_PARALLEL_RNG_LATENTS_SHAPE, device=device)
        with torch.no_grad():
            output = model(latents)
        return_dict["latents"] = latents.cpu().tolist()
        return_dict["output"] = output.cpu().tolist()
    except Exception as e:
        return_dict["error"] = str(e)


def _parallel_rng_sync_worker(
    rank, world_size, master_port, device_type, parallelism, loading, checkpoint_dir, return_dict
):
    """Seed every rank differently, load the model with `parallelism`, and report the latents each rank then draws."""
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)

        dist.init_process_group(backend=DEVICE_CONFIG[device_type]["backend"], rank=rank, world_size=world_size)
        device = _parallel_rng_device(device_type, rank)

        if parallelism == "tp":
            config = TensorParallelConfig(tp_degree=world_size)
        elif parallelism == "cp":
            config = ContextParallelConfig(ulysses_degree=world_size)
        else:
            # Every rank is its own one-rank TP group, i.e. an independent replica.
            mesh = init_device_mesh(device_type, mesh_shape=(world_size, 1), mesh_dim_names=("dp", "tp"))
            config = TensorParallelConfig(mesh=mesh["tp"])

        torch.manual_seed(rank)
        if loading == "from_pretrained":
            model = _ParallelRNGDummyModel.from_pretrained(checkpoint_dir, parallel_config=config).to(device)
        else:
            model = _ParallelRNGDummyModel.from_pretrained(checkpoint_dir).to(device)
            model.enable_parallelism(config=config)

        latents = randn_tensor(_PARALLEL_RNG_LATENTS_SHAPE, device=device)
        with torch.no_grad():
            output = model(latents)

        gathered_latents = [torch.empty_like(latents) for _ in range(world_size)]
        dist.all_gather(gathered_latents, latents)
        if rank == 0:
            return_dict["same_latents"] = all(torch.equal(latents, other) for other in gathered_latents)
            return_dict["latents"] = latents.cpu().tolist()
            return_dict["output"] = output.cpu().tolist()

    except Exception as e:
        if rank == 0:
            return_dict["error"] = str(e)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@is_tensor_parallel
@is_context_parallel
@require_torch_multi_accelerator(include_neuron=True, include_tpu=True)
class TestParallelRNGSync:
    """Loading with a parallel config gives every rank of the parallel mesh the same default RNG, and only those ranks.

    Rank 0 of the parallel run must also draw the latents, and produce the output, of a non-parallel run with its seed.
    """

    @pytest.mark.parametrize("loading", ["enable_parallelism", "from_pretrained"])
    @pytest.mark.parametrize(
        "parallelism,expect_same",
        [("tp", True), ("cp", True), ("tp_submesh", False)],
        ids=["tensor_parallel", "context_parallel", "tensor_parallel_submesh"],
    )
    def test_parallelism_syncs_rng_within_mesh(self, parallelism, expect_same, loading, tmp_path):
        device_type = _parallel_rng_device_type()
        # Smallest slice every TPU generation supports, as in `TensorParallelTPUTesterMixin`.
        world_size = 4 if device_type == "tpu" else 2
        if device_type == "tpu":
            from torch_tpu._internal.utils import hardware

            if hardware.get_tpu_device_count() < world_size:
                pytest.skip(f"Needs at least {world_size} TPU chips.")

        _ParallelRNGDummyModel().save_pretrained(tmp_path)

        reference = mp.Manager().dict()
        with _parallel_rng_spawn_env(device_type, 1):
            mp.spawn(_parallel_rng_reference_worker, args=(device_type, tmp_path, reference), nprocs=1, join=True)
        assert "error" not in reference, f"RNG reference worker failed: {reference.get('error')}"

        return_dict = mp.Manager().dict()
        with _parallel_rng_spawn_env(device_type, world_size):
            mp.spawn(
                _parallel_rng_sync_worker,
                args=(world_size, _find_free_port(), device_type, parallelism, loading, tmp_path, return_dict),
                nprocs=world_size,
                join=True,
            )

        assert "error" not in return_dict, f"RNG sync worker failed: {return_dict.get('error')}"
        assert return_dict["same_latents"] == expect_same
        assert torch.equal(torch.tensor(return_dict["latents"]), torch.tensor(reference["latents"]))
        torch.testing.assert_close(
            torch.tensor(return_dict["output"]), torch.tensor(reference["output"]), atol=1e-2, rtol=1e-2
        )
