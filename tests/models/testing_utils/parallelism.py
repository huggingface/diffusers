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
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers.models._modeling_parallel import ContextParallelConfig
from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry

from ...testing_utils import (
    is_context_parallel,
    require_torch_multi_accelerator,
    torch_device,
)


# Device configuration mapping
DEVICE_CONFIG = {
    "cuda": {"backend": "nccl", "module": torch.cuda},
    "xpu": {"backend": "xccl", "module": torch.xpu},
}


def _find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def _context_parallel_worker(rank, world_size, master_port, model_class, init_dict, cp_dict, inputs_dict, return_dict):
    """Worker function for context parallel testing."""
    try:
        # Set up distributed environment
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Get device configuration
        device_config = DEVICE_CONFIG.get(torch_device, DEVICE_CONFIG["cuda"])
        backend = device_config["backend"]
        device_module = device_config["module"]

        # Initialize process group
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

        # Set device for this process
        device_module.set_device(rank)
        device = torch.device(f"{torch_device}:{rank}")

        # Create model
        model = model_class(**init_dict)
        model.to(device)
        model.eval()

        # Move inputs to device
        inputs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_dict.items()}

        # Enable context parallelism
        cp_config = ContextParallelConfig(**cp_dict)
        model.enable_parallelism(config=cp_config)

        # Run forward pass
        with torch.no_grad():
            output = model(**inputs_on_device, return_dict=False)[0]

        # Only rank 0 reports results
        if rank == 0:
            return_dict["status"] = "success"
            return_dict["output_shape"] = list(output.shape)

    except Exception as e:
        if rank == 0:
            return_dict["status"] = "error"
            return_dict["error"] = str(e)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _custom_mesh_worker(
    rank,
    world_size,
    master_port,
    model_class,
    init_dict,
    cp_dict,
    mesh_shape,
    mesh_dim_names,
    inputs_dict,
    return_dict,
):
    """Worker function for context parallel testing with a user-provided custom DeviceMesh."""
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Get device configuration
        device_config = DEVICE_CONFIG.get(torch_device, DEVICE_CONFIG["cuda"])
        backend = device_config["backend"]
        device_module = device_config["module"]

        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

        # Set device for this process
        device_module.set_device(rank)
        device = torch.device(f"{torch_device}:{rank}")

        model = model_class(**init_dict)
        model.to(device)
        model.eval()

        inputs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_dict.items()}

        # DeviceMesh must be created after init_process_group, inside each worker process.
        mesh = torch.distributed.device_mesh.init_device_mesh(
            torch_device, mesh_shape=mesh_shape, mesh_dim_names=mesh_dim_names
        )
        cp_config = ContextParallelConfig(**cp_dict, mesh=mesh)
        model.enable_parallelism(config=cp_config)

        with torch.no_grad():
            output = model(**inputs_on_device, return_dict=False)[0]

        if rank == 0:
            return_dict["status"] = "success"
            return_dict["output_shape"] = list(output.shape)

    except Exception as e:
        if rank == 0:
            return_dict["status"] = "error"
            return_dict["error"] = str(e)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@is_context_parallel
@require_torch_multi_accelerator
class ContextParallelTesterMixin:
    @pytest.mark.parametrize("cp_type", ["ulysses_degree", "ring_degree"], ids=["ulysses", "ring"])
    def test_context_parallel_inference(self, cp_type, batch_size: int = 1):
        if not torch.distributed.is_available():
            pytest.skip("torch.distributed is not available.")

        if not hasattr(self.model_class, "_cp_plan") or self.model_class._cp_plan is None:
            pytest.skip("Model does not have a _cp_plan defined for context parallel inference.")

        if cp_type == "ring_degree":
            active_backend, _ = _AttentionBackendRegistry.get_active_backend()
            if active_backend == AttentionBackendName.NATIVE:
                pytest.skip("Ring attention is not supported with the native attention backend.")

        world_size = 2
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs(batch_size=batch_size)

        # Move all tensors to CPU for multiprocessing
        inputs_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs_dict.items()}
        cp_dict = {cp_type: world_size}

        # Find a free port for distributed communication
        master_port = _find_free_port()

        # Use multiprocessing manager for cross-process communication
        manager = mp.Manager()
        return_dict = manager.dict()

        # Spawn worker processes
        mp.spawn(
            _context_parallel_worker,
            args=(world_size, master_port, self.model_class, init_dict, cp_dict, inputs_dict, return_dict),
            nprocs=world_size,
            join=True,
        )

        assert return_dict.get("status") == "success", (
            f"Context parallel inference failed: {return_dict.get('error', 'Unknown error')}"
        )

    @pytest.mark.parametrize("cp_type", ["ulysses_degree", "ring_degree"], ids=["ulysses", "ring"])
    def test_context_parallel_batch_inputs(self, cp_type):
        self.test_context_parallel_inference(cp_type, batch_size=2)

    @pytest.mark.parametrize(
        "cp_type,mesh_shape,mesh_dim_names",
        [
            ("ring_degree", (2, 1, 1), ("ring", "ulysses", "fsdp")),
            ("ulysses_degree", (1, 2, 1), ("ring", "ulysses", "fsdp")),
        ],
        ids=["ring-3d-fsdp", "ulysses-3d-fsdp"],
    )
    def test_context_parallel_custom_mesh(self, cp_type, mesh_shape, mesh_dim_names):
        if not torch.distributed.is_available():
            pytest.skip("torch.distributed is not available.")

        if not hasattr(self.model_class, "_cp_plan") or self.model_class._cp_plan is None:
            pytest.skip("Model does not have a _cp_plan defined for context parallel inference.")

        if cp_type == "ring_degree":
            active_backend, _ = _AttentionBackendRegistry.get_active_backend()
            if active_backend == AttentionBackendName.NATIVE:
                pytest.skip("Ring attention is not supported with the native attention backend.")

        world_size = 2
        init_dict = self.get_init_dict()
        inputs_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.get_dummy_inputs().items()}
        cp_dict = {cp_type: world_size}

        master_port = _find_free_port()
        manager = mp.Manager()
        return_dict = manager.dict()

        mp.spawn(
            _custom_mesh_worker,
            args=(
                world_size,
                master_port,
                self.model_class,
                init_dict,
                cp_dict,
                mesh_shape,
                mesh_dim_names,
                inputs_dict,
                return_dict,
            ),
            nprocs=world_size,
            join=True,
        )

        assert return_dict.get("status") == "success", (
            f"Custom mesh context parallel inference failed: {return_dict.get('error', 'Unknown error')}"
        )
