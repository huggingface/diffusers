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
import socket
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers.models._modeling_parallel import ContextParallelConfig, TensorParallelConfig
from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry

from ...testing_utils import (
    is_attention,
    is_context_parallel,
    is_kernels_available,
    is_tensor_parallel,
    require_torch_multi_accelerator,
    require_torch_tpu,
    torch_device,
)
from .utils import _maybe_cast_to_bf16


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


def _context_parallel_worker(
    rank,
    world_size,
    master_port,
    model_class,
    init_dict,
    cp_dict,
    inputs_dict,
    return_dict,
    attention_backend=None,
    state_dict=None,
):
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
        if state_dict is not None:
            model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Cast as needed.
        model, inputs_dict = _maybe_cast_to_bf16(attention_backend, model, inputs_dict)

        # Move inputs to device
        inputs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_dict.items()}

        # Enable attention backend
        if attention_backend:
            model.set_attention_backend(attention_backend)

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
            if state_dict is not None:
                # Serialise via nested list so the manager dict can transport it across processes.
                return_dict["output"] = output.cpu().tolist()

    except Exception as e:
        if rank == 0:
            return_dict["status"] = "error"
            return_dict["error"] = str(e)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _context_parallel_backward_worker(
    rank, world_size, master_port, model_class, init_dict, cp_dict, inputs_dict, return_dict
):
    """Worker function for context parallel backward pass testing."""
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

        # Create model in training mode
        model = model_class(**init_dict)
        model.to(device)
        model.train()

        # Move inputs to device
        inputs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_dict.items()}

        # Enable context parallelism
        cp_config = ContextParallelConfig(**cp_dict)
        model.enable_parallelism(config=cp_config)

        # Run forward and backward pass
        output = model(**inputs_on_device, return_dict=False)[0]
        loss = output.sum()
        loss.backward()

        # Check that backward actually produced at least one valid gradient
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        has_valid_grads = len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)

        # Only rank 0 reports results
        if rank == 0:
            return_dict["status"] = "success"
            return_dict["has_valid_grads"] = bool(has_valid_grads)

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


def _tensor_parallel_worker(
    rank, world_size, master_port, model_class, init_dict, inputs_dict, return_dict, state_dict
):
    """Worker function for tensor parallel inference testing.

    Each rank builds the (identical, `state_dict`-loaded) model, sets up the accelerator device, shards the model
    with `enable_parallelism(config=TensorParallelConfig(tp_degree=world_size))` and runs a forward pass. Rank 0
    reports its output so the caller can compare it against a single-device reference (TP is mathematically equivalent
    to the unsharded model up to floating-point reduction order).
    """
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        device_config = DEVICE_CONFIG.get(torch_device, DEVICE_CONFIG["cuda"])
        backend = device_config["backend"]
        device_module = device_config["module"]

        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

        device_module.set_device(rank)
        device = torch.device(f"{torch_device}:{rank}")

        model = model_class(**init_dict)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        inputs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_dict.items()}

        # Shard the model across all ranks; the device mesh is built from `tp_degree` on the active accelerator.
        model.enable_parallelism(config=TensorParallelConfig(tp_degree=world_size))

        with torch.no_grad():
            output = model(**inputs_on_device, return_dict=False)[0]

        if rank == 0:
            return_dict["status"] = "success"
            return_dict["output_shape"] = list(output.shape)
            # Serialise via nested list so the manager dict can transport it across processes.
            return_dict["output"] = output.float().cpu().tolist()

    except Exception as e:
        if rank == 0:
            return_dict["status"] = "error"
            return_dict["error"] = str(e)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@is_tensor_parallel
@require_torch_multi_accelerator
class TensorParallelTesterMixin:
    def test_tensor_parallel_inference(self, batch_size: int = 1):
        if not torch.distributed.is_available():
            pytest.skip("torch.distributed is not available.")

        if getattr(self.model_class, "_tp_plan", None) is None:
            pytest.skip("Model does not define a `_tp_plan` for tensor parallel inference.")

        world_size = 2
        init_dict = self.get_init_dict()
        num_heads = init_dict.get("num_attention_heads")
        if num_heads is not None and num_heads % world_size != 0:
            pytest.skip(f"`num_attention_heads` ({num_heads}) is not divisible by tp_degree ({world_size}).")

        inputs_dict = self.get_dummy_inputs(batch_size=batch_size)

        # Single-device reference
        model = self.model_class(**init_dict).eval().to(torch_device)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        with torch.no_grad():
            ref_output = model(**inputs_dict, return_dict=False)[0].float().cpu()

        # Move all tensors to CPU for multiprocessing
        inputs_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs_dict.items()}

        master_port = _find_free_port()
        manager = mp.Manager()
        return_dict = manager.dict()

        mp.spawn(
            _tensor_parallel_worker,
            args=(world_size, master_port, self.model_class, init_dict, inputs_dict, return_dict, state_dict),
            nprocs=world_size,
            join=True,
        )

        assert return_dict.get("status") == "success", (
            f"Tensor parallel inference failed: {return_dict.get('error', 'Unknown error')}"
        )

        tp_output = torch.tensor(return_dict["output"])
        # Sharded matmuls + all-reduce reorder the summation, so allow a small tolerance over the reference.
        torch.testing.assert_close(ref_output, tp_output, atol=1e-3, rtol=1e-3)

    def test_tensor_parallel_batch_inputs(self):
        self.test_tensor_parallel_inference(batch_size=2)


def _run_tp_worker_subprocess(worker_filename: str, spec: str, world_size: int, timeout_s: int = 900) -> None:
    """Launch a `torchrun` TP-correctness worker subprocess and assert it exits cleanly.

    Args:
        worker_filename: Name of the worker script, resolved relative to `tests/models/transformers/` (e.g.
            `"_tpu_tp_worker.py"`).
        spec: `module:function` reference forwarded to the worker, see `_tp_worker_common.run_tp_correctness_worker`.
        world_size: Number of ranks to launch (`torchrun --nproc_per_node`).
        timeout_s: Seconds to wait for the subprocess before failing the test. The worker itself only needs a couple
            of minutes even from a cold compile; this generously bounds it so a real hang (e.g. a distributed-runtime
            barrier timeout) fails the test loudly instead of stalling the run.
    """
    worker = os.path.join(os.path.dirname(__file__), "..", "transformers", worker_filename)
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={world_size}", worker, spec]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            f"TP worker did not finish within {timeout_s}s (likely stuck on a distributed-runtime barrier).\n"
            f"--- stdout ---\n{e.stdout}\n--- stderr ---\n{e.stderr}"
        ) from e
    assert result.returncode == 0, (
        f"TP worker failed (exit {result.returncode}).\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


@is_tensor_parallel
@require_torch_tpu
class TensorParallelTPUTesterMixin:
    """Mixin for a tensor-parallel correctness test on TPU, run via `_tpu_tp_worker.py`.

    TPU TP runs through `torchrun` with the `"tpu_dist"` distributed backend, so — like `TestFlux2TransformerTensorParallelNeuron`
    for Neuron — it cannot use `TensorParallelTesterMixin`'s `torch.multiprocessing.spawn`/NCCL path above and instead
    launches a subprocess worker script and checks its exit code.

    Subclasses set `TP_SPEC` to a `module:function` reference (see
    `_tp_worker_common.run_tp_correctness_worker`'s `spec` argument) and, only if the model spec's head count
    doesn't divide 4, override `WORLD_SIZE`.

    `WORLD_SIZE` defaults to 4 rather than an arbitrary rank count: `torch_tpu`'s per-generation topology table
    (`torch_tpu._internal.utils.hardware`) only enumerates whole-pod-slice chip counts (1/4/8 for v6e, for example),
    not arbitrary sub-slices of a larger single host. A rank count with no matching whole-slice topology has
    nothing to advertise and the PJRT client never completes its start-session barrier — the test would hang for
    the barrier's full multi-minute timeout instead of failing. 4 is the smallest whole-slice count every current
    TPU generation defines (see `_V4_TOPOLOGY` / `_V5E_TOPOLOGY` / `_V6E_TOPOLOGY` / `_V7_TOPOLOGY` in
    `torch_tpu._internal.utils.hardware`). `skip_if_unsupported` below still checks the actual host up front and
    skips fast instead of hanging when it doesn't have exactly that many chips.

    Requires `TORCH_TPU_TOPOLOGY` and `TORCH_TPU_SLICEBUILDER_ADDRESSES` to be set. Source them via::

        eval $(python -m torch_tpu._internal.distributed.launchers.singlehost_wrapper | sed 's/^/export /')
    """

    WORLD_SIZE = 4
    # The worker itself only needs a couple of minutes even from a cold XLA compile; this generously bounds the
    # subprocess so a real hang (e.g. a barrier timeout this skip failed to catch) fails the test loudly instead of
    # stalling the run.
    TIMEOUT_S = 900
    TP_SPEC: str = ""

    def skip_if_unsupported(self):
        """Skip unless the host has exactly `WORLD_SIZE` TPU chips.

        A topology *string* existing for a chip count (`hardware.get_tpu_topology`) isn't enough to guarantee the
        PJRT client can actually form that session: a sub-slice of a larger single host (e.g. claiming 2 of a
        4-chip v6e-4's chips via `TORCH_TPU_TOPOLOGY`/`TORCH_TPU_SLICEBUILDER_ADDRESSES`) can still fail with a
        low-level `START_SESSION` GRPC error, since the runtime's session setup is tied to the host's actual
        provisioned slice, not just a topology label. The only combination verified to work is running with exactly
        as many ranks as the host has chips.
        """
        from torch_tpu._internal.utils import hardware

        try:
            device_count = hardware.get_tpu_device_count()
        except Exception as e:  # pragma: no cover - defensive, hardware detection is best-effort
            pytest.skip(f"Could not determine local TPU chip count: {e}")
            return

        if device_count != self.WORLD_SIZE:
            pytest.skip(
                f"This host exposes {device_count} TPU chip(s), but this test requires exactly "
                f"{self.WORLD_SIZE} (a TPU single-host tensor-parallel job must use all chips on the host; "
                f"sub-slicing a larger host is not reliably supported by the runtime). Run this test on a host "
                f"with exactly {self.WORLD_SIZE} TPU chips."
            )

    def test_tensor_parallel_tpu_inference(self):
        self.skip_if_unsupported()
        _run_tp_worker_subprocess(
            "_tpu_tp_worker.py", self.TP_SPEC, world_size=self.WORLD_SIZE, timeout_s=self.TIMEOUT_S
        )


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

        # Single-GPU reference
        model = self.model_class(**init_dict).eval().to(torch_device)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        with torch.no_grad():
            ref_output = model(**inputs_dict, return_dict=False)[0].cpu()

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
            args=(
                world_size,
                master_port,
                self.model_class,
                init_dict,
                cp_dict,
                inputs_dict,
                return_dict,
                None,
                state_dict,
            ),
            nprocs=world_size,
            join=True,
        )

        assert return_dict.get("status") == "success", (
            f"Context parallel inference failed: {return_dict.get('error', 'Unknown error')}"
        )

        cp_output = torch.tensor(return_dict["output"])
        torch.testing.assert_close(ref_output, cp_output, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("cp_type", ["ulysses_degree", "ring_degree"], ids=["ulysses", "ring"])
    def test_context_parallel_batch_inputs(self, cp_type):
        self.test_context_parallel_inference(cp_type, batch_size=2)

    @pytest.mark.parametrize("cp_type", ["ulysses_degree", "ring_degree"], ids=["ulysses", "ring"])
    def test_context_parallel_backward(self, cp_type, batch_size: int = 1):
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
            _context_parallel_backward_worker,
            args=(world_size, master_port, self.model_class, init_dict, cp_dict, inputs_dict, return_dict),
            nprocs=world_size,
            join=True,
        )

        assert return_dict.get("status") == "success", (
            f"Context parallel backward pass failed: {return_dict.get('error', 'Unknown error')}"
        )
        assert return_dict.get("has_valid_grads"), "Context parallel backward pass did not produce valid gradients."

    @pytest.mark.parametrize("cp_type", ["ulysses_degree", "ring_degree"], ids=["ulysses", "ring"])
    def test_context_parallel_backward_batch_inputs(self, cp_type):
        self.test_context_parallel_backward(cp_type, batch_size=2)

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


@is_attention
@is_context_parallel
@require_torch_multi_accelerator
class ContextParallelAttentionBackendsTesterMixin:
    unsupported_attn_backends: list[str] = []

    @pytest.mark.parametrize("cp_type", ["ulysses_degree", "ring_degree"])
    @pytest.mark.parametrize(
        "attention_backend",
        [
            "native",
            pytest.param(
                "flash_hub",
                marks=pytest.mark.skipif(not is_kernels_available(), reason="`kernels` is not available."),
            ),
            pytest.param(
                "flash_varlen_hub",
                marks=pytest.mark.skipif(not is_kernels_available(), reason="`kernels` is not available."),
            ),
            pytest.param(
                "_flash_3_hub",
                marks=pytest.mark.skipif(not is_kernels_available(), reason="`kernels` is not available."),
            ),
            pytest.param(
                "_flash_3_varlen_hub",
                marks=pytest.mark.skipif(not is_kernels_available(), reason="`kernels` is not available."),
            ),
        ],
    )
    @pytest.mark.parametrize("ulysses_anything", [True, False])
    @torch.no_grad()
    def test_context_parallel_attn_backend_inference(self, cp_type, attention_backend, ulysses_anything):
        if not torch.distributed.is_available():
            pytest.skip("torch.distributed is not available.")

        if getattr(self.model_class, "_cp_plan", None) is None:
            pytest.skip("Model does not have a _cp_plan defined for context parallel inference.")

        if attention_backend in self.unsupported_attn_backends:
            pytest.skip(f"{attention_backend} is not supported for this model.")

        if cp_type == "ring_degree":
            if attention_backend == AttentionBackendName.NATIVE:
                pytest.skip("Skipping test because ring isn't supported with native attention backend.")
            elif attention_backend in ("flash_varlen_hub", "_flash_3_varlen_hub"):
                pytest.skip("`ring_degree` is not yet supported for varlen attention hub kernels.")

        if ulysses_anything and "ulysses" not in cp_type:
            pytest.skip("Skipping test as ulysses anything needs the ulysses degree set.")

        world_size = 2
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()

        # Single-GPU reference with the same attention backend (no context parallel)
        model = self.model_class(**init_dict).eval().to(torch_device)
        if attention_backend:
            model.set_attention_backend(attention_backend)

        # Copy inputs and cast model + inputs as needed
        ref_inputs = inputs_dict.copy()
        model, ref_inputs = _maybe_cast_to_bf16(attention_backend, model, ref_inputs)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        with torch.no_grad():
            ref_output = model(**ref_inputs, return_dict=False)[0].cpu()

        # Move all tensors to CPU for multiprocessing
        inputs_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs_dict.items()}
        cp_dict = {cp_type: world_size}
        if ulysses_anything:
            cp_dict.update({"ulysses_anything": ulysses_anything})

        # Find a free port for distributed communication
        master_port = _find_free_port()

        # Use multiprocessing manager for cross-process communication
        manager = mp.Manager()
        return_dict = manager.dict()

        # Spawn worker processes
        mp.spawn(
            _context_parallel_worker,
            args=(
                world_size,
                master_port,
                self.model_class,
                init_dict,
                cp_dict,
                inputs_dict,
                return_dict,
                attention_backend,
                state_dict,
            ),
            nprocs=world_size,
            join=True,
        )

        assert return_dict.get("status") == "success", (
            f"Context parallel inference failed: {return_dict.get('error', 'Unknown error')}"
        )

        cp_output = torch.tensor(return_dict["output"], dtype=ref_output.dtype)
        torch.testing.assert_close(ref_output, cp_output, atol=1e-2, rtol=1e-2)
