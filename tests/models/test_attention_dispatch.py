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
import torch.nn.functional as F

from diffusers.models._modeling_parallel import ContextParallelConfig, ParallelConfig
from diffusers.models.attention_dispatch import attention_backend as attention_backend_ctx
from diffusers.models.attention_dispatch import dispatch_attention_fn

from ..testing_utils import (
    is_attention,
    is_context_parallel,
    is_kernels_available,
    require_torch_multi_accelerator,
    torch_device,
)
from .testing_utils.parallelism import DEVICE_CONFIG, _find_free_port


# Max allowed relative error between the context parallel gradients and the single-process reference.
GRAD_RTOL = 2e-2


def _attention_backward_parity_worker(rank, world_size, master_port, cp_dict, attention_backend, return_dict):
    """Op-level worker: check `dispatch_attention_fn` gradients against a single-process reference.

    This guards the ring-attention backward pass, which historically produced silently wrong
    gradients (see https://github.com/huggingface/diffusers/issues/14265) because it recomputed
    every ring iteration against the iteration-0 KV chunk.
    """
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        device_config = DEVICE_CONFIG.get(torch_device, DEVICE_CONFIG["cuda"])
        device_type = torch_device.split(":")[0]
        device_module = device_config["module"]

        dist.init_process_group(backend=device_config["backend"], rank=rank, world_size=world_size)
        device_module.set_device(rank)
        device = torch.device(f"{device_type}:{rank}")

        # Identical inputs on every rank so each rank can compute the same full-sequence reference.
        B, S, H, D = 1, 128, 4, 64
        torch.manual_seed(777)
        q = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
        grad_out = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)

        # Reference: full-sequence fp32 SDPA gradients on a single process.
        q_ref, k_ref, v_ref = (t.float().requires_grad_(True) for t in (q, k, v))
        ref_out = F.scaled_dot_product_attention(
            q_ref.transpose(1, 2), k_ref.transpose(1, 2), v_ref.transpose(1, 2)
        ).transpose(1, 2)
        ref_dq, ref_dk, ref_dv = torch.autograd.grad(ref_out, (q_ref, k_ref, v_ref), grad_out.float())

        mesh = dist.device_mesh.init_device_mesh(
            device_type,
            (cp_dict.get("ring_degree", 1), cp_dict.get("ulysses_degree", 1)),
            mesh_dim_names=("ring", "ulysses"),
        )
        cp_config = ContextParallelConfig(**cp_dict)
        cp_config.setup(rank, world_size, device, mesh)
        parallel_config = ParallelConfig(context_parallel_config=cp_config)

        # Each rank runs its sequence shard through the templated CP attention path.
        shard = slice(rank * S // world_size, (rank + 1) * S // world_size)
        qs, ks, vs = (t.detach()[:, shard].clone().requires_grad_(True) for t in (q, k, v))
        with attention_backend_ctx(attention_backend):
            out = dispatch_attention_fn(qs, ks, vs, parallel_config=parallel_config)
        out.backward(grad_out[:, shard])

        rel_errs = {}
        for name, got, ref in (("dq", qs.grad, ref_dq), ("dk", ks.grad, ref_dk), ("dv", vs.grad, ref_dv)):
            ref_shard = ref[:, shard].to(got.dtype)
            rel_errs[name] = ((got - ref_shard).norm() / ref_shard.norm()).item()

        if rank == 0:
            return_dict["status"] = "success"
            return_dict["rel_errs"] = dict(rel_errs)

    except Exception as e:
        if rank == 0:
            return_dict["status"] = "error"
            return_dict["error"] = repr(e)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@is_attention
@is_context_parallel
@require_torch_multi_accelerator
class TestContextParallelAttentionBackward:
    """Op-level tests for the context parallel backward of `dispatch_attention_fn`, model independent."""

    @pytest.mark.parametrize("cp_type", ["ulysses_degree", "ring_degree"])
    @pytest.mark.parametrize(
        "attention_backend",
        [
            "_native_flash",
            "_native_cudnn",
            pytest.param(
                "flash_hub",
                marks=pytest.mark.skipif(not is_kernels_available(), reason="`kernels` is not available."),
            ),
            pytest.param(
                "_flash_3_hub",
                marks=pytest.mark.skipif(not is_kernels_available(), reason="`kernels` is not available."),
            ),
        ],
    )
    def test_attn_backend_backward_parity(self, cp_type, attention_backend):
        """Ring and Ulysses attention gradients must match a single-GPU reference.

        Regression test for https://github.com/huggingface/diffusers/issues/14265, where the ring
        backward silently used the iteration-0 KV chunk for every ring iteration.
        """
        if not torch.distributed.is_available():
            pytest.skip("torch.distributed is not available.")

        world_size = 2
        cp_dict = {cp_type: world_size}
        master_port = _find_free_port()

        manager = mp.Manager()
        return_dict = manager.dict()
        mp.spawn(
            _attention_backward_parity_worker,
            args=(world_size, master_port, cp_dict, attention_backend, return_dict),
            nprocs=world_size,
            join=True,
        )

        assert return_dict.get("status") == "success", (
            f"Context parallel backward parity run failed: {return_dict.get('error', 'Unknown error')}"
        )
        rel_errs = return_dict["rel_errs"]
        for name, rel in rel_errs.items():
            assert rel < GRAD_RTOL, (
                f"{attention_backend} {cp_type} gradient `{name}` rel_err={rel:.3e} "
                f"exceeds tol={GRAD_RTOL:.1e}: {rel_errs}"
            )
