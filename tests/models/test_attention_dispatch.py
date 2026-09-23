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
import torch.nn.functional as F

from diffusers.models._modeling_parallel import ContextParallelConfig, ParallelConfig
from diffusers.models.attention_dispatch import attention_backend as attention_backend_ctx
from diffusers.models.attention_dispatch import dispatch_attention_fn

from ..testing_utils import (
    is_attention,
    is_context_parallel,
    is_kernels_available,
    is_torch_compile,
    require_torch_accelerator,
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


def _ulysses_anything_parity_worker(rank, world_size, master_port, attention_backend):
    device_type = torch_device.split(":")[0]
    if device_type == "cpu":
        backend, device = "cpu:gloo", torch.device("cpu")
    else:
        device_config = DEVICE_CONFIG[device_type]
        backend = device_config["backend"]
        device_config["module"].set_device(rank)
        device = torch.device(f"{device_type}:{rank}")

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = dist.device_mesh.init_device_mesh(device_type, (1, world_size), mesh_dim_names=("ring", "ulysses"))
        config = ContextParallelConfig(ulysses_degree=world_size, ulysses_anything=True)
        config.setup(rank, world_size, device, mesh)
        parallel_config = ParallelConfig(context_parallel_config=config)
        dtype = torch.float32 if attention_backend == "native" else torch.bfloat16
        tolerance = 1e-5 if dtype == torch.float32 else 2e-2

        cases = [
            (8, 8, 4, None, False),
            (9, 9, 4, None, False),
            (9, 7, 4, None, False),
            (9, 7, 7, None, False),
            (9, 9, 4, "bool", True),
            (9, 7, 7, "bool", False),
            (3, 2, 7, None, False),
        ]
        if attention_backend == "native":
            cases.append((9, 9, 7, "additive", True))

        for query_length, key_length, num_heads, mask_type, local_mask in cases:
            torch.manual_seed(777)
            query = torch.randn(2, query_length, num_heads, 64, device=device, dtype=dtype)
            key, value = (torch.randn(2, key_length, num_heads, 64, device=device, dtype=dtype) for _ in range(2))
            grad_out = torch.randn_like(query)
            mask = None
            if mask_type is not None:
                mask = torch.ones(2, 1, 1, key_length, device=device, dtype=torch.bool)
                mask[0, ..., -1] = False
                mask[1, ..., 0] = False
                if mask_type == "additive":
                    mask = torch.zeros_like(mask, dtype=dtype).masked_fill(~mask, -float("inf"))

            query_ref, key_ref, value_ref = (tensor.float().requires_grad_() for tensor in (query, key, value))
            ref_out = F.scaled_dot_product_attention(
                query_ref.transpose(1, 2), key_ref.transpose(1, 2), value_ref.transpose(1, 2), attn_mask=mask
            ).transpose(1, 2)
            ref_grads = torch.autograd.grad(ref_out, (query_ref, key_ref, value_ref), grad_out.float())

            query_local, key_local, value_local = (
                tensor.tensor_split(world_size, dim=1)[rank].detach().clone().requires_grad_()
                for tensor in (query, key, value)
            )
            mask_local = mask.tensor_split(world_size, dim=-1)[rank] if local_mask else mask
            return_lse = attention_backend != "native"
            with attention_backend_ctx(attention_backend):
                out = dispatch_attention_fn(
                    query_local,
                    key_local,
                    value_local,
                    attn_mask=mask_local,
                    attention_kwargs={"return_lse": return_lse},
                    parallel_config=parallel_config,
                )
            if return_lse:
                out, lse = out
                scores = query_ref.transpose(1, 2) @ key_ref.transpose(1, 2).transpose(-1, -2) / query.shape[-1] ** 0.5
                if mask is not None:
                    scores = scores.masked_fill(~mask, -float("inf"))
                expected_lse = scores.logsumexp(dim=-1).transpose(1, 2).tensor_split(world_size, dim=1)[rank]
                torch.testing.assert_close(lse, expected_lse, atol=tolerance, rtol=tolerance)
                assert not lse.requires_grad
            torch.testing.assert_close(
                out.float(), ref_out.tensor_split(world_size, dim=1)[rank], atol=tolerance, rtol=tolerance
            )
            out.backward(grad_out.tensor_split(world_size, dim=1)[rank])
            for tensor, grad_ref in zip((query_local, key_local, value_local), ref_grads):
                torch.testing.assert_close(
                    tensor.grad.float(),
                    grad_ref.tensor_split(world_size, dim=1)[rank],
                    atol=tolerance,
                    rtol=tolerance,
                )

            with torch.no_grad(), attention_backend_ctx(attention_backend):
                inference_out = dispatch_attention_fn(
                    query_local, key_local, value_local, attn_mask=mask_local, parallel_config=parallel_config
                )
            torch.testing.assert_close(inference_out, out)

        query_local, key_local, value_local = (torch.randn(2, 4, 4, 64, device=device, dtype=dtype) for _ in range(3))
        per_head_mask = torch.ones(2, 4, 1, 4 * world_size, device=device, dtype=torch.bool)
        with pytest.raises(ValueError, match="per-head"), attention_backend_ctx(attention_backend):
            dispatch_attention_fn(
                query_local, key_local, value_local, attn_mask=per_head_mask, parallel_config=parallel_config
            )
    finally:
        dist.destroy_process_group()


@is_attention
@is_context_parallel
class TestUlyssesAnythingAttentionBackward:
    @pytest.mark.parametrize("world_size", [2, 4])
    @pytest.mark.parametrize(
        "attention_backend",
        [
            "native",
            pytest.param(
                "_flash_3_varlen_hub",
                marks=pytest.mark.skipif(
                    torch_device != "cuda" or not is_kernels_available(),
                    reason="FlashAttention 3 requires CUDA and kernels.",
                ),
            ),
        ],
    )
    def test_uneven_sequence_and_head_gradients(self, world_size, attention_backend):
        """Uneven sequence/head partitions preserve outputs and all three input gradients."""
        if not dist.is_available():
            pytest.skip("torch.distributed is not available.")
        if torch_device != "cpu" and DEVICE_CONFIG[torch_device.split(":")[0]]["module"].device_count() < world_size:
            pytest.skip(f"Requires {world_size} devices.")
        mp.spawn(
            _ulysses_anything_parity_worker,
            args=(world_size, _find_free_port(), attention_backend),
            nprocs=world_size,
            join=True,
        )


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


@is_attention
class TestVarlenAttentionCompile:
    """Op-level tests for the varlen backends under `torch.compile` with dynamic shapes, model independent."""

    @pytest.mark.parametrize(
        "attention_backend",
        [
            pytest.param(
                "flash_varlen_hub",
                marks=pytest.mark.skipif(not is_kernels_available(), reason="`kernels` is not available."),
            ),
            pytest.param(
                "_flash_3_varlen_hub",
                marks=pytest.mark.skipif(not is_kernels_available(), reason="`kernels` is not available."),
            ),
        ],
    )
    @is_torch_compile
    @require_torch_accelerator
    def test_dynamic_shapes(self, attention_backend):
        def make_qkv(seq_len):
            torch.manual_seed(0)
            return tuple(torch.randn(2, seq_len, 4, 64, device=torch_device, dtype=torch.bfloat16) for _ in range(3))

        with contextlib.ExitStack() as stack, torch.no_grad():
            try:
                stack.enter_context(attention_backend_ctx(attention_backend))
                dispatch_attention_fn(*make_qkv(128))
            except Exception as e:
                pytest.skip(f"Skipping test for backend '{attention_backend}': {e}")

            torch.compiler.reset()
            compiled_attention = torch.compile(dispatch_attention_fn, dynamic=True, fullgraph=True)
            try:
                with (
                    torch._inductor.utils.fresh_inductor_cache(),
                    torch._dynamo.config.patch(error_on_recompile=True),
                ):
                    for seq_len in (128, 256, 256):
                        q, k, v = make_qkv(seq_len)
                        torch.testing.assert_close(
                            compiled_attention(q, k, v), dispatch_attention_fn(q, k, v), atol=1e-3, rtol=1e-3
                        )
            finally:
                torch.compiler.reset()
