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
from diffusers.models.transformers.transformer_flux2 import Flux2KVLayerCache
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVLayerCache

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


class TestBlockCausalAttention:
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("pad_keys", [False, True])
    @pytest.mark.parametrize("has_prefix", [False, True])
    def test_matches_dense_attention(self, batch_size, pad_keys, has_prefix):
        torch.manual_seed(0)
        query, key, value = [torch.randn(batch_size, 12, 2, 16, requires_grad=True) for _ in range(3)]
        segments = [(0, 3, True), (3, 5, False), (5, 7, False), (7, 8, True)] if has_prefix else []
        block_ids = torch.tensor([-1, -1, -1, 0, 0, 1, 1, -1, 2, 2, 2, 2] if has_prefix else [0] * 12)
        positions = torch.arange(12)
        same_image = (block_ids[:, None] == block_ids[None, :]) & (block_ids[:, None] >= 0)
        mask = ((positions[:, None] >= positions[None, :]) | same_image)[None, None]
        key_valid = None
        if pad_keys:
            key_valid = torch.ones(batch_size, 12, dtype=torch.bool)
            key_valid[-1, 1] = False
            mask = mask & key_valid[:, None, None, :]
        expected = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=mask
        ).transpose(1, 2)
        output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None if key_valid is None else key_valid[:, None, None, :],
            block_causal_segments=segments,
        )
        torch.testing.assert_close(output, expected)
        expected_grads = torch.autograd.grad(expected.sum(), (query, key, value))
        grads = torch.autograd.grad(output.sum(), (query, key, value))
        for grad, expected_grad in zip(grads, expected_grads):
            torch.testing.assert_close(grad, expected_grad)


def _context_parallel_kv_cache_worker(rank, world_size, port, ulysses_anything):
    dist.init_process_group("cpu:gloo", init_method=f"tcp://localhost:{port}", rank=rank, world_size=world_size)
    try:
        mesh = dist.device_mesh.init_device_mesh("cpu", (1, world_size), mesh_dim_names=("ring", "ulysses"))
        cp = ContextParallelConfig(ulysses_degree=world_size, ulysses_anything=ulysses_anything)
        cp.setup(rank, world_size, torch.device("cpu"), mesh)
        parallel = ParallelConfig(context_parallel_config=cp)
        torch.manual_seed(4)
        prefix_len = 3 if ulysses_anything else 4
        heads = 7 if ulysses_anything else 4
        q, k, v = [torch.randn(2, 8, heads, 16) for _ in range(3)]

        def shard(tensor):
            return torch.tensor_split(tensor, world_size, dim=1)[rank].contiguous()

        valid = torch.ones(2, 8, dtype=torch.bool)
        valid[-1, 1] = False
        mask = torch.ones(8, 8, dtype=torch.bool).tril()
        mask[prefix_len:] = True
        mask = mask[None, None] & valid[:, None, None]
        ref = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
        ).transpose(1, 2)
        uncached = dispatch_attention_fn(shard(q), shard(k), shard(v), attn_mask=mask, parallel_config=parallel)
        torch.testing.assert_close(uncached, shard(ref))
        train_inputs = [shard(tensor).requires_grad_() for tensor in (q, k, v)]
        output = dispatch_attention_fn(
            *train_inputs,
            attn_mask=valid[:, None, None],
            block_causal_segments=[(0, prefix_len, True)],
            parallel_config=parallel,
        )
        if ulysses_anything:
            with pytest.raises(NotImplementedError, match="Backward pass for Ulysses Anything"):
                output.sum().backward()
        else:
            output.sum().backward()
            ref_inputs = [tensor.detach().requires_grad_() for tensor in (q, k, v)]
            ref_output = F.scaled_dot_product_attention(
                *(tensor.transpose(1, 2) for tensor in ref_inputs), attn_mask=mask
            )
            ref_output.sum().backward()
            for tensor, reference in zip(train_inputs, ref_inputs):
                torch.testing.assert_close(tensor.grad, shard(reference.grad))
        for cache_class in (QwenImage21KVLayerCache, Flux2KVLayerCache):
            cache = cache_class()
            out = dispatch_attention_fn(
                shard(q),
                shard(k),
                shard(v),
                attn_mask=valid[:, None, None],
                parallel_config=parallel,
                kv_cache=cache,
                kv_cache_mode="extract",
                cache_write_slice=slice(0, prefix_len),
                block_causal_segments=[(0, prefix_len, True)],
            )
            torch.testing.assert_close(out, shard(ref))
            for cached, original in zip(cache.get(), (k, v)):
                expected = torch.tensor_split(original[:, :prefix_len], world_size, dim=2)[rank]
                torch.testing.assert_close(cached, expected)
                assert cached.untyped_storage().nbytes() == cached.numel() * cached.element_size()
            qd, kd, vd = [torch.randn(2, 8 - prefix_len, heads, 16) for _ in range(3)]
            fullk, fullv = torch.cat([k[:, :prefix_len], kd], dim=1), torch.cat([v[:, :prefix_len], vd], dim=1)
            decode_ref = F.scaled_dot_product_attention(
                qd.transpose(1, 2), fullk.transpose(1, 2), fullv.transpose(1, 2), attn_mask=valid[:, None, None]
            ).transpose(1, 2)
            out = dispatch_attention_fn(
                shard(qd),
                shard(kd),
                shard(vd),
                attn_mask=valid[:, None, None],
                parallel_config=parallel,
                kv_cache=cache,
                kv_cache_mode="cached",
            )
            torch.testing.assert_close(out, shard(decode_ref))
    finally:
        dist.destroy_process_group()


@is_context_parallel
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
class TestContextParallelKVCache:
    @pytest.mark.parametrize("world_size", [2, 4])
    @pytest.mark.parametrize("ulysses_anything", [False, True])
    def test_cached_attention_matches_full_attention(self, world_size, ulysses_anything):
        mp.spawn(
            _context_parallel_kv_cache_worker,
            args=(world_size, _find_free_port(), ulysses_anything),
            nprocs=world_size,
            join=True,
        )


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
