import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers.models.attention_dispatch import TemplatedUnifiedAttention
import os

def run(rank, world_size):
    dist.init_process_group(
        backend="gloo", 
        rank=rank,
        world_size=world_size
    )

    torch.manual_seed(0)

    B, S, H, D = 2, 8, 4, 16  # small toy
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)

    q.requires_grad_(True)

    from diffusers.models._modeling_parallel import (
        ParallelConfig, 
        ContextParallelConfig
    )

    pc = ParallelConfig(
        context_parallel_config=ContextParallelConfig(
            ring_degree=2,
            ulysses_degree=2,
        )
    )

    pc.context_parallel_config.setup(
        rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
        mesh=dist.device_mesh.init_device_mesh("cpu",
            (2,2),
            mesh_dim_names=["ring", "ulysses"],
        )
    )

    def dummy_forward_op(
        ctx,
        q,
        k,
        v,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
        return_lse,
        *,
        _save_ctx=True,
        _parallel_config=None,
    ):
        head_scale = math.sqrt(D)
        attn = (q @ k.transpose(-1, -2)) / head_scale
        out = attn @ v
        lse = torch.logsumexp(attn, dim=-1)

        if _save_ctx:
            ctx.save_for_backward(q, k, v)
            ctx._cached_qkv = []
            ctx._cached_iter = 0

        if not hasattr(ctx, "_cached_qkv"):
            ctx._cached_qkv = []

        ctx._cached_qkv.append((q.detach(), k.detach(), v.detach()))

        return (out, lse) if return_lse else out

    def dummy_backward_op(ctx, grad_out, *args, **kwargs):
        if not hasattr(ctx, "_cached_qkv"):
            raise RuntimeError("No cached tensors for backward.")

        if not hasattr(ctx, "_cached_iter"):
            ctx._cached_iter = 0

        if ctx._cached_iter >= len(ctx._cached_qkv):
            raise RuntimeError("Backward called more times than cached forwards.")

        q, k, v = ctx._cached_qkv[ctx._cached_iter]
        ctx._cached_iter += 1

        head_scale = math.sqrt(D)
        attn = (q @ k.transpose(-1, -2)) / head_scale

        grad_v = attn.transpose(-1, -2) @ grad_out
        grad_attn = grad_out @ v.transpose(-1, -2)
        grad_q = (grad_attn @ k) / head_scale
        grad_k = (grad_attn.transpose(-1, -2) @ q) / head_scale

        return (
            grad_q,
            grad_k,
            grad_v,
        )

    attn = TemplatedUnifiedAttention()

    out = attn(
        None,
        q, k, v, None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        return_lse=False,
        forward_op=dummy_forward_op,
        backward_op=dummy_backward_op,
        _parallel_config=pc,
    )

    print(f"[RANK {rank}] output:", out.shape)

    out.sum().backward()
    print(f"[RANK {rank}] grad:", q.grad.shape)

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    mp.spawn(run, args=(world_size,), nprocs=world_size)