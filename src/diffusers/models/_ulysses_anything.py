# Copyright 2025 The HuggingFace Team. All rights reserved.
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

# Adapted from: https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/attention/_templated_ulysses.py
import copy
import functools
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc
import torch.nn.functional as F

from ..utils.torch_utils import maybe_allow_in_graph
from ._modeling_parallel import ParallelConfig


def _wait_tensor(tensor) -> torch.Tensor:
    if isinstance(tensor, fc.AsyncCollectiveTensor):
        tensor = tensor.wait()

    return tensor


def _get_rank_world_size(
    group: dist.ProcessGroup,
) -> Tuple[int, int]:
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    return rank, world_size


@functools.lru_cache(maxsize=128)
def _gather_size_by_comm(size: int, group: dist.ProcessGroup) -> List[int]:
    r"""Gather the local size from all ranks.
    size: int, local size return: List[int], list of size from all ranks
    """
    world_size = dist.get_world_size(group=group)
    # HACK: Use Gloo backend for all_gather to avoid H2D and D2H overhead
    comm_backends = str(dist.get_backend(group=group))
    # NOTE: e.g., dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    gather_device = "cpu" if "cpu" in comm_backends else "cuda"
    gathered_sizes = [torch.empty((1,), device=gather_device, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(
        gathered_sizes,
        torch.tensor([size], device=gather_device, dtype=torch.int64),
        group=group,
    )

    gathered_sizes = [s[0].item() for s in gathered_sizes]
    # NOTE: DON'T use tolist here due to graph break - Explanation:
    # Backend compiler `inductor` failed with aten._local_scalar_dense.default
    return gathered_sizes


# Helper functions to pad/unpad head dimension for QKV and O projections
def _maybe_pad_qkv_head(
    x: torch.Tensor,
    H: int,
    group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, int]:
    r"""Maybe pad the head dimension to be divisible by world_size.
    x: torch.Tensor, shape (B, S_LOCAL, H, D) H: int, original global head num return: Tuple[torch.Tensor, int], padded
    tensor (B, S_LOCAL, H + H_PAD, D) and H_PAD
    """
    _, world_size = _get_rank_world_size(group)
    H_PAD = 0
    if H % world_size != 0:
        H_PAD = world_size - (H % world_size)
        NEW_H_LOCAL = (H + H_PAD) // world_size
        # e.g., Allow: H=30, world_size=8 -> NEW_H_LOCAL=4, H_PAD=2.
        # NOT ALLOW: H=30, world_size=16 -> NEW_H_LOCAL=2, H_PAD=14.
        assert H_PAD < NEW_H_LOCAL, f"Padding head num {H_PAD} should be less than new local head num {NEW_H_LOCAL}"
        x = F.pad(x, (0, 0, 0, H_PAD)).contiguous()
    return x, H_PAD


def _maybe_unpad_qkv_head(
    x: torch.Tensor,
    H_PAD: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    r"""Maybe unpad the head dimension.
    x: torch.Tensor, shape (B, S_GLOBAL, H_LOCAL + H_PAD, D) H_PAD: int, head padding num return: torch.Tensor,
    unpadded tensor (B, S_GLOBAL, H_LOCAL, D)
    """
    rank, world_size = _get_rank_world_size(group)
    # Only the last rank may have padding
    if H_PAD > 0 and rank == world_size - 1:
        x = x[:, :, :-H_PAD, :]
    return x.contiguous()


def _maybe_pad_o_head(
    x: torch.Tensor,
    H: int,
    group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, int]:
    r"""Maybe pad the head dimension to be divisible by world_size.
    x: torch.Tensor, shape (B, S_GLOBAL, H_LOCAL, D) H: int, original global head num return: Tuple[torch.Tensor, int],
    padded tensor (B, S_GLOBAL, H_LOCAL + H_PAD, D) and H_PAD
    """
    if H is None:
        return x, 0

    rank, world_size = _get_rank_world_size(group)
    H_PAD = 0
    # Only the last rank may need padding
    if H % world_size != 0:
        # We need to broadcast H_PAD to all ranks to keep consistency
        # in unpadding step later for all ranks.
        H_PAD = world_size - (H % world_size)
        NEW_H_LOCAL = (H + H_PAD) // world_size
        assert H_PAD < NEW_H_LOCAL, f"Padding head num {H_PAD} should be less than new local head num {NEW_H_LOCAL}"
        if rank == world_size - 1:
            x = F.pad(x, (0, 0, 0, H_PAD)).contiguous()
    return x, H_PAD


def _maybe_unpad_o_head(
    x: torch.Tensor,
    H_PAD: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    r"""Maybe unpad the head dimension.
    x: torch.Tensor, shape (B, S_LOCAL, H_GLOBAL + H_PAD, D) H_PAD: int, head padding num return: torch.Tensor,
    unpadded tensor (B, S_LOCAL, H_GLOBAL, D)
    """
    if H_PAD > 0:
        x = x[:, :, :-H_PAD, :]
    return x.contiguous()


# Helper functions to for all-to-all communication with Ulysses Anything Attention
def _comm_metadata(
    query: torch.Tensor,
    **kwargs,
) -> dict:
    num_qo_head = query.shape[2]  # (B, S_LOCAL, H_GLOBAL, D)
    extra_kwargs = {}
    extra_kwargs["num_qo_head"] = num_qo_head
    # May ddd other kwargs if needed in future
    return extra_kwargs


@maybe_allow_in_graph
def _all_to_all_single_any_qkv_async(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    **kwargs,
) -> Callable[..., torch.Tensor]:
    r"""
    x: torch.Tensor, shape (B, S_LOCAL, H, D) return: Callable that returns (B, S_GLOBAL, H_LOCAL, D)
    """
    _, world_size = _get_rank_world_size(group)
    B, S_LOCAL, H, D = x.shape
    x, H_PAD = _maybe_pad_qkv_head(x, H, group)
    H_LOCAL = (H + H_PAD) // world_size
    # (world_size, S_LOCAL, B, H_LOCAL, D)
    x = x.reshape(B, S_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()

    input_split_sizes = [S_LOCAL] * world_size
    # S_LOCAL maybe not equal for all ranks in dynamic shape case,
    # since we don't know the actual shape before this timing, thus,
    # we have to use all gather to collect the S_LOCAL first.
    output_split_sizes = _gather_size_by_comm(S_LOCAL, group)
    x = x.flatten(0, 1)  # (world_size * S_LOCAL, B, H_LOCAL, D)
    x = fc.all_to_all_single(x, output_split_sizes, input_split_sizes, group)

    def wait() -> torch.Tensor:
        nonlocal x, H_PAD
        x = _wait_tensor(x)  # (S_GLOBAL, B, H_LOCAL, D)
        # (S_GLOBAL, B, H_LOCAL, D)
        # -> (B, S_GLOBAL, H_LOCAL, D)
        x = x.permute(1, 0, 2, 3).contiguous()
        x = _maybe_unpad_qkv_head(x, H_PAD, group)
        return x

    return wait


@maybe_allow_in_graph
def _all_to_all_single_any_o_async(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    **kwargs,
) -> Callable[..., torch.Tensor]:
    r"""
    x: torch.Tensor, shape (B, S_GLOBAL, H_LOCAL, D) return: Callable that returns (B, S_LOCAL, H_GLOBAL, D)
    """
    # Assume H is provided in kwargs, since we can't infer H from x's shape.
    # The padding logic needs H to determine if padding is necessary.
    H = kwargs.get("num_qo_head", None)
    rank, world_size = _get_rank_world_size(group)
    x, H_PAD = _maybe_pad_o_head(x, H, group)
    shape = x.shape  # (B, S_GLOBAL, H_LOCAL, D)
    (B, S_GLOBAL, H_LOCAL, D) = shape
    # NOTE: We use tensor_split here to ensure the same split policy
    # that we have used in the EquipartitionSharder sharding strategy. Please
    # note that the 'tensor_split' splits a tensor into multiple sub-tensors,
    # all of which are views of input, thus may not introduce extra IO access.
    input_split_sizes = [o.size(1) for o in torch.tensor_split(x, world_size, dim=1)]
    # input_split: e.g, S_GLOBAL=9 input splits across ranks [[5,4], [5,4],..]
    # output_split: e.g, S_GLOBAL=9 output splits across ranks [[5,5], [4,4],..]
    S_LOCAL = input_split_sizes[rank]
    x = x.permute(1, 0, 2, 3).contiguous()  # (S_GLOBAL, B, H_LOCAL, D)
    output_split_sizes = [S_LOCAL] * world_size
    x = fc.all_to_all_single(x, output_split_sizes, input_split_sizes, group)

    def wait() -> torch.Tensor:
        nonlocal x, H_PAD
        x = _wait_tensor(x)  # (S_GLOBAL, B, H_LOCAL, D)
        x = x.reshape(world_size, S_LOCAL, B, H_LOCAL, D)
        x = x.permute(2, 1, 0, 3, 4).contiguous()
        x = x.reshape(B, S_LOCAL, world_size * H_LOCAL, D)
        x = _maybe_unpad_o_head(x, H_PAD, group)
        return x

    return wait


class TemplatedUlyssesAnythingAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        is_causal: bool,
        scale: Optional[float],
        enable_gqa: bool,
        return_lse: bool,
        forward_op,
        backward_op,
        _parallel_config: Optional["ParallelConfig"] = None,
        **kwargs,
    ):
        ulysses_mesh = _parallel_config.context_parallel_config._ulysses_mesh
        group = ulysses_mesh.get_group()

        ctx.forward_op = forward_op
        ctx.backward_op = backward_op
        ctx._parallel_config = _parallel_config

        metadata = _comm_metadata(query)
        query_wait = _all_to_all_single_any_qkv_async(query, group, **metadata)
        key_wait = _all_to_all_single_any_qkv_async(key, group, **metadata)
        value_wait = _all_to_all_single_any_qkv_async(value, group, **metadata)

        query = query_wait()  # type: torch.Tensor
        key = key_wait()  # type: torch.Tensor
        value = value_wait()  # type: torch.Tensor

        out = forward_op(
            ctx,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            _save_ctx=False,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse, *_ = out

        # out: (B, S_Q_GLOBAL, H_LOCAL, D) -> (B, S_Q_LOCAL, H_GLOBAL, D)
        out_wait = _all_to_all_single_any_o_async(out, group, **metadata)

        if return_lse:
            # lse: (B, S_Q_GLOBAL, H_LOCAL)
            lse = lse.unsqueeze(-1)  # (B, S_Q_GLOBAL, H_LOCAL, D=1)
            lse_wait = _all_to_all_single_any_o_async(lse, group, **metadata)
            out = out_wait()  # type: torch.Tensor
            lse = lse_wait()  # type: torch.Tensor
            lse = lse.squeeze(-1).contiguous()  # (B, S_Q_LOCAL, H_GLOBAL)
        else:
            out = out_wait()  # type: torch.Tensor
            lse = None

        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args,
    ):
        raise NotImplementedError("Backward pass for Ulysses Anything Attention in diffusers is not implemented yet.")


@functools.lru_cache(maxsize=64)
def _fill_gather_shapes(shape: Tuple[int], gather_dims: Tuple[int], dim: int, world_size: int) -> List[List[int]]:
    gather_shapes = []
    for i in range(world_size):
        rank_shape = list(copy.deepcopy(shape))
        rank_shape[dim] = gather_dims[i]
        gather_shapes.append(rank_shape)
    return gather_shapes


@maybe_allow_in_graph
def _all_gather_anything(  # noqa: F811
    tensor: torch.Tensor,
    dim: int,
    group: dist.device_mesh.DeviceMesh,
) -> torch.Tensor:
    _, world_size = _get_rank_world_size(group)
    tensor = tensor.contiguous()
    shape = tensor.shape
    rank_dim = shape[dim]
    gather_dims = _gather_size_by_comm(rank_dim, group)

    gather_shapes = _fill_gather_shapes(
        tuple(shape),
        tuple(gather_dims),
        dim,
        world_size,
    )

    gathered_tensors = [
        torch.empty(
            shape,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        for shape in gather_shapes
    ]

    dist.all_gather(gathered_tensors, tensor, group=group)
    gathered_tensor = torch.cat(gathered_tensors, dim=dim)
    return gathered_tensor


class AllGatherAnythingFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        dim: int,
        group: dist.device_mesh.DeviceMesh,
    ):
        ctx.dim = dim
        ctx.group = group
        ctx.world_size = dist.get_world_size(group)
        ctx.rank = dist.get_rank(group)
        gathered_tensor = _all_gather_anything(tensor, dim, group)
        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # NOTE: We use `tensor_split` instead of chunk, because the `chunk`
        # function may return fewer than the specified number of chunks!
        grad_splits = torch.tensor_split(grad_output, ctx.world_size, dim=ctx.dim)
        return grad_splits[ctx.rank], None, None


class PartitionAnythingSharder:
    @classmethod
    def shard_anything(
        cls, tensor: torch.Tensor, dim: int, mesh: torch.distributed.device_mesh.DeviceMesh
    ) -> torch.Tensor:
        assert tensor.size()[dim] >= mesh.size(), (
            f"Cannot shard tensor of size {tensor.size()} along dim {dim} across mesh of size {mesh.size()}."
        )
        # NOTE: We use `tensor_split` instead of chunk, because the `chunk`
        # function may return fewer than the specified number of chunks! For example,
        # x = torch.tensor([1,2,3,4,5]), torch.chunk(x, 4) will return only 3 chunks:
        # (tensor([1, 2]), tensor([3, 4]), tensor([5])). This behavior can lead to
        # inconsistencies when sharding tensors across multiple devices. In contrast,
        # tensor_split will always return the specified number of chunks, the last chunk
        # may be smaller if the tensor size is not divisible by the number of chunks.
        # For example, torch.tensor_split(x, 4) will return 4 chunks:
        # (tensor([1, 2]), tensor([3]), tensor([4]), tensor([5])).
        return tensor.tensor_split(mesh.size(), dim=dim)[dist.get_rank(mesh.get_group())]

    @classmethod
    def unshard_anything(
        cls, tensor: torch.Tensor, dim: int, mesh: torch.distributed.device_mesh.DeviceMesh
    ) -> torch.Tensor:
        tensor = tensor.contiguous()
        # NOTE: We use AllGatherAnythingFunction to support gathering
        # tensors with complex and uneven sizes across all ranks. It handles the
        # case where the tensor size (the seq_len of hidden_states) along the
        # specified dimension is not divisible by the number of ranks in the mesh.
        tensor = AllGatherAnythingFunction.apply(tensor, dim, mesh.get_group())
        return tensor
