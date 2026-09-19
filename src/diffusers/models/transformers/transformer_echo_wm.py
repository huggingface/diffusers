# Copyright 2026 The Echo-WM and HuggingFace Teams.
# All rights reserved.
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

# Echo-WM is derived from LTX-2, but keeps its UCPE camera attention and Flash KV-cache behavior in a separate model
# implementation so changes to Echo-WM do not alter the LTX-2 architecture.

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import BaseOutput, apply_lora_scale, is_torch_version, logging
from ...utils.torch_utils import maybe_adjust_dtype_for_device
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings, PixArtAlphaTextProjection
from ..modeling_utils import ModelMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def apply_interleaved_rotary_emb(x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = (value.to(x.dtype) for value in freqs)
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)  # [B, S, C // 2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = x * cos + x_rotated * sin
    return out


def apply_split_rotary_emb(x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = (value.to(x.dtype) for value in freqs)

    x_dtype = x.dtype
    needs_reshape = False
    if x.ndim != 4 and cos.ndim == 4:
        # cos is (b, h, t, r) -> reshape x to (b, h, t, dim_per_head)
        b, h, t, _ = cos.shape
        x = x.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    # Split last dim (2*r) into (d=2, r)
    last = x.shape[-1]
    if last % 2 != 0:
        raise ValueError(f"Expected x.shape[-1] to be even for split rotary, got {last}.")
    r = last // 2

    # (..., 2, r)
    # The reference rounds the first product before addcmul; upcasting changes BF16 attention scores.
    split_x = x.reshape(*x.shape[:-1], 2, r)
    first_x = split_x[..., :1, :]  # (..., 1, r)
    second_x = split_x[..., 1:, :]  # (..., 1, r)

    cos_u = cos.unsqueeze(-2)  # broadcast to (..., 1, r) against (..., 2, r)
    sin_u = sin.unsqueeze(-2)

    out = split_x * cos_u
    first_out = out[..., :1, :]
    second_out = out[..., 1:, :]

    first_out.addcmul_(-sin_u, second_x)
    second_out.addcmul_(sin_u, first_x)

    out = out.reshape(*out.shape[:-2], last)

    if needs_reshape:
        out = out.swapaxes(1, 2).reshape(b, t, -1)

    out = out.to(dtype=x_dtype)
    return out


def _apply_echo_wm_rotary_emb(
    hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor], rope_type: str
) -> torch.Tensor:
    if rope_type == "interleaved":
        return apply_interleaved_rotary_emb(hidden_states, rotary_emb)
    return apply_split_rotary_emb(hidden_states, rotary_emb)


def _slice_echo_wm_rotary_emb(
    rotary_emb: tuple[torch.Tensor, torch.Tensor], start: int, end: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return rotary_emb[0][..., start:end, :], rotary_emb[1][..., start:end, :]


def _ucpe_rope_coefficients(
    positions: torch.Tensor,
    freq_base: float,
    freq_scale: float,
    feature_dim: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if feature_dim % 2 != 0:
        raise ValueError(f"UCPE rotary feature dimension must be even, got {feature_dim}.")
    num_frequencies = feature_dim // 2
    frequencies = freq_scale * freq_base ** (
        -torch.arange(num_frequencies, device=positions.device, dtype=torch.float32) / num_frequencies
    )
    angles = positions.to(torch.float32)[:, None] * frequencies[None]
    return angles.cos()[None, None].to(dtype), angles.sin()[None, None].to(dtype)


def _ucpe_apply_rope(
    hidden_states: torch.Tensor, coefficients: tuple[torch.Tensor, torch.Tensor], inverse: bool = False
) -> torch.Tensor:
    cos, sin = coefficients
    if cos.shape[2] != hidden_states.shape[2]:
        if hidden_states.shape[2] % cos.shape[2] != 0:
            raise ValueError(
                f"UCPE sequence length {hidden_states.shape[2]} is not divisible by the spatial grid size "
                f"{cos.shape[2]}."
            )
        repeats = hidden_states.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, repeats, 1)
        sin = sin.repeat(1, 1, repeats, 1)
    # Keep the coefficient dtype. In Base inference the reference combines BF16 projections with FP32 rotary
    # coefficients, promoting the rotary blocks (and therefore UCPE attention) to FP32. The cached Flash path
    # already passes FP32 activations here.
    cos = cos.to(device=hidden_states.device)
    sin = sin.to(device=hidden_states.device)
    first, second = hidden_states.chunk(2, dim=-1)
    if inverse:
        return torch.cat((cos * first - sin * second, sin * first + cos * second), dim=-1)
    return torch.cat((cos * first + sin * second, -sin * first + cos * second), dim=-1)


def _ucpe_invert_se3(matrices: torch.Tensor) -> torch.Tensor:
    rotation = matrices[..., :3, :3].transpose(-1, -2)
    result = torch.zeros_like(matrices)
    result[..., :3, :3] = rotation
    result[..., :3, 3] = -torch.einsum("...ij,...j->...i", rotation, matrices[..., :3, 3])
    result[..., 3, 3] = 1.0
    return result


def _ucpe_rebase_translation(viewmats: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=viewmats.device.type, enabled=False):
        matrices, anchor = viewmats.float(), anchor.float()
        shift = -(anchor[..., :3, :3].transpose(-1, -2) @ anchor[..., :3, 3:4])
        result = matrices.clone()
        result[..., :3, 3:4] += result[..., :3, :3] @ shift
    return result


def _ucpe_lift_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((*intrinsics.shape[:-2], 4, 4), device=intrinsics.device, dtype=intrinsics.dtype)
    result[..., :3, :3] = intrinsics
    result[..., 3, 3] = 1.0
    return result


def _ucpe_invert_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(intrinsics)
    result[..., 0, 0] = 1.0 / intrinsics[..., 0, 0]
    result[..., 1, 1] = 1.0 / intrinsics[..., 1, 1]
    result[..., 0, 2] = -intrinsics[..., 0, 2] / intrinsics[..., 0, 0]
    result[..., 1, 2] = -intrinsics[..., 1, 2] / intrinsics[..., 1, 1]
    result[..., 2, 2] = 1.0
    return result


def _ucpe_apply_projection(hidden_states: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, sequence_length, feature_dim = hidden_states.shape
    matrix_dim = matrices.shape[-1]
    if feature_dim % matrix_dim != 0:
        raise ValueError(f"UCPE feature dimension {feature_dim} must be divisible by {matrix_dim}.")
    matrices = matrices.to(device=hidden_states.device, dtype=hidden_states.dtype)
    if matrices.shape[1] == sequence_length:
        values = hidden_states.reshape(batch_size, num_heads, sequence_length, feature_dim // matrix_dim, matrix_dim)
        return torch.einsum("btij,bntpj->bntpi", matrices, values).reshape_as(hidden_states)
    num_cameras = matrices.shape[1]
    if sequence_length % num_cameras != 0:
        raise ValueError(
            f"UCPE sequence length {sequence_length} must be divisible by the camera count {num_cameras}."
        )
    values = hidden_states.reshape(batch_size, num_heads, num_cameras, -1, feature_dim // matrix_dim, matrix_dim)
    return torch.einsum("bcij,bncpkj->bncpki", matrices, values).reshape_as(hidden_states)


def _ucpe_apply_blocks(
    hidden_states: torch.Tensor, transforms: list[tuple[Callable[[torch.Tensor], torch.Tensor], int]]
) -> torch.Tensor:
    sizes = [size for _, size in transforms]
    if hidden_states.shape[-1] != sum(sizes):
        raise ValueError(f"UCPE block sizes {sizes} do not match head dimension {hidden_states.shape[-1]}.")
    return torch.cat([fn(value) for (fn, _), value in zip(transforms, hidden_states.split(sizes, dim=-1))], dim=-1)


def _ucpe_transform(transform: Callable[[torch.Tensor], torch.Tensor], hidden_states: torch.Tensor) -> torch.Tensor:
    dtype = hidden_states.dtype
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        return transform(hidden_states.float()).to(dtype)


class EchoWMCameraRotaryPosEmbed(nn.Module):
    r"""Camera-relative positional encoding used by Echo-WM's additional video attention branch."""

    def __init__(
        self,
        head_dim: int,
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
    ):
        super().__init__()
        if head_dim % 8 != 0:
            raise ValueError(f"UCPE head dimension must be divisible by 8, got {head_dim}.")
        if min(patches_x, patches_y, image_width, image_height) <= 0:
            raise ValueError("UCPE grid and image dimensions must be positive.")
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height
        self.freq_base = freq_base
        self.freq_scale = freq_scale
        x_positions = torch.arange(patches_x).tile(patches_y)
        y_positions = torch.arange(patches_y).repeat_interleave(patches_x)
        x_cos, x_sin = _ucpe_rope_coefficients(x_positions, freq_base, freq_scale, head_dim // 4)
        y_cos, y_sin = _ucpe_rope_coefficients(y_positions, freq_base, freq_scale, head_dim // 4)
        self.register_buffer("x_cos", x_cos, persistent=False)
        self.register_buffer("x_sin", x_sin, persistent=False)
        self.register_buffer("y_cos", y_cos, persistent=False)
        self.register_buffer("y_sin", y_sin, persistent=False)

    def prepare_transforms(
        self, viewmats: torch.Tensor, intrinsics: torch.Tensor
    ) -> tuple[Callable, Callable, Callable]:
        if viewmats.ndim != 4 or viewmats.shape[-2:] != (4, 4):
            raise ValueError(f"`ucpe_viewmats` must have shape (batch, frames, 4, 4), got {viewmats.shape}.")
        if intrinsics.shape != (*viewmats.shape[:2], 3, 3):
            raise ValueError(
                f"`ucpe_intrinsics` must have shape {(*viewmats.shape[:2], 3, 3)}, got {intrinsics.shape}."
            )

        if self.x_cos.is_meta or self.x_cos.dtype != torch.float32:
            # The reference precomputes these coefficients in FP32 on CPU, then moves them to the execution device.
            # Recreate them when a broad module dtype cast has converted the non-persistent buffers to BF16.
            x_positions = torch.arange(self.patches_x).tile(self.patches_y)
            y_positions = torch.arange(self.patches_y).repeat_interleave(self.patches_x)
            x_coefficients = _ucpe_rope_coefficients(
                x_positions, self.freq_base, self.freq_scale, self.head_dim // 4, torch.float32
            )
            y_coefficients = _ucpe_rope_coefficients(
                y_positions, self.freq_base, self.freq_scale, self.head_dim // 4, torch.float32
            )
            x_coefficients = tuple(value.to(viewmats.device) for value in x_coefficients)
            y_coefficients = tuple(value.to(viewmats.device) for value in y_coefficients)
        else:
            x_coefficients = (self.x_cos, self.x_sin)
            y_coefficients = (self.y_cos, self.y_sin)

        normalized = torch.zeros_like(intrinsics)
        normalized[..., 0, 0] = intrinsics[..., 0, 0] / self.image_width
        normalized[..., 1, 1] = intrinsics[..., 1, 1] / self.image_height
        normalized[..., 0, 2] = intrinsics[..., 0, 2] / self.image_width - 0.5
        normalized[..., 1, 2] = intrinsics[..., 1, 2] / self.image_height - 0.5
        normalized[..., 2, 2] = 1.0
        projection = torch.einsum("...ij,...jk->...ik", _ucpe_lift_intrinsics(normalized), viewmats)
        projection_t = projection.transpose(-1, -2)
        projection_inv = torch.einsum(
            "...ij,...jk->...ik",
            _ucpe_invert_se3(viewmats),
            _ucpe_lift_intrinsics(_ucpe_invert_intrinsics(normalized)),
        )
        half, quarter = self.head_dim // 2, self.head_dim // 4

        def apply_query(value):
            return _ucpe_apply_blocks(
                value,
                [
                    (lambda x: _ucpe_apply_projection(x, projection_t), half),
                    (lambda x: _ucpe_apply_rope(x, x_coefficients), quarter),
                    (lambda x: _ucpe_apply_rope(x, y_coefficients), quarter),
                ],
            )

        def apply_key_value(value):
            return _ucpe_apply_blocks(
                value,
                [
                    (lambda x: _ucpe_apply_projection(x, projection_inv), half),
                    (lambda x: _ucpe_apply_rope(x, x_coefficients), quarter),
                    (lambda x: _ucpe_apply_rope(x, y_coefficients), quarter),
                ],
            )

        def apply_output(value):
            return _ucpe_apply_blocks(
                value,
                [
                    (lambda x: _ucpe_apply_projection(x, projection), half),
                    (lambda x: _ucpe_apply_rope(x, x_coefficients, inverse=True), quarter),
                    (lambda x: _ucpe_apply_rope(x, y_coefficients, inverse=True), quarter),
                ],
            )

        return apply_query, apply_key_value, apply_output


@dataclass
class AudioVisualModelOutput(BaseOutput):
    r"""
    Holds the output of an audiovisual model which produces both visual (e.g. video) and audio outputs.

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, num_frames, height, width)`):
            The hidden states output conditioned on the `encoder_hidden_states` input, representing the visual output
            of the model. This is typically a video (spatiotemporal) output.
        audio_sample (`torch.Tensor` of shape `(batch_size, num_audio_tokens, audio_out_channels)`):
            The denoised audio latent patch sequence.
    """

    sample: "torch.Tensor"  # noqa: F821
    audio_sample: "torch.Tensor"  # noqa: F821


class EchoWMKVCache:
    """Container holding one mutable KV-cache dictionary per Echo-WM transformer layer."""

    def __init__(self, layer_caches: list[dict[str, dict[str, Any]]]):
        self.layer_caches = layer_caches

    def __getitem__(self, layer_idx: int) -> dict[str, dict[str, Any]]:
        return self.layer_caches[layer_idx]

    def __iter__(self):
        return iter(self.layer_caches)

    def __len__(self) -> int:
        return len(self.layer_caches)


def _update_causal_kv_cache(
    cache: dict[str, Any], start: int, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace a noisy token range and return its active sink-plus-FIFO history."""
    if torch.is_grad_enabled():
        raise RuntimeError("Echo-WM Flash KV caches are inference-only.")
    end = start + key.shape[1]
    positions = torch.arange(start, end, device=key.device)
    if cache.get("key") is not None:
        keep = cache["positions"] < start
        positions = torch.cat((cache["positions"][keep], positions))
        key = torch.cat((cache["key"][:, keep], key), dim=1)
        value = torch.cat((cache["value"][:, keep], value), dim=1)
    local_size = int(cache["local_size"])
    sink_size = int(cache["sink_size"])
    if positions.numel() > local_size:
        sink = positions < sink_size
        recent_start = max(sink_size, end - (local_size - int(sink.sum())))
        keep = sink | (positions >= recent_start)
        positions, key, value = positions[keep], key[:, keep], value[:, keep]
    cache.update(positions=positions, key=key.detach(), value=value.detach())
    return key, value


class EchoWMAdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://huggingface.co/papers/2310.00426; Section 2.3) and adapted by the LTX-2.0
    model. In particular, the number of modulation parameters to be calculated is now configurable.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_mod_params (`int`, *optional*, defaults to `6`):
            The number of modulation parameters which will be calculated in the first return argument. The default of 6
            is standard, but sometimes we may want to have a different (usually smaller) number of modulation
            parameters.
        use_additional_conditions (`bool`, *optional*, defaults to `False`):
            Whether to use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, num_mod_params: int = 6, use_additional_conditions: bool = False):
        super().__init__()
        self.num_mod_params = num_mod_params

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, self.num_mod_params * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
        batch_size: int | None = None,
        hidden_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # No modulation happening here.
        added_cond_kwargs = added_cond_kwargs or {"resolution": None, "aspect_ratio": None}
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class EchoWMAudioVideoAttnProcessor:
    r"""Attention processor for Echo-WM's video, audio, and cross-modal attention layers."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if is_torch_version("<", "2.0"):
            raise ValueError(
                "Echo-WM attention processors require PyTorch 2.0 or newer. Please upgrade your PyTorch installation."
            )

    def __call__(
        self,
        attn: "EchoWMAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: dict[str, Any] | None = None,
        kv_cache_start: int = 0,
        crossattn_cache: dict[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.to_gate_logits is not None:
            # Calculate gate logits on original hidden_states
            gate_logits = attn.to_gate_logits(hidden_states)

        query = attn.to_q(hidden_states)
        query = attn.norm_q(query)
        if crossattn_cache is not None and crossattn_cache["key"] is not None:
            key = crossattn_cache["key"]
            value = crossattn_cache["value"]
        else:
            key = attn.norm_k(attn.to_k(encoder_hidden_states))
            value = attn.to_v(encoder_hidden_states)
            if crossattn_cache is not None:
                crossattn_cache.update(key=key.detach(), value=value.detach())

        local_rotary_emb = kv_cache.get("local_rotary_emb") if kv_cache is not None else None
        local_query_rotary_emb = kv_cache.get("local_query_rotary_emb") if kv_cache is not None else None
        local_key_rotary_emb = kv_cache.get("local_key_rotary_emb") if kv_cache is not None else None
        if local_rotary_emb is not None:
            key, value = _update_causal_kv_cache(kv_cache, kv_cache_start, key, value)
            query_length = query.shape[1]
            query = _apply_echo_wm_rotary_emb(
                query,
                _slice_echo_wm_rotary_emb(local_rotary_emb, key.shape[1] - query_length, key.shape[1]),
                attn.rope_type,
            )
            key = _apply_echo_wm_rotary_emb(
                key, _slice_echo_wm_rotary_emb(local_rotary_emb, 0, key.shape[1]), attn.rope_type
            )
        elif local_query_rotary_emb is not None and local_key_rotary_emb is not None:
            new_key_length = key.shape[1]
            key, value = _update_causal_kv_cache(kv_cache, kv_cache_start, key, value)
            query_start, query_end = kv_cache["local_query_slices"][(kv_cache_start, kv_cache_start + new_key_length)]
            query = _apply_echo_wm_rotary_emb(
                query, _slice_echo_wm_rotary_emb(local_query_rotary_emb, query_start, query_end), attn.rope_type
            )
            key = _apply_echo_wm_rotary_emb(
                key, _slice_echo_wm_rotary_emb(local_key_rotary_emb, 0, key.shape[1]), attn.rope_type
            )
        else:
            if query_rotary_emb is not None:
                query = _apply_echo_wm_rotary_emb(query, query_rotary_emb, attn.rope_type)
                key = _apply_echo_wm_rotary_emb(
                    key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb, attn.rope_type
                )
            if kv_cache is not None:
                key, value = _update_causal_kv_cache(kv_cache, kv_cache_start, key, value)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if attn.to_gate_logits is not None:
            hidden_states = hidden_states.unflatten(2, (attn.heads, -1))  # [B, T, H, D]
            # The factor of 2.0 is so that if the gates logits are zero-initialized the initial gates are all 1
            gates = 2.0 * torch.sigmoid(gate_logits)  # [B, T, H]
            hidden_states = hidden_states * gates.unsqueeze(-1)
            hidden_states = hidden_states.flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class EchoWMPerturbedAttnProcessor:
    r"""Echo-WM attention processor with perturbation masking and per-head gating."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if is_torch_version("<", "2.0"):
            raise ValueError(
                "Echo-WM attention processors require PyTorch 2.0 or newer. Please upgrade your PyTorch installation."
            )

    def __call__(
        self,
        attn: "EchoWMAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool | None = None,
        kv_cache: dict[str, Any] | None = None,
        kv_cache_start: int = 0,
        crossattn_cache: dict[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.to_gate_logits is not None:
            # Calculate gate logits on original hidden_states
            gate_logits = attn.to_gate_logits(hidden_states)

        has_cached_context = crossattn_cache is not None and crossattn_cache["key"] is not None
        value = crossattn_cache["value"] if has_cached_context else attn.to_v(encoder_hidden_states)
        if all_perturbed is None:
            all_perturbed = torch.all(perturbation_mask == 0) if perturbation_mask is not None else False

        if all_perturbed:
            # Skip attention, use the value projection value
            hidden_states = value
        else:
            query = attn.to_q(hidden_states)
            query = attn.norm_q(query)
            if has_cached_context:
                key = crossattn_cache["key"]
            else:
                key = attn.norm_k(attn.to_k(encoder_hidden_states))
                if crossattn_cache is not None:
                    crossattn_cache.update(key=key.detach(), value=value.detach())

            local_rotary_emb = kv_cache.get("local_rotary_emb") if kv_cache is not None else None
            local_query_rotary_emb = kv_cache.get("local_query_rotary_emb") if kv_cache is not None else None
            local_key_rotary_emb = kv_cache.get("local_key_rotary_emb") if kv_cache is not None else None
            if local_rotary_emb is not None:
                key, value = _update_causal_kv_cache(kv_cache, kv_cache_start, key, value)
                query_length = query.shape[1]
                query = _apply_echo_wm_rotary_emb(
                    query,
                    _slice_echo_wm_rotary_emb(local_rotary_emb, key.shape[1] - query_length, key.shape[1]),
                    attn.rope_type,
                )
                key = _apply_echo_wm_rotary_emb(
                    key, _slice_echo_wm_rotary_emb(local_rotary_emb, 0, key.shape[1]), attn.rope_type
                )
            elif local_query_rotary_emb is not None and local_key_rotary_emb is not None:
                new_key_length = key.shape[1]
                key, value = _update_causal_kv_cache(kv_cache, kv_cache_start, key, value)
                query_start, query_end = kv_cache["local_query_slices"][
                    (kv_cache_start, kv_cache_start + new_key_length)
                ]
                query = _apply_echo_wm_rotary_emb(
                    query,
                    _slice_echo_wm_rotary_emb(local_query_rotary_emb, query_start, query_end),
                    attn.rope_type,
                )
                key = _apply_echo_wm_rotary_emb(
                    key, _slice_echo_wm_rotary_emb(local_key_rotary_emb, 0, key.shape[1]), attn.rope_type
                )
            else:
                if query_rotary_emb is not None:
                    query = _apply_echo_wm_rotary_emb(query, query_rotary_emb, attn.rope_type)
                    key = _apply_echo_wm_rotary_emb(
                        key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb, attn.rope_type
                    )
                if kv_cache is not None:
                    key, value = _update_causal_kv_cache(kv_cache, kv_cache_start, key, value)

            query = query.unflatten(2, (attn.heads, -1))
            key = key.unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))

            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(query.dtype)

            if perturbation_mask is not None:
                value = value.flatten(2, 3)
                hidden_states = torch.lerp(value, hidden_states, perturbation_mask)

        if attn.to_gate_logits is not None:
            hidden_states = hidden_states.unflatten(2, (attn.heads, -1))  # [B, T, H, D]
            # The factor of 2.0 is so that if the gates logits are zero-initialized the initial gates are all 1
            gates = 2.0 * torch.sigmoid(gate_logits)  # [B, T, H]
            hidden_states = hidden_states * gates.unsqueeze(-1)
            hidden_states = hidden_states.flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class EchoWMAttention(torch.nn.Module, AttentionModuleMixin):
    r"""
    Attention class for Echo-WM. It supports separate query and key RoPE embeddings for audio-to-video (a2v) and
    video-to-audio (v2a) cross-attention, together with bounded KV caches for Flash inference.
    """

    _default_processor_cls = EchoWMAudioVideoAttnProcessor
    _available_processors = [EchoWMAudioVideoAttnProcessor, EchoWMPerturbedAttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        kv_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        cross_attention_dim: int | None = None,
        out_bias: bool = True,
        qk_norm: str = "rms_norm_across_heads",
        norm_eps: float = 1e-6,
        norm_elementwise_affine: bool = True,
        rope_type: str = "interleaved",
        apply_gated_attention: bool = False,
        processor=None,
    ):
        super().__init__()
        if qk_norm != "rms_norm_across_heads":
            raise NotImplementedError("Only 'rms_norm_across_heads' is supported as a valid value for `qk_norm`.")

        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = query_dim
        self.heads = heads
        self.rope_type = rope_type

        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head * kv_heads, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_v = torch.nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(torch.nn.Dropout(dropout))

        if apply_gated_attention:
            # Per head gate values
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        hidden_states = self.processor(
            self, hidden_states, encoder_hidden_states, attention_mask, query_rotary_emb, key_rotary_emb, **kwargs
        )
        return hidden_states


class EchoWMVideoTransformerBlock(nn.Module):
    r"""
    Transformer block used in Echo-WM, derived from the LTX-2 audiovisual transformer block.

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_dim: int,
        audio_num_attention_heads: int,
        audio_attention_head_dim,
        audio_cross_attention_dim: int,
        video_gated_attn: bool = False,
        video_cross_attn_adaln: bool = False,
        audio_gated_attn: bool = False,
        audio_cross_attn_adaln: bool = False,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        rope_type: str = "interleaved",
        perturbed_attn: bool = False,
        ff_bias: bool = True,
        audio_ff_bias: bool = True,
        ucpe_attention_dim: int | None = None,
        ucpe_num_attention_heads: int = 8,
        ucpe_patches_x: int = 40,
        ucpe_patches_y: int = 22,
        ucpe_image_width: int = 1280,
        ucpe_image_height: int = 704,
        ucpe_freq_base: float = 100.0,
        ucpe_freq_scale: float = 1.0,
    ):
        super().__init__()

        self.perturbed_attn = perturbed_attn
        if perturbed_attn:
            attn_processor_cls = EchoWMPerturbedAttnProcessor
        else:
            attn_processor_cls = EchoWMAudioVideoAttnProcessor

        # 1. Self-Attention (video and audio)
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = EchoWMAttention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
            apply_gated_attention=video_gated_attn,
            processor=attn_processor_cls(),
        )

        self.ucpe_enabled = ucpe_attention_dim is not None
        if self.ucpe_enabled:
            if ucpe_attention_dim % ucpe_num_attention_heads != 0:
                raise ValueError(
                    f"UCPE attention dimension {ucpe_attention_dim} must be divisible by "
                    f"{ucpe_num_attention_heads} heads."
                )
            self.ucpe_num_attention_heads = ucpe_num_attention_heads
            self.ucpe_head_dim = ucpe_attention_dim // ucpe_num_attention_heads
            self.ucpe_q_proj = nn.Linear(dim, ucpe_attention_dim, bias=False)
            self.ucpe_k_proj = nn.Linear(dim, ucpe_attention_dim, bias=False)
            self.ucpe_v_proj = nn.Linear(dim, ucpe_attention_dim, bias=False)
            self.ucpe_out_proj = nn.Linear(ucpe_attention_dim, dim, bias=True)
            nn.init.xavier_uniform_(self.ucpe_q_proj.weight)
            nn.init.xavier_uniform_(self.ucpe_k_proj.weight)
            nn.init.xavier_uniform_(self.ucpe_v_proj.weight)
            nn.init.zeros_(self.ucpe_out_proj.weight)
            nn.init.zeros_(self.ucpe_out_proj.bias)
            self.ucpe = EchoWMCameraRotaryPosEmbed(
                head_dim=self.ucpe_head_dim,
                patches_x=ucpe_patches_x,
                patches_y=ucpe_patches_y,
                image_width=ucpe_image_width,
                image_height=ucpe_image_height,
                freq_base=ucpe_freq_base,
                freq_scale=ucpe_freq_scale,
            )

        self.audio_norm1 = nn.RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_attn1 = EchoWMAttention(
            query_dim=audio_dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
            apply_gated_attention=audio_gated_attn,
            processor=attn_processor_cls(),
        )

        # 2. Prompt Cross-Attention
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = EchoWMAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
            apply_gated_attention=video_gated_attn,
            processor=attn_processor_cls(),
        )

        self.audio_norm2 = nn.RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_attn2 = EchoWMAttention(
            query_dim=audio_dim,
            cross_attention_dim=audio_cross_attention_dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
            apply_gated_attention=audio_gated_attn,
            processor=attn_processor_cls(),
        )

        # 3. Audio-to-Video (a2v) and Video-to-Audio (v2a) Cross-Attention
        # Audio-to-Video (a2v) Attention --> Q: Video; K,V: Audio
        self.audio_to_video_norm = nn.RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_to_video_attn = EchoWMAttention(
            query_dim=dim,
            cross_attention_dim=audio_dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
            apply_gated_attention=video_gated_attn,
            processor=attn_processor_cls(),
        )

        # Video-to-Audio (v2a) Attention --> Q: Audio; K,V: Video
        self.video_to_audio_norm = nn.RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.video_to_audio_attn = EchoWMAttention(
            query_dim=audio_dim,
            cross_attention_dim=dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
            apply_gated_attention=audio_gated_attn,
            processor=attn_processor_cls(),
        )

        # 4. Feedforward layers
        self.norm3 = nn.RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.ff = FeedForward(dim, activation_fn=activation_fn, bias=ff_bias)

        self.audio_norm3 = nn.RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_ff = FeedForward(audio_dim, activation_fn=activation_fn, bias=audio_ff_bias)

        # 5. Per-Layer Modulation Parameters
        # Self-Attention (attn1) / Feedforward AdaLayerNorm-Zero mod params
        # 6 base mod params for text cross-attn K,V; if cross_attn_adaln, also has mod params for Q
        self.video_cross_attn_adaln = video_cross_attn_adaln
        self.audio_cross_attn_adaln = audio_cross_attn_adaln
        video_mod_param_num = 9 if self.video_cross_attn_adaln else 6
        audio_mod_param_num = 9 if self.audio_cross_attn_adaln else 6
        self.scale_shift_table = nn.Parameter(torch.randn(video_mod_param_num, dim) / dim**0.5)
        self.audio_scale_shift_table = nn.Parameter(torch.randn(audio_mod_param_num, audio_dim) / audio_dim**0.5)

        # Prompt cross-attn (attn2) additional modulation params
        self.cross_attn_adaln = video_cross_attn_adaln or audio_cross_attn_adaln
        if self.cross_attn_adaln:
            self.prompt_scale_shift_table = nn.Parameter(torch.randn(2, dim))
            self.audio_prompt_scale_shift_table = nn.Parameter(torch.randn(2, audio_dim))

        # Per-layer a2v, v2a Cross-Attention mod params
        self.video_a2v_cross_attn_scale_shift_table = nn.Parameter(torch.randn(5, dim))
        self.audio_a2v_cross_attn_scale_shift_table = nn.Parameter(torch.randn(5, audio_dim))

    @staticmethod
    def get_mod_params(
        scale_shift_table: torch.Tensor, temb: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]
        ada_values = scale_shift_table[None, None].to(temb.device) + temb.reshape(
            batch_size, temb.shape[1], num_ada_params, -1
        )
        ada_params = ada_values.unbind(dim=2)
        return ada_params

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        temb_audio: torch.Tensor,
        temb_ca_scale_shift: torch.Tensor,
        temb_ca_audio_scale_shift: torch.Tensor,
        temb_ca_gate: torch.Tensor,
        temb_ca_audio_gate: torch.Tensor,
        temb_prompt: torch.Tensor | None = None,
        temb_prompt_audio: torch.Tensor | None = None,
        video_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        audio_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        ca_video_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        ca_audio_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        audio_encoder_attention_mask: torch.Tensor | None = None,
        self_attention_mask: torch.Tensor | None = None,
        audio_self_attention_mask: torch.Tensor | None = None,
        a2v_cross_attention_mask: torch.Tensor | None = None,
        v2a_cross_attention_mask: torch.Tensor | None = None,
        use_a2v_cross_attention: bool = True,
        use_v2a_cross_attention: bool = True,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool | None = None,
        ucpe_viewmats: torch.Tensor | None = None,
        ucpe_intrinsics: torch.Tensor | None = None,
        kv_cache: dict[str, dict[str, Any]] | None = None,
        current_video_token_start: int = 0,
        current_audio_token_start: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.size(0)

        # 1. Video and Audio Self-Attention
        # 1.1. Video Self-Attention
        video_ada_params = self.get_mod_params(self.scale_shift_table, temb, batch_size)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = video_ada_params[:6]
        if self.video_cross_attn_adaln:
            shift_text_q, scale_text_q, gate_text_q = video_ada_params[6:9]

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        video_self_attn_args = {
            "hidden_states": norm_hidden_states,
            "encoder_hidden_states": None,
            "query_rotary_emb": video_rotary_emb,
            "attention_mask": self_attention_mask,
            "kv_cache": kv_cache.get("video_self") if kv_cache is not None else None,
            "kv_cache_start": current_video_token_start,
        }
        if self.perturbed_attn:
            video_self_attn_args["perturbation_mask"] = perturbation_mask
            video_self_attn_args["all_perturbed"] = all_perturbed

        attn_hidden_states = self.attn1(**video_self_attn_args)
        if self.ucpe_enabled and ucpe_viewmats is not None and ucpe_intrinsics is not None:
            patches_per_frame = self.ucpe.patches_x * self.ucpe.patches_y
            if norm_hidden_states.shape[1] % patches_per_frame != 0:
                raise ValueError(
                    f"UCPE video tokens must contain complete {self.ucpe.patches_y}x{self.ucpe.patches_x} frames, "
                    f"got {norm_hidden_states.shape[1]} tokens."
                )
            batch_size, sequence_length, _ = norm_hidden_states.shape
            query = self.ucpe_q_proj(norm_hidden_states).view(
                batch_size, sequence_length, self.ucpe_num_attention_heads, self.ucpe_head_dim
            )
            key = self.ucpe_k_proj(norm_hidden_states).view(
                batch_size, sequence_length, self.ucpe_num_attention_heads, self.ucpe_head_dim
            )
            value = self.ucpe_v_proj(norm_hidden_states).view(
                batch_size, sequence_length, self.ucpe_num_attention_heads, self.ucpe_head_dim
            )
            frame_start = current_video_token_start // patches_per_frame
            frame_end = frame_start + sequence_length // patches_per_frame
            query_viewmats = ucpe_viewmats[:, frame_start:frame_end]
            query_intrinsics = ucpe_intrinsics[:, frame_start:frame_end]
            ucpe_cache = kv_cache.get("video_ucpe") if kv_cache is not None else None
            if ucpe_cache is not None:
                raw_key = key.flatten(2, 3)
                raw_value = value.flatten(2, 3)
                raw_key, raw_value = _update_causal_kv_cache(ucpe_cache, current_video_token_start, raw_key, raw_value)
                key = raw_key.unflatten(2, (self.ucpe_num_attention_heads, self.ucpe_head_dim))
                value = raw_value.unflatten(2, (self.ucpe_num_attention_heads, self.ucpe_head_dim))
                token_positions = ucpe_cache["positions"]
                frame_indices = token_positions[::patches_per_frame] // patches_per_frame
                key_viewmats = ucpe_viewmats.index_select(1, frame_indices)
                key_intrinsics = ucpe_intrinsics.index_select(1, frame_indices)
                local_frames = ucpe_cache["local_size"] // patches_per_frame
                sink_frames = ucpe_cache["sink_size"] // patches_per_frame
                anchor_index = (
                    0 if frame_end <= local_frames else max(sink_frames, frame_end - (local_frames - sink_frames))
                )
                anchor = ucpe_viewmats[:, anchor_index : anchor_index + 1]
                query_viewmats = _ucpe_rebase_translation(query_viewmats, anchor)
                key_viewmats = _ucpe_rebase_translation(key_viewmats, anchor)
                apply_query, _, apply_output = self.ucpe.prepare_transforms(query_viewmats, query_intrinsics)
                _, apply_key_value, _ = self.ucpe.prepare_transforms(key_viewmats, key_intrinsics)
                query = _ucpe_transform(apply_query, query.transpose(1, 2)).transpose(1, 2)
                key = _ucpe_transform(apply_key_value, key.transpose(1, 2)).transpose(1, 2)
                value = _ucpe_transform(apply_key_value, value.transpose(1, 2)).transpose(1, 2)
            else:
                apply_query, apply_key_value, apply_output = self.ucpe.prepare_transforms(
                    query_viewmats, query_intrinsics
                )
                # Base inference calls SDPA after UCPE with [B, H, S, D] inputs. Preserve that layout through the
                # dispatcher's model-layout conversion to retain the same kernel and rounding path.
                query = apply_query(query.transpose(1, 2))
                key = apply_key_value(key.transpose(1, 2))
                value = apply_key_value(value.transpose(1, 2))
            if ucpe_cache is None:
                query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
            ucpe_hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                backend=self.attn1.processor._attention_backend,
                parallel_config=self.attn1.processor._parallel_config,
            )
            if ucpe_cache is not None:
                ucpe_hidden_states = _ucpe_transform(apply_output, ucpe_hidden_states.transpose(1, 2)).transpose(1, 2)
            else:
                ucpe_hidden_states = apply_output(ucpe_hidden_states.transpose(1, 2)).transpose(1, 2)
            ucpe_hidden_states = ucpe_hidden_states.flatten(2, 3).to(norm_hidden_states.dtype)
            attn_hidden_states = attn_hidden_states + self.ucpe_out_proj(ucpe_hidden_states)
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        # 1.2. Audio Self-Attention
        audio_ada_params = self.get_mod_params(self.audio_scale_shift_table, temb_audio, batch_size)
        audio_shift_msa, audio_scale_msa, audio_gate_msa, audio_shift_mlp, audio_scale_mlp, audio_gate_mlp = (
            audio_ada_params[:6]
        )
        if self.audio_cross_attn_adaln:
            audio_shift_text_q, audio_scale_text_q, audio_gate_text_q = audio_ada_params[6:9]

        norm_audio_hidden_states = self.audio_norm1(audio_hidden_states)
        norm_audio_hidden_states = norm_audio_hidden_states * (1 + audio_scale_msa) + audio_shift_msa

        audio_self_attn_args = {
            "hidden_states": norm_audio_hidden_states,
            "encoder_hidden_states": None,
            "query_rotary_emb": audio_rotary_emb,
            "attention_mask": audio_self_attention_mask,
            "kv_cache": kv_cache.get("audio_self") if kv_cache is not None else None,
            "kv_cache_start": current_audio_token_start,
        }
        if self.perturbed_attn:
            audio_self_attn_args["perturbation_mask"] = perturbation_mask
            audio_self_attn_args["all_perturbed"] = all_perturbed

        attn_audio_hidden_states = self.audio_attn1(**audio_self_attn_args)
        audio_hidden_states = audio_hidden_states + attn_audio_hidden_states * audio_gate_msa

        # 2. Video and Audio Cross-Attention with the text embeddings (Q: Video or Audio; K,V: Text)
        if self.cross_attn_adaln:
            # `temb_prompt`/`temb_prompt_audio` are `None` when `use_prompt_adaln_single=False` (KV-cacheable
            # cross-attention): the prompt-side scale/shift is then timestep-independent, so only the static
            # per-layer table is used.
            if temb_prompt is not None:
                shift_text_kv, scale_text_kv = self.get_mod_params(
                    self.prompt_scale_shift_table, temb_prompt, batch_size
                )
            else:
                shift_text_kv, scale_text_kv = (
                    self.prompt_scale_shift_table[None, None]
                    .to(device=hidden_states.device, dtype=hidden_states.dtype)
                    .unbind(dim=2)
                )

            if temb_prompt_audio is not None:
                audio_shift_text_kv, audio_scale_text_kv = self.get_mod_params(
                    self.audio_prompt_scale_shift_table, temb_prompt_audio, batch_size
                )
            else:
                audio_shift_text_kv, audio_scale_text_kv = (
                    self.audio_prompt_scale_shift_table[None, None]
                    .to(device=audio_hidden_states.device, dtype=audio_hidden_states.dtype)
                    .unbind(dim=2)
                )

        # 2.1. Video-Text Cross-Attention (Q: Video; K,V: Text)
        norm_hidden_states = self.norm2(hidden_states)
        if self.video_cross_attn_adaln:
            norm_hidden_states = norm_hidden_states * (1 + scale_text_q) + shift_text_q
        if self.cross_attn_adaln:
            encoder_hidden_states = encoder_hidden_states * (1 + scale_text_kv) + shift_text_kv

        attn_hidden_states = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_rotary_emb=None,
            attention_mask=encoder_attention_mask,
            crossattn_cache=kv_cache.get("video_text") if kv_cache is not None and temb_prompt is None else None,
        )
        if self.video_cross_attn_adaln:
            attn_hidden_states = attn_hidden_states * gate_text_q
        hidden_states = hidden_states + attn_hidden_states

        # 2.2. Audio-Text Cross-Attention
        norm_audio_hidden_states = self.audio_norm2(audio_hidden_states)
        if self.audio_cross_attn_adaln:
            norm_audio_hidden_states = norm_audio_hidden_states * (1 + audio_scale_text_q) + audio_shift_text_q
        if self.cross_attn_adaln:
            audio_encoder_hidden_states = audio_encoder_hidden_states * (1 + audio_scale_text_kv) + audio_shift_text_kv

        attn_audio_hidden_states = self.audio_attn2(
            norm_audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            query_rotary_emb=None,
            attention_mask=audio_encoder_attention_mask,
            crossattn_cache=kv_cache.get("audio_text") if kv_cache is not None and temb_prompt_audio is None else None,
        )
        if self.audio_cross_attn_adaln:
            attn_audio_hidden_states = attn_audio_hidden_states * audio_gate_text_q
        audio_hidden_states = audio_hidden_states + attn_audio_hidden_states

        # 3. Audio-to-Video (a2v) and Video-to-Audio (v2a) Cross-Attention
        if use_a2v_cross_attention or use_v2a_cross_attention:
            norm_hidden_states = self.audio_to_video_norm(hidden_states)
            norm_audio_hidden_states = self.video_to_audio_norm(audio_hidden_states)

            # 3.1. Combine global and per-layer cross attention modulation parameters
            # Video
            video_per_layer_ca_scale_shift = self.video_a2v_cross_attn_scale_shift_table[:4, :]
            video_per_layer_ca_gate = self.video_a2v_cross_attn_scale_shift_table[4:, :]

            video_ca_ada_params = self.get_mod_params(video_per_layer_ca_scale_shift, temb_ca_scale_shift, batch_size)
            video_ca_gate_param = self.get_mod_params(video_per_layer_ca_gate, temb_ca_gate, batch_size)

            video_a2v_ca_scale, video_a2v_ca_shift, video_v2a_ca_scale, video_v2a_ca_shift = video_ca_ada_params
            a2v_gate = video_ca_gate_param[0].squeeze(2)

            # Audio
            audio_per_layer_ca_scale_shift = self.audio_a2v_cross_attn_scale_shift_table[:4, :]
            audio_per_layer_ca_gate = self.audio_a2v_cross_attn_scale_shift_table[4:, :]

            audio_ca_ada_params = self.get_mod_params(
                audio_per_layer_ca_scale_shift, temb_ca_audio_scale_shift, batch_size
            )
            audio_ca_gate_param = self.get_mod_params(audio_per_layer_ca_gate, temb_ca_audio_gate, batch_size)

            audio_a2v_ca_scale, audio_a2v_ca_shift, audio_v2a_ca_scale, audio_v2a_ca_shift = audio_ca_ada_params
            v2a_gate = audio_ca_gate_param[0].squeeze(2)

            # 3.2. Audio-to-Video Cross Attention: Q: Video; K,V: Audio
            if use_a2v_cross_attention:
                mod_norm_hidden_states = norm_hidden_states * (
                    1 + video_a2v_ca_scale.squeeze(2)
                ) + video_a2v_ca_shift.squeeze(2)
                mod_norm_audio_hidden_states = norm_audio_hidden_states * (
                    1 + audio_a2v_ca_scale.squeeze(2)
                ) + audio_a2v_ca_shift.squeeze(2)

                a2v_attn_hidden_states = self.audio_to_video_attn(
                    mod_norm_hidden_states,
                    encoder_hidden_states=mod_norm_audio_hidden_states,
                    query_rotary_emb=ca_video_rotary_emb,
                    key_rotary_emb=ca_audio_rotary_emb,
                    attention_mask=a2v_cross_attention_mask,
                    kv_cache=kv_cache.get("a2v") if kv_cache is not None else None,
                    kv_cache_start=current_audio_token_start,
                )

                hidden_states = hidden_states + a2v_gate * a2v_attn_hidden_states

            # 3.3. Video-to-Audio Cross Attention: Q: Audio; K,V: Video
            if use_v2a_cross_attention:
                mod_norm_hidden_states = norm_hidden_states * (
                    1 + video_v2a_ca_scale.squeeze(2)
                ) + video_v2a_ca_shift.squeeze(2)
                mod_norm_audio_hidden_states = norm_audio_hidden_states * (
                    1 + audio_v2a_ca_scale.squeeze(2)
                ) + audio_v2a_ca_shift.squeeze(2)

                v2a_attn_hidden_states = self.video_to_audio_attn(
                    mod_norm_audio_hidden_states,
                    encoder_hidden_states=mod_norm_hidden_states,
                    query_rotary_emb=ca_audio_rotary_emb,
                    key_rotary_emb=ca_video_rotary_emb,
                    attention_mask=v2a_cross_attention_mask,
                    kv_cache=kv_cache.get("v2a") if kv_cache is not None else None,
                    kv_cache_start=current_video_token_start,
                )

                audio_hidden_states = audio_hidden_states + v2a_gate * v2a_attn_hidden_states

        # 4. Feedforward
        norm_hidden_states = self.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp

        norm_audio_hidden_states = self.audio_norm3(audio_hidden_states) * (1 + audio_scale_mlp) + audio_shift_mlp
        audio_ff_output = self.audio_ff(norm_audio_hidden_states)
        audio_hidden_states = audio_hidden_states + audio_ff_output * audio_gate_mlp

        return hidden_states, audio_hidden_states


class EchoWMAudioVideoRotaryPosEmbed(nn.Module):
    """
    Video and audio rotary positional embeddings (RoPE) for the LTX-2.0 model.

    Args:
        causal_offset (`int`, *optional*, defaults to `1`):
            Offset in the temporal axis for causal VAE modeling. This is typically 1 (for causal modeling where the VAE
            treats the very first frame differently), but could also be 0 (for non-causal modeling).
    """

    def __init__(
        self,
        dim: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        scale_factors: tuple[int, ...] = (8, 32, 32),
        theta: float = 10000.0,
        causal_offset: int = 1,
        modality: str = "video",
        double_precision: bool = True,
        rope_type: str = "interleaved",
        num_attention_heads: int = 32,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t

        if rope_type not in ["interleaved", "split"]:
            raise ValueError(f"{rope_type=} not supported. Choose between 'interleaved' and 'split'.")
        self.rope_type = rope_type

        self.base_num_frames = base_num_frames
        self.num_attention_heads = num_attention_heads

        # Video-specific
        self.base_height = base_height
        self.base_width = base_width

        # Audio-specific
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.audio_latents_per_second = float(sampling_rate) / float(hop_length) / float(scale_factors[0])

        self.scale_factors = scale_factors
        self.theta = theta
        self.causal_offset = causal_offset

        self.modality = modality
        if self.modality not in ["video", "audio"]:
            raise ValueError(f"Modality {modality} is not supported. Supported modalities are `video` and `audio`.")
        self.double_precision = double_precision

    def prepare_video_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        fps: float = 24.0,
    ) -> torch.Tensor:
        """
        Create per-dimension bounds [inclusive start, exclusive end) for each patch with respect to the original pixel
        space video grid (num_frames, height, width). This will ultimately have shape (batch_size, 3, num_patches, 2)
        where
            - axis 1 (size 3) enumerates (frame, height, width) dimensions (e.g. idx 0 corresponds to frames)
            - axis 3 (size 2) stores `[start, end)` indices within each dimension

        Args:
            batch_size (`int`):
                Batch size of the video latents.
            num_frames (`int`):
                Number of latent frames in the video latents.
            height (`int`):
                Latent height of the video latents.
            width (`int`):
                Latent width of the video latents.
            device (`torch.device`):
                Device on which to create the video grid.

        Returns:
            `torch.Tensor`:
                Per-dimension patch boundaries tensor of shape [batch_size, 3, num_patches, 2].
        """

        # 1. Generate grid coordinates for each spatiotemporal dimension (frames, height, width)
        # Always compute rope in fp32
        grid_f = torch.arange(start=0, end=num_frames, step=self.patch_size_t, dtype=torch.float32, device=device)
        grid_h = torch.arange(start=0, end=height, step=self.patch_size, dtype=torch.float32, device=device)
        grid_w = torch.arange(start=0, end=width, step=self.patch_size, dtype=torch.float32, device=device)
        # indexing='ij' ensures that the dimensions are kept in order as (frames, height, width)
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)  # [3, N_F, N_H, N_W], where e.g. N_F is the number of temporal patches

        # 2. Get the patch boundaries with respect to the latent video grid
        patch_size = (self.patch_size_t, self.patch_size, self.patch_size)
        patch_size_delta = torch.tensor(patch_size, dtype=grid.dtype, device=grid.device)
        patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)

        # Combine the start (grid) and end (patch_ends) coordinates along new trailing dimension
        latent_coords = torch.stack([grid, patch_ends], dim=-1)  # [3, N_F, N_H, N_W, 2]
        # Reshape to (batch_size, 3, num_patches, 2)
        latent_coords = latent_coords.flatten(1, 3)
        latent_coords = latent_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # 3. Calculate the pixel space patch boundaries from the latent boundaries.
        scale_tensor = torch.tensor(self.scale_factors, device=latent_coords.device)
        # Broadcast the VAE scale factors such that they are compatible with latent_coords's shape
        broadcast_shape = [1] * latent_coords.ndim
        broadcast_shape[1] = -1  # This is the (frame, height, width) dim
        # Apply per-axis scaling to convert latent coordinates to pixel space coordinates
        pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)

        # As the VAE temporal stride for the first frame is 1 instead of self.vae_scale_factors[0], we need to shift
        # and clamp to keep the first-frame timestamps causal and non-negative.
        pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + self.causal_offset - self.scale_factors[0]).clamp(min=0)

        # Scale the temporal coordinates by the video FPS
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps

        return pixel_coords

    def prepare_audio_coords(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        shift: int = 0,
    ) -> torch.Tensor:
        """
        Create per-dimension bounds [inclusive start, exclusive end) of start and end timestamps for each latent frame.
        This will ultimately have shape (batch_size, 3, num_patches, 2) where
            - axis 1 (size 1) represents the temporal dimension
            - axis 3 (size 2) stores `[start, end)` indices within each dimension

        Args:
            batch_size (`int`):
                Batch size of the audio latents.
            num_frames (`int`):
                Number of latent frames in the audio latents.
            device (`torch.device`):
                Device on which to create the audio grid.
            shift (`int`, *optional*, defaults to `0`):
                Offset on the latent indices. Different shift values correspond to different overlapping windows with
                respect to the same underlying latent grid.

        Returns:
            `torch.Tensor`:
                Per-dimension patch boundaries tensor of shape [batch_size, 1, num_patches, 2].
        """

        # 1. Generate coordinates in the frame (time) dimension.
        # Always compute rope in fp32
        grid_f = torch.arange(
            start=shift, end=num_frames + shift, step=self.patch_size_t, dtype=torch.float32, device=device
        )

        # 2. Calculate start timestamps in seconds with respect to the original spectrogram grid
        audio_scale_factor = self.scale_factors[0]
        # Scale back to mel spectrogram space
        grid_start_mel = grid_f * audio_scale_factor
        # Handle first frame causal offset, ensuring non-negative timestamps
        grid_start_mel = (grid_start_mel + self.causal_offset - audio_scale_factor).clip(min=0)
        # Convert mel bins back into seconds
        grid_start_s = grid_start_mel * self.hop_length / self.sampling_rate

        # 3. Calculate start timestamps in seconds with respect to the original spectrogram grid
        grid_end_mel = (grid_f + self.patch_size_t) * audio_scale_factor
        grid_end_mel = (grid_end_mel + self.causal_offset - audio_scale_factor).clip(min=0)
        grid_end_s = grid_end_mel * self.hop_length / self.sampling_rate

        audio_coords = torch.stack([grid_start_s, grid_end_s], dim=-1)  # [num_patches, 2]
        audio_coords = audio_coords.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_patches, 2]
        audio_coords = audio_coords.unsqueeze(1)  # [batch_size, 1, num_patches, 2]
        return audio_coords

    def prepare_coords(self, *args, **kwargs):
        if self.modality == "video":
            return self.prepare_video_coords(*args, **kwargs)
        elif self.modality == "audio":
            return self.prepare_audio_coords(*args, **kwargs)

    def forward(
        self, coords: torch.Tensor, device: str | torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or coords.device

        # Number of spatiotemporal dimensions (3 for video, 1 (temporal) for audio and cross attn)
        num_pos_dims = coords.shape[1]

        # 1. If the coords are patch boundaries [start, end), use the midpoint of these boundaries as the patch
        # position index
        if coords.ndim == 4:
            coords_start, coords_end = coords.chunk(2, dim=-1)
            coords = (coords_start + coords_end) / 2.0
            coords = coords.squeeze(-1)  # [B, num_pos_dims, num_patches]

        # 2. Get coordinates as a fraction of the base data shape
        if self.modality == "video":
            max_positions = (self.base_num_frames, self.base_height, self.base_width)
        elif self.modality == "audio":
            max_positions = (self.base_num_frames,)
        # [B, num_pos_dims, num_patches] --> [B, num_patches, num_pos_dims]
        grid = torch.stack([coords[:, i] / max_positions[i] for i in range(num_pos_dims)], dim=-1).to(device)
        # Number of spatiotemporal dimensions (3 for video, 1 for audio and cross attn) times 2 for cos, sin
        num_rope_elems = num_pos_dims * 2

        # 3. Create a 1D grid of frequencies for RoPE
        freqs_dtype = maybe_adjust_dtype_for_device(torch.float64 if self.double_precision else torch.float32, device)
        pow_indices = torch.pow(
            self.theta,
            torch.linspace(start=0.0, end=1.0, steps=self.dim // num_rope_elems, dtype=freqs_dtype, device=device),
        )
        freqs = (pow_indices * torch.pi / 2.0).to(dtype=torch.float32)

        # 4. Tensor-vector outer product between pos ids tensor of shape (B, 3, num_patches) and freqs vector of shape
        # (self.dim // num_elems,)
        freqs = (grid.unsqueeze(-1) * 2 - 1) * freqs  # [B, num_patches, num_pos_dims, self.dim // num_elems]
        freqs = freqs.transpose(-1, -2).flatten(2)  # [B, num_patches, self.dim // 2]

        # 5. Get real, interleaved (cos, sin) frequencies, padded to self.dim
        # TODO: consider implementing this as a utility and reuse in `connectors.py`.
        # src/diffusers/pipelines/ltx2/connectors.py
        if self.rope_type == "interleaved":
            cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
            sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

            if self.dim % num_rope_elems != 0:
                cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % num_rope_elems])
                sin_padding = torch.zeros_like(cos_freqs[:, :, : self.dim % num_rope_elems])
                cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
                sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        elif self.rope_type == "split":
            expected_freqs = self.dim // 2
            current_freqs = freqs.shape[-1]
            pad_size = expected_freqs - current_freqs
            cos_freq = freqs.cos()
            sin_freq = freqs.sin()

            if pad_size != 0:
                cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
                sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])

                cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
                sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)

            # Reshape freqs to be compatible with multi-head attention
            b = cos_freq.shape[0]
            t = cos_freq.shape[1]

            cos_freq = cos_freq.reshape(b, t, self.num_attention_heads, -1)
            sin_freq = sin_freq.reshape(b, t, self.num_attention_heads, -1)

            cos_freqs = torch.swapaxes(cos_freq, 1, 2)  # (B,H,T,D//2)
            sin_freqs = torch.swapaxes(sin_freq, 1, 2)  # (B,H,T,D//2)

        return cos_freqs, sin_freqs


class EchoWMTransformer3DModel(ModelMixin, ConfigMixin, AttentionMixin, PeftAdapterMixin, CacheMixin):
    r"""
    A joint video-audio Transformer used by Echo-WM. It is derived from LTX-2 and adds UCPE camera attention and the
    bounded KV caches required by Echo-WM Flash.

    Args:
        in_channels (`int`, defaults to `128`):
            The number of channels in the input.
        out_channels (`int`, defaults to `128`):
            The number of channels in the output.
        patch_size (`int`, defaults to `1`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the temporal patches to use in the patch embedding layer.
        num_attention_heads (`int`, defaults to `32`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        cross_attention_dim (`int`, defaults to `4096`):
            The number of channels for cross attention heads.
        num_layers (`int`, defaults to `48`):
            The number of layers of Transformer blocks to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        qk_norm (`str`, defaults to `"rms_norm_across_heads"`):
            The normalization layer to use.
        ff_bias (`bool`, defaults to `True`):
            Whether the video feed-forward layer's linear layers include a bias term. `False` for LTX-2.5.
        audio_ff_bias (`bool`, defaults to `True`):
            Whether the audio feed-forward layer's linear layers include a bias term.
        use_prompt_adaln_single (`bool`, defaults to `True`):
            Whether the prompt's cross-attention Key/Value modulation is timestep-dependent. When `False`, it uses a
            fixed per-layer table instead, making the cross-attention Key/Value values cacheable across denoising steps
            for a given prompt.
        use_keyframes_abs_pos_embedding (`bool`, defaults to `False`):
            Whether to store a learned `(1, inner_dim)` absolute-position embedding for generated-keyframe tokens
            (LTX-2.5). When `True`, tokens selected by `video_keyframes_mask` receive this embedding. The argument is
            optional; omitting it leaves the distilled forward path unchanged.
        ucpe_block_indices (`tuple[int, ...]`, *optional*):
            Transformer block indices that receive Echo-WM's UCPE camera-attention branch. `None` disables UCPE and
            leaves the original LTX-2 architecture unchanged.
        ucpe_attention_dim (`int`, defaults to `1024`):
            Inner dimension of each enabled UCPE branch.
        ucpe_num_attention_heads (`int`, defaults to `8`):
            Number of attention heads in each enabled UCPE branch.
        ucpe_patches_x (`int`, defaults to `40`):
            Width of the per-frame latent patch grid used by UCPE.
        ucpe_patches_y (`int`, defaults to `22`):
            Height of the per-frame latent patch grid used by UCPE.
        ucpe_image_width (`int`, defaults to `1280`):
            Pixel canvas width used to normalize camera intrinsics.
        ucpe_image_height (`int`, defaults to `704`):
            Pixel canvas height used to normalize camera intrinsics.
        ucpe_freq_base (`float`, defaults to `100.0`):
            Frequency base for the UCPE spatial rotary embeddings.
        ucpe_freq_scale (`float`, defaults to `1.0`):
            Frequency scale for the UCPE spatial rotary embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["EchoWMVideoTransformerBlock"]
    _skip_layerwise_casting_patterns = ["norm"]
    _repeated_blocks = ["EchoWMVideoTransformerBlock"]
    _skip_keys = ["kv_caches"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,  # Video Arguments
        out_channels: int | None = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        cross_attention_dim: int = 4096,
        vae_scale_factors: tuple[int, int, int] = (8, 32, 32),
        pos_embed_max_pos: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        gated_attn: bool = False,
        cross_attn_mod: bool = False,
        audio_in_channels: int = 128,  # Audio Arguments
        audio_out_channels: int | None = 128,
        audio_patch_size: int = 1,
        audio_patch_size_t: int = 1,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_cross_attention_dim: int = 2048,
        audio_scale_factor: int = 4,
        audio_pos_embed_max_pos: int = 20,
        audio_sampling_rate: int = 16000,
        audio_hop_length: int = 160,
        audio_gated_attn: bool = False,
        audio_cross_attn_mod: bool = False,
        num_layers: int = 48,  # Shared arguments
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 3840,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        causal_offset: int = 1,
        timestep_scale_multiplier: int = 1000,
        cross_attn_timestep_scale_multiplier: int = 1000,
        rope_type: str = "interleaved",
        use_prompt_embeddings=True,
        perturbed_attn: bool = False,
        ff_bias: bool = True,
        audio_ff_bias: bool = True,
        use_prompt_adaln_single: bool = True,
        use_keyframes_abs_pos_embedding: bool = False,
        ucpe_block_indices: tuple[int, ...] | None = None,
        ucpe_attention_dim: int = 1024,
        ucpe_num_attention_heads: int = 8,
        ucpe_patches_x: int = 40,
        ucpe_patches_y: int = 22,
        ucpe_image_width: int = 1280,
        ucpe_image_height: int = 704,
        ucpe_freq_base: float = 100.0,
        ucpe_freq_scale: float = 1.0,
    ) -> None:
        super().__init__()

        ucpe_block_indices = tuple(ucpe_block_indices or ())
        invalid_ucpe_blocks = [index for index in ucpe_block_indices if index < 0 or index >= num_layers]
        if invalid_ucpe_blocks:
            raise ValueError(
                f"`ucpe_block_indices` contains indices outside [0, {num_layers}), got {invalid_ucpe_blocks}."
            )

        out_channels = out_channels or in_channels
        audio_out_channels = audio_out_channels or audio_in_channels
        inner_dim = num_attention_heads * attention_head_dim
        audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim

        # 1. Patchification input projections
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.audio_proj_in = nn.Linear(audio_in_channels, audio_inner_dim)

        # Marks single-pixel-frame keyframe tokens. Zero-initialized in the reference; unused by the regular
        # distilled forward until a dedicated keyframes pipeline applies it after `proj_in`.
        if use_keyframes_abs_pos_embedding:
            self.keyframes_abs_pos_embedding = nn.Parameter(torch.zeros(1, inner_dim))

        # 2. Prompt embeddings
        if use_prompt_embeddings:
            # LTX-2.0; LTX-2.3 uses per-modality feature projections in the connector instead
            self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)
            self.audio_caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels, hidden_size=audio_inner_dim
            )

        # 3. Timestep Modulation Params and Embedding
        self.prompt_modulation = cross_attn_mod or audio_cross_attn_mod  # used by LTX-2.3

        # 3.1. Global Timestep Modulation Parameters (except for cross-attention) and timestep + size embedding
        # time_embed and audio_time_embed calculate both the timestep embedding and (global) modulation parameters
        video_time_emb_mod_params = 9 if cross_attn_mod else 6
        audio_time_emb_mod_params = 9 if audio_cross_attn_mod else 6
        self.time_embed = EchoWMAdaLayerNormSingle(
            inner_dim, num_mod_params=video_time_emb_mod_params, use_additional_conditions=False
        )
        self.audio_time_embed = EchoWMAdaLayerNormSingle(
            audio_inner_dim, num_mod_params=audio_time_emb_mod_params, use_additional_conditions=False
        )

        # 3.2. Global Cross Attention Modulation Parameters
        # Used in the audio-to-video and video-to-audio cross attention layers as a global set of modulation params,
        # which are then further modified by per-block modulaton params in each transformer block.
        # There are 2 sets of scale/shift parameters for each modality, 1 each for audio-to-video (a2v) and
        # video-to-audio (v2a) cross attention
        self.av_cross_attn_video_scale_shift = EchoWMAdaLayerNormSingle(
            inner_dim, num_mod_params=4, use_additional_conditions=False
        )
        self.av_cross_attn_audio_scale_shift = EchoWMAdaLayerNormSingle(
            audio_inner_dim, num_mod_params=4, use_additional_conditions=False
        )
        # Gate param for audio-to-video (a2v) cross attn (where the video is the queries (Q) and the audio is the keys
        # and values (KV))
        self.av_cross_attn_video_a2v_gate = EchoWMAdaLayerNormSingle(
            inner_dim, num_mod_params=1, use_additional_conditions=False
        )
        # Gate param for video-to-audio (v2a) cross attn (where the audio is the queries (Q) and the video is the keys
        # and values (KV))
        self.av_cross_attn_audio_v2a_gate = EchoWMAdaLayerNormSingle(
            audio_inner_dim, num_mod_params=1, use_additional_conditions=False
        )

        # 3.3. Output Layer Scale/Shift Modulation parameters
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.audio_scale_shift_table = nn.Parameter(torch.randn(2, audio_inner_dim) / audio_inner_dim**0.5)

        # 3.4. Prompt Scale/Shift Modulation parameters (LTX-2.3)
        # When `use_prompt_adaln_single=False` (LTX-2.5 KV-cacheable cross-attention), this MLP is dropped so the
        # cross-attention K/V modulation becomes timestep-independent (static per-layer table only).
        if self.prompt_modulation and use_prompt_adaln_single:
            self.prompt_adaln = EchoWMAdaLayerNormSingle(inner_dim, num_mod_params=2, use_additional_conditions=False)
            self.audio_prompt_adaln = EchoWMAdaLayerNormSingle(
                audio_inner_dim, num_mod_params=2, use_additional_conditions=False
            )

        # 4. Rotary Positional Embeddings (RoPE)
        # Self-Attention
        self.rope = EchoWMAudioVideoRotaryPosEmbed(
            dim=inner_dim,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            base_num_frames=pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            scale_factors=vae_scale_factors,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )
        self.audio_rope = EchoWMAudioVideoRotaryPosEmbed(
            dim=audio_inner_dim,
            patch_size=audio_patch_size,
            patch_size_t=audio_patch_size_t,
            base_num_frames=audio_pos_embed_max_pos,
            sampling_rate=audio_sampling_rate,
            hop_length=audio_hop_length,
            scale_factors=[audio_scale_factor],
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=audio_num_attention_heads,
        )

        # Audio-to-Video, Video-to-Audio Cross-Attention
        cross_attn_pos_embed_max_pos = max(pos_embed_max_pos, audio_pos_embed_max_pos)
        self.cross_attn_rope = EchoWMAudioVideoRotaryPosEmbed(
            dim=audio_cross_attention_dim,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )
        self.cross_attn_audio_rope = EchoWMAudioVideoRotaryPosEmbed(
            dim=audio_cross_attention_dim,
            patch_size=audio_patch_size,
            patch_size_t=audio_patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            sampling_rate=audio_sampling_rate,
            hop_length=audio_hop_length,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=audio_num_attention_heads,
        )

        # 5. Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                EchoWMVideoTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    audio_dim=audio_inner_dim,
                    audio_num_attention_heads=audio_num_attention_heads,
                    audio_attention_head_dim=audio_attention_head_dim,
                    audio_cross_attention_dim=audio_cross_attention_dim,
                    video_gated_attn=gated_attn,
                    video_cross_attn_adaln=cross_attn_mod,
                    audio_gated_attn=audio_gated_attn,
                    audio_cross_attn_adaln=audio_cross_attn_mod,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                    rope_type=rope_type,
                    perturbed_attn=perturbed_attn,
                    ff_bias=ff_bias,
                    audio_ff_bias=audio_ff_bias,
                    ucpe_attention_dim=ucpe_attention_dim if block_index in ucpe_block_indices else None,
                    ucpe_num_attention_heads=ucpe_num_attention_heads,
                    ucpe_patches_x=ucpe_patches_x,
                    ucpe_patches_y=ucpe_patches_y,
                    ucpe_image_width=ucpe_image_width,
                    ucpe_image_height=ucpe_image_height,
                    ucpe_freq_base=ucpe_freq_base,
                    ucpe_freq_scale=ucpe_freq_scale,
                )
                for block_index in range(num_layers)
            ]
        )

        # 6. Output layers
        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)

        self.audio_norm_out = nn.LayerNorm(audio_inner_dim, eps=1e-6, elementwise_affine=False)
        self.audio_proj_out = nn.Linear(audio_inner_dim, audio_out_channels)

        self.gradient_checkpointing = False

    def init_echo_wm_causal_caches(
        self,
        *,
        video_local_tokens: int,
        video_sink_tokens: int,
        audio_local_tokens: int,
        audio_sink_tokens: int,
    ) -> EchoWMKVCache:
        r"""Create the bounded per-layer KV caches used by Echo-WM Flash."""

        def make_cache(local_size: int, sink_size: int) -> dict[str, Any]:
            if local_size <= 0 or not 0 <= sink_size < local_size:
                raise ValueError(f"Expected `0 <= sink_size < local_size`, got {sink_size} and {local_size}.")
            return {"key": None, "value": None, "positions": None, "local_size": local_size, "sink_size": sink_size}

        caches = []
        for block in self.transformer_blocks:
            layer = {
                "video_self": make_cache(video_local_tokens, video_sink_tokens),
                "video_text": {"key": None, "value": None},
                "audio_self": make_cache(audio_local_tokens, audio_sink_tokens),
                "audio_text": {"key": None, "value": None},
                "a2v": make_cache(audio_local_tokens, audio_sink_tokens),
                "v2a": make_cache(video_local_tokens, video_sink_tokens),
            }
            if block.ucpe_enabled:
                layer["video_ucpe"] = make_cache(video_local_tokens, video_sink_tokens)
            caches.append(layer)
        return EchoWMKVCache(caches)

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        audio_timestep: torch.LongTensor | None = None,
        sigma: torch.Tensor | None = None,
        audio_sigma: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        audio_encoder_attention_mask: torch.Tensor | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float = 24.0,
        audio_num_frames: int | None = None,
        video_coords: torch.Tensor | None = None,
        audio_coords: torch.Tensor | None = None,
        isolate_modalities: bool = False,
        spatio_temporal_guidance_blocks: list[int] | None = None,
        perturbation_mask: torch.Tensor | None = None,
        use_cross_timestep: bool = False,
        attention_kwargs: dict[str, Any] | None = None,
        video_self_attention_mask: torch.Tensor | None = None,
        video_keyframes_mask: torch.Tensor | None = None,
        ucpe_viewmats: torch.Tensor | None = None,
        ucpe_intrinsics: torch.Tensor | None = None,
        kv_caches: EchoWMKVCache | None = None,
        current_video_token_start: int = 0,
        current_audio_token_start: int = 0,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for the Echo-WM audiovisual transformer.

        Args:
            hidden_states (`torch.Tensor`):
                Input patchified video latents of shape `(batch_size, num_video_tokens, in_channels)`.
            audio_hidden_states (`torch.Tensor`):
                Input patchified audio latents of shape `(batch_size, num_audio_tokens, audio_in_channels)`.
            encoder_hidden_states (`torch.Tensor`):
                Input video text embeddings of shape `(batch_size, text_seq_len, self.config.caption_channels)`.
            audio_encoder_hidden_states (`torch.Tensor`):
                Input audio text embeddings of shape `(batch_size, text_seq_len, self.config.caption_channels)`.
            timestep (`torch.Tensor`):
                Input timestep of shape `(batch_size, num_video_tokens)`. These should already be scaled by
                `self.config.timestep_scale_multiplier`.
            audio_timestep (`torch.Tensor`, *optional*):
                Input timestep of shape `(batch_size,)` or `(batch_size, num_audio_tokens)` for audio modulation
                params. This is only used by certain pipelines such as the I2V pipeline.
            sigma (`torch.Tensor`, *optional*):
                Input scaled timestep of shape (batch_size,). Used for video prompt cross attention modulation in
                models such as LTX-2.3.
            audio_sigma (`torch.Tensor`, *optional*):
                Input scaled timestep of shape (batch_size,). Used for audio prompt cross attention modulation in
                models such as LTX-2.3. If `sigma` is supplied but `audio_sigma` is not, `audio_sigma` will be set to
                the provided `sigma` value.
            encoder_attention_mask (`torch.Tensor`, *optional*):
                Optional multiplicative text attention mask of shape `(batch_size, text_seq_len)`.
            audio_encoder_attention_mask (`torch.Tensor`, *optional*):
                Optional multiplicative text attention mask of shape `(batch_size, text_seq_len)` for audio modeling.
            num_frames (`int`, *optional*):
                The number of latent video frames. Used if calculating the video coordinates for RoPE.
            height (`int`, *optional*):
                The latent video height. Used if calculating the video coordinates for RoPE.
            width (`int`, *optional*):
                The latent video width. Used if calculating the video coordinates for RoPE.
            fps: (`float`, *optional*, defaults to `24.0`):
                The desired frames per second of the generated video. Used if calculating the video coordinates for
                RoPE.
            audio_num_frames: (`int`, *optional*):
                The number of latent audio frames. Used if calculating the audio coordinates for RoPE.
            video_coords (`torch.Tensor`, *optional*):
                The video coordinates to be used when calculating the rotary positional embeddings (RoPE) of shape
                `(batch_size, 3, num_video_tokens, 2)`. If not supplied, this will be calculated inside `forward`.
            audio_coords (`torch.Tensor`, *optional*):
                The audio coordinates to be used when calculating the rotary positional embeddings (RoPE) of shape
                `(batch_size, 1, num_audio_tokens, 2)`. If not supplied, this will be calculated inside `forward`.
            isolate_modalities (`bool`, *optional*, defaults to `False`):
                Whether to isolate each modality by turning off cross-modality (audio-to-video and video-to-audio)
                cross attention (for all blocks). Use for modality guidance in LTX-2.3.
            spatio_temporal_guidance_blocks (`list[int]`, *optional*, defaults to `None`):
                The transformer block indices at which to apply spatio-temporal guidance (STG), which shortcuts the
                self-attention operations by simply using the values rather than the full scaled dot-product attention
                (SDPA) operation. If `None` or empty, STG will not be applied to any block.
            perturbation_mask (`torch.Tensor`, *optional*):
                Perturbation mask for STG of shape `(batch_size,)` or `(batch_size, 1, 1)`. Should be 0 at batch
                elements where STG should be applied and 1 elsewhere. If STG is being used but `peturbation_mask` is
                not supplied, will default to applying STG (perturbing) all batch elements.
            use_cross_timestep (`bool` *optional*, defaults to `False`):
                Whether to use the cross modality (audio is the cross modality of video, and vice versa) sigma when
                calculating the cross attention modulation parameters. `True` is the newer (e.g. LTX-2.3) behavior;
                `False` is the legacy LTX-2.0 behavior.
            attention_kwargs (`dict[str, Any]`, *optional*):
                Optional dict of keyword args to be passed to the attention processor.
            video_self_attention_mask (`torch.Tensor`, *optional*):
                Optional multiplicative self-attention mask of shape `(batch_size, num_video_tokens, num_video_tokens)`
                applied to the video self-attention in each transformer block. Values in `[0, 1]` where `1` means full
                attention and `0` means masked. Used e.g. by the IC-LoRA pipeline to control attention strength between
                noisy tokens and appended reference tokens. Audio self-attention is not affected.
            video_keyframes_mask (`torch.Tensor`, *optional*):
                Optional per-token marker of shape `(batch_size, num_video_tokens, 1)`, non-zero on video tokens whose
                latent frame encodes a single pixel frame. Those tokens receive `keyframes_abs_pos_embedding`. Ignored
                when the model was built without `use_keyframes_abs_pos_embedding`.
            ucpe_viewmats (`torch.Tensor`, *optional*):
                Camera-to-world matrices of shape `(batch_size, num_latent_frames, 4, 4)` for Echo-WM camera
                conditioning. Must be provided together with `ucpe_intrinsics` when UCPE blocks are configured.
            ucpe_intrinsics (`torch.Tensor`, *optional*):
                Pixel-space camera intrinsics of shape `(batch_size, num_latent_frames, 3, 3)` for Echo-WM camera
                conditioning.
            kv_caches (`EchoWMKVCache`, *optional*):
                Per-layer bounded sink-plus-FIFO caches used by Echo-WM Flash autoregressive inference.
            current_video_token_start (`int`, *optional*, defaults to `0`):
                Global token offset of the current video chunk when updating `kv_caches`.
            current_audio_token_start (`int`, *optional*, defaults to `0`):
                Global token offset of the current audio chunk when updating `kv_caches`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a dict-like structured output of type `AudioVisualModelOutput` or a tuple.

        Returns:
            `AudioVisualModelOutput` or `tuple`:
                If `return_dict` is `True`, returns a structured output of type `AudioVisualModelOutput`, otherwise a
                `tuple` is returned where the first element is the denoised video latent patch sequence and the second
                element is the denoised audio latent patch sequence.
        """
        # Determine timestep for audio.
        audio_timestep = audio_timestep if audio_timestep is not None else timestep
        audio_sigma = audio_sigma if audio_sigma is not None else sigma

        if (ucpe_viewmats is None) != (ucpe_intrinsics is None):
            raise ValueError("`ucpe_viewmats` and `ucpe_intrinsics` must be provided together.")
        if ucpe_viewmats is not None and not self.config.ucpe_block_indices:
            raise ValueError("Camera conditioning was provided, but this transformer has no configured UCPE blocks.")
        if kv_caches is not None:
            if torch.is_grad_enabled():
                raise RuntimeError("Echo-WM Flash KV caches are inference-only.")
            if len(kv_caches) != len(self.transformer_blocks):
                raise ValueError(
                    f"Expected one KV cache per transformer block ({len(self.transformer_blocks)}), "
                    f"got {len(kv_caches)}."
                )

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
            audio_encoder_attention_mask = (1 - audio_encoder_attention_mask.to(audio_hidden_states.dtype)) * -10000.0
            audio_encoder_attention_mask = audio_encoder_attention_mask.unsqueeze(1)

        # Convert video_self_attention_mask from multiplicative mask ([0, 1]) to additive bias form (0 / -10000)
        # matching the encoder_attention_mask convention above. Shape is preserved: (B, T_v, T_v).
        if video_self_attention_mask is not None:
            video_self_attention_mask = (1 - video_self_attention_mask.to(hidden_states.dtype)) * -10000.0

        batch_size = hidden_states.size(0)

        # 1. Prepare RoPE positional embeddings
        if video_coords is None:
            video_coords = self.rope.prepare_video_coords(
                batch_size, num_frames, height, width, hidden_states.device, fps=fps
            )
        if audio_coords is None:
            audio_coords = self.audio_rope.prepare_audio_coords(
                batch_size, audio_num_frames, audio_hidden_states.device
            )

        video_rotary_emb = self.rope(video_coords, device=hidden_states.device)
        audio_rotary_emb = self.audio_rope(audio_coords, device=audio_hidden_states.device)

        video_cross_attn_rotary_emb = self.cross_attn_rope(video_coords[:, 0:1, :], device=hidden_states.device)
        audio_cross_attn_rotary_emb = self.cross_attn_audio_rope(
            audio_coords[:, 0:1, :], device=audio_hidden_states.device
        )

        # 2. Patchify input projections
        hidden_states = self.proj_in(hidden_states)
        audio_hidden_states = self.audio_proj_in(audio_hidden_states)

        # 2.1. Mark tokens whose latent encodes a single pixel frame (causal first frame, generated keyframe slots).
        if self.config.use_keyframes_abs_pos_embedding and video_keyframes_mask is not None:
            marker = (video_keyframes_mask > 0).to(dtype=hidden_states.dtype)
            hidden_states = hidden_states + marker * self.keyframes_abs_pos_embedding.to(dtype=hidden_states.dtype)

        # 3. Prepare timestep embeddings and modulation parameters
        timestep_cross_attn_gate_scale_factor = (
            self.config.cross_attn_timestep_scale_multiplier / self.config.timestep_scale_multiplier
        )

        # 3.1. Prepare global modality (video and audio) timestep embedding and modulation parameters
        # temb is used in the transformer blocks (as expected), while embedded_timestep is used for the output layer
        # modulation with scale_shift_table (and similarly for audio)
        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        temb_audio, audio_embedded_timestep = self.audio_time_embed(
            audio_timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
        audio_embedded_timestep = audio_embedded_timestep.view(batch_size, -1, audio_embedded_timestep.size(-1))

        if self.prompt_modulation and self.config.use_prompt_adaln_single:
            # LTX-2.3
            temb_prompt, _ = self.prompt_adaln(
                sigma.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )
            temb_prompt_audio, _ = self.audio_prompt_adaln(
                audio_sigma.flatten(), batch_size=batch_size, hidden_dtype=audio_hidden_states.dtype
            )
            temb_prompt = temb_prompt.view(batch_size, -1, temb_prompt.size(-1))
            temb_prompt_audio = temb_prompt_audio.view(batch_size, -1, temb_prompt_audio.size(-1))
        else:
            temb_prompt = temb_prompt_audio = None

        # 3.2. Prepare global modality cross attention modulation parameters
        video_ca_timestep = audio_sigma.flatten() if use_cross_timestep else timestep.flatten()
        video_cross_attn_scale_shift, _ = self.av_cross_attn_video_scale_shift(
            video_ca_timestep,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        video_cross_attn_a2v_gate, _ = self.av_cross_attn_video_a2v_gate(
            video_ca_timestep * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        video_cross_attn_scale_shift = video_cross_attn_scale_shift.view(
            batch_size, -1, video_cross_attn_scale_shift.shape[-1]
        )
        video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.view(batch_size, -1, video_cross_attn_a2v_gate.shape[-1])

        audio_ca_timestep = sigma.flatten() if use_cross_timestep else audio_timestep.flatten()
        audio_cross_attn_scale_shift, _ = self.av_cross_attn_audio_scale_shift(
            audio_ca_timestep,
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
            audio_ca_timestep * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.view(
            batch_size, -1, audio_cross_attn_scale_shift.shape[-1]
        )
        audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.view(batch_size, -1, audio_cross_attn_v2a_gate.shape[-1])

        # 4. Prepare prompt embeddings (LTX-2.0)
        if self.config.use_prompt_embeddings:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

            audio_encoder_hidden_states = self.audio_caption_projection(audio_encoder_hidden_states)
            audio_encoder_hidden_states = audio_encoder_hidden_states.view(
                batch_size, -1, audio_hidden_states.size(-1)
            )

        # 5. Run transformer blocks
        spatio_temporal_guidance_blocks = spatio_temporal_guidance_blocks or []
        if len(spatio_temporal_guidance_blocks) > 0 and perturbation_mask is None:
            # If STG is being used and perturbation_mask is not supplied, default to perturbing all batch elements.
            perturbation_mask = torch.zeros((batch_size,))
        if perturbation_mask is not None and perturbation_mask.ndim == 1:
            perturbation_mask = perturbation_mask[:, None, None]  # unsqueeze to 3D to broadcast with hidden_states
        all_perturbed = torch.all(perturbation_mask == 0) if perturbation_mask is not None else False
        stg_blocks = set(spatio_temporal_guidance_blocks)

        for block_idx, block in enumerate(self.transformer_blocks):
            block_perturbation_mask = perturbation_mask if block_idx in stg_blocks else None
            block_all_perturbed = all_perturbed if block_idx in stg_blocks else False

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, audio_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    audio_hidden_states,
                    encoder_hidden_states,
                    audio_encoder_hidden_states,
                    temb,
                    temb_audio,
                    video_cross_attn_scale_shift,
                    audio_cross_attn_scale_shift,
                    video_cross_attn_a2v_gate,
                    audio_cross_attn_v2a_gate,
                    temb_prompt,
                    temb_prompt_audio,
                    video_rotary_emb,
                    audio_rotary_emb,
                    video_cross_attn_rotary_emb,
                    audio_cross_attn_rotary_emb,
                    encoder_attention_mask,
                    audio_encoder_attention_mask,
                    video_self_attention_mask,  # self_attention_mask (video-only)
                    None,  # audio_self_attention_mask
                    None,  # a2v_cross_attention_mask
                    None,  # v2a_cross_attention_mask
                    not isolate_modalities,  # use_a2v_cross_attention
                    not isolate_modalities,  # use_v2a_cross_attention
                    block_perturbation_mask,
                    block_all_perturbed,
                    ucpe_viewmats,
                    ucpe_intrinsics,
                    kv_caches[block_idx] if kv_caches is not None else None,
                    current_video_token_start,
                    current_audio_token_start,
                )
            else:
                hidden_states, audio_hidden_states = block(
                    hidden_states=hidden_states,
                    audio_hidden_states=audio_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                    temb=temb,
                    temb_audio=temb_audio,
                    temb_ca_scale_shift=video_cross_attn_scale_shift,
                    temb_ca_audio_scale_shift=audio_cross_attn_scale_shift,
                    temb_ca_gate=video_cross_attn_a2v_gate,
                    temb_ca_audio_gate=audio_cross_attn_v2a_gate,
                    temb_prompt=temb_prompt,
                    temb_prompt_audio=temb_prompt_audio,
                    video_rotary_emb=video_rotary_emb,
                    audio_rotary_emb=audio_rotary_emb,
                    ca_video_rotary_emb=video_cross_attn_rotary_emb,
                    ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    audio_encoder_attention_mask=audio_encoder_attention_mask,
                    self_attention_mask=video_self_attention_mask,
                    audio_self_attention_mask=None,
                    a2v_cross_attention_mask=None,
                    v2a_cross_attention_mask=None,
                    use_a2v_cross_attention=not isolate_modalities,
                    use_v2a_cross_attention=not isolate_modalities,
                    perturbation_mask=block_perturbation_mask,
                    all_perturbed=block_all_perturbed,
                    ucpe_viewmats=ucpe_viewmats,
                    ucpe_intrinsics=ucpe_intrinsics,
                    kv_cache=kv_caches[block_idx] if kv_caches is not None else None,
                    current_video_token_start=current_video_token_start,
                    current_audio_token_start=current_audio_token_start,
                )

        # 6. Output layers (including unpatchification)
        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        audio_scale_shift_values = self.audio_scale_shift_table[None, None] + audio_embedded_timestep[:, :, None]
        audio_shift, audio_scale = audio_scale_shift_values[:, :, 0], audio_scale_shift_values[:, :, 1]

        audio_hidden_states = self.audio_norm_out(audio_hidden_states)
        audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
        audio_output = self.audio_proj_out(audio_hidden_states)

        if not return_dict:
            return (output, audio_output)
        return AudioVisualModelOutput(sample=output, audio_sample=audio_output)
