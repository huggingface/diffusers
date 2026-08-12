# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import math

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import AttentionBackendName, dispatch_attention_fn
from ..embeddings import Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm


def rope_params(max_seq_len, dim, theta=10000, offset=0):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len) + offset,
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_apply(x, grid_sizes, freqs, time_stride=1):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][: f * time_stride : time_stride].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device,
    )
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


def _get_qkv_projections(attn, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if not attn.is_cross_attention:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


def _get_added_kv_projections(attn, encoder_hidden_states_img: torch.Tensor):
    if attn.fused_projections:
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


class WanAnimate2KVLayerCache:
    """Per-layer K/V cache for the reference tokens.

    Holds the *pre-RoPE* projections: the generation pass re-applies rotary embeddings to the reference keys using the
    reference grid and the `refer_offset_*` offsets. Tensor format: `(batch_size, seq_len, num_heads, head_dim)`.
    """

    def __init__(self):
        self.key: torch.Tensor | None = None
        self.value: torch.Tensor | None = None

    def store(self, key: torch.Tensor, value: torch.Tensor):
        self.key = key
        self.value = value

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key is None:
            raise RuntimeError("The KV cache is empty. Run the reference pass before the generation pass.")
        return self.key, self.value

    def clear(self):
        self.key = None
        self.value = None


class WanAnimate2KVCache:
    """Container holding one [`WanAnimate2KVLayerCache`] per transformer layer."""

    def __init__(self, num_layers: int):
        self.layer_caches = [WanAnimate2KVLayerCache() for _ in range(num_layers)]

    def get(self, layer_idx: int) -> WanAnimate2KVLayerCache:
        return self.layer_caches[layer_idx]

    def clear(self):
        for cache in self.layer_caches:
            cache.clear()


class WanAnimate2AttnProcessor:
    r"""
    Self-attention for the Wan-Animate-2 in-context reference mechanism.

    With `kv_cache_mode="extract"` (the reference pass) this is dense self-attention over the reference tokens; the
    projected K/V are written to `kv_cache` before rotary embeddings are applied.

    With `kv_cache_mode="cached"` (the generation pass) the generation tokens and the cached reference tokens are
    packed into a 128-aligned `[generation | reference]` buffer and attended through a flex `BlockMask`, so each
    generation frame attends to every generation token plus the reference tokens at the same frame index. Because the
    pattern is expressed as a `BlockMask`, this path runs on the `flex` backend only.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "WanAnimate2Attention",
        hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor,
        grid_sizes: torch.Tensor,
        kv_cache: WanAnimate2KVLayerCache,
        kv_cache_mode: str,
        rope_stride: int = 1,
        reference_rotary_emb: torch.Tensor | None = None,
        reference_grid_sizes: torch.Tensor | None = None,
        reference_rope_stride: int = 1,
        attention_mask: BlockMask | None = None,
        origin_latent_frames: int | None = None,
        origin_latent_hw: int | None = None,
    ) -> torch.Tensor:
        query, key, value = _get_qkv_projections(attn, hidden_states, None)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if kv_cache_mode == "extract":
            kv_cache.store(key, value)

            # `rope_apply` computes in float64 and returns float32; attention runs in the model dtype.
            query = rope_apply(query, grid_sizes, rotary_emb, rope_stride).type_as(value)
            key = rope_apply(key, grid_sizes, rotary_emb, rope_stride).type_as(value)

            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=None,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        elif kv_cache_mode == "cached":
            query = rope_apply(query, grid_sizes, rotary_emb, rope_stride).type_as(value)
            key = rope_apply(key, grid_sizes, rotary_emb, rope_stride).type_as(value)

            key_ref, value_ref = kv_cache.get()
            key_ref = rope_apply(key_ref, reference_grid_sizes, reference_rotary_emb, reference_rope_stride).type_as(
                value
            )

            frames, height, width = grid_sizes[0].tolist()
            ref_frames, ref_height, ref_width = reference_grid_sizes[0].tolist()
            hw, ref_hw = height * width, ref_height * ref_width
            valid_length, ref_valid_length = frames * hw, ref_frames * ref_hw

            batch_size, _, heads, head_dim = query.shape

            # The block mask is built once for the full video resolution, so this segment is
            # scattered into a buffer of that size. Both segments are padded to a multiple of
            # 128 to match the block mask's granularity.
            packed_length = math.ceil((origin_latent_frames + 1) * origin_latent_hw / 128) * 128
            packed_ref_length = math.ceil(origin_latent_frames * origin_latent_hw / 128) * 128

            query_packed = query.new_zeros(batch_size, packed_length, heads, head_dim)
            key_packed = key.new_zeros(batch_size, packed_length + packed_ref_length, heads, head_dim)
            value_packed = value.new_zeros(batch_size, packed_length + packed_ref_length, heads, head_dim)

            generation = slice(0, frames * origin_latent_hw)
            query_packed[:, generation].view(batch_size, frames, origin_latent_hw, heads, head_dim)[:, :, :hw] = query[
                :, :valid_length
            ].view(batch_size, frames, hw, heads, head_dim)
            key_packed[:, generation].view(batch_size, frames, origin_latent_hw, heads, head_dim)[:, :, :hw] = key[
                :, :valid_length
            ].view(batch_size, frames, hw, heads, head_dim)
            value_packed[:, generation].view(batch_size, frames, origin_latent_hw, heads, head_dim)[:, :, :hw] = value[
                :, :valid_length
            ].view(batch_size, frames, hw, heads, head_dim)

            reference = slice(packed_length, packed_length + ref_frames * origin_latent_hw)
            key_packed[:, reference].view(batch_size, ref_frames, origin_latent_hw, heads, head_dim)[:, :, :ref_hw] = (
                key_ref[:, :ref_valid_length].view(batch_size, ref_frames, ref_hw, heads, head_dim)
            )
            value_packed[:, reference].view(batch_size, ref_frames, origin_latent_hw, heads, head_dim)[
                :, :, :ref_hw
            ] = value_ref[:, :ref_valid_length].view(batch_size, ref_frames, ref_hw, heads, head_dim)

            hidden_states = dispatch_attention_fn(
                query_packed,
                key_packed,
                value_packed,
                attn_mask=attention_mask,
                backend=AttentionBackendName.FLEX,
                parallel_config=self._parallel_config,
            )

            hidden_states = (
                hidden_states[:, generation]
                .view(batch_size, frames, origin_latent_hw, heads, head_dim)[:, :, :hw]
                .reshape(batch_size, valid_length, heads, head_dim)
            )
            # Padded query positions are not attended and pass through unchanged.
            hidden_states = torch.cat([hidden_states, query[:, valid_length:]], dim=1)
        else:
            raise ValueError(f"`kv_cache_mode` must be either 'extract' or 'cached', got {kv_cache_mode}.")

        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanAnimate2CrossAttnProcessor:
    r"""
    Cross-attention to the text embeddings, plus an additive branch over the CLIP image embeddings.

    The two token streams are passed as separate arguments rather than sliced out of one concatenated tensor, so the
    CLIP token count does not have to be hardcoded.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "WanAnimate2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        if encoder_hidden_states_image is not None:
            key_image, value_image = _get_added_kv_projections(attn, encoder_hidden_states_image)
            key_image = attn.norm_added_k(key_image)

            key_image = key_image.unflatten(2, (attn.heads, -1))
            value_image = value_image.unflatten(2, (attn.heads, -1))

            hidden_states_image = dispatch_attention_fn(
                query,
                key_image,
                value_image,
                attn_mask=None,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states = hidden_states + hidden_states_image.flatten(2, 3).type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanAnimate2Attention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WanAnimate2AttnProcessor
    _available_processors = [WanAnimate2AttnProcessor, WanAnimate2CrossAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int,
        eps: float = 1e-6,
        dropout: float = 0.0,
        added_kv_proj_dim: int | None = None,
        processor=None,
        is_cross_attention: bool = False,
    ):
        super().__init__()

        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.is_cross_attention = is_cross_attention
        self.use_bias = True

        self.to_q = torch.nn.Linear(dim, dim, bias=True)
        self.to_k = torch.nn.Linear(dim, dim, bias=True)
        self.to_v = torch.nn.Linear(dim, dim, bias=True)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(dim, dim, bias=True), torch.nn.Dropout(dropout)])
        self.norm_q = torch.nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim, eps=eps, elementwise_affine=True)

        self.set_processor(processor if processor is not None else self._default_processor_cls())

    # Copied from diffusers.models.transformers.transformer_wan.WanAttention.fuse_projections
    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if not self.is_cross_attention:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    # Copied from diffusers.models.transformers.transformer_wan.WanAttention.unfuse_projections
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.processor(self, hidden_states, **kwargs)


class WanAnimate2TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=False,
        eps=1e-6,
        refer_stride=1,
        use_img_emb=True,
    ):
        super().__init__()
        self.refer_stride = refer_stride

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.self_attn = WanAnimate2Attention(
            dim=dim,
            heads=num_heads,
            eps=eps,
            processor=WanAnimate2AttnProcessor(),
        )

        # 2. Cross-attention
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanAnimate2Attention(
            dim=dim,
            heads=num_heads,
            eps=eps,
            added_kv_proj_dim=dim if use_img_emb else None,
            is_cross_attention=True,
            processor=WanAnimate2CrossAttnProcessor(),
        )

        # 3. Feed-forward
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        kv_cache: WanAnimate2KVLayerCache,
        kv_cache_mode: str,
        rotary_emb: torch.Tensor,
        grid_sizes: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        reference_rotary_emb: torch.Tensor | None = None,
        reference_grid_sizes: torch.Tensor | None = None,
        attention_mask: BlockMask | None = None,
        origin_latent_frames: int | None = None,
        origin_latent_hw: int | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (self.modulation + temb).chunk(6, dim=1)

        # 1. Self-attention. The reference tokens sit on a strided time axis, so `refer_stride`
        # applies to whichever stream is the reference one in this mode.
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.self_attn(
            norm_hidden_states,
            rotary_emb=rotary_emb,
            grid_sizes=grid_sizes,
            kv_cache=kv_cache,
            kv_cache_mode=kv_cache_mode,
            rope_stride=self.refer_stride if kv_cache_mode == "extract" else 1,
            reference_rotary_emb=reference_rotary_emb,
            reference_grid_sizes=reference_grid_sizes,
            reference_rope_stride=self.refer_stride,
            attention_mask=attention_mask,
            origin_latent_frames=origin_latent_frames,
            origin_latent_hw=origin_latent_hw,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        hidden_states = hidden_states + self.cross_attn(
            self.norm3(hidden_states.float()).type_as(hidden_states),
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
        )

        # 3. Feed-forward
        norm_hidden_states = (self.norm2(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        shift, scale = (self.modulation + e.float().unsqueeze(1)).chunk(2, dim=1)
        x = self.head((self.norm(x.float()) * (1 + scale) + shift).type_as(x))
        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanAnimate2Transformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, AttentionMixin):
    r"""
    A Transformer model for video-like data used in the Wan-Animate-2 model.

    Wan-Animate-2 uses an in-context attention mechanism with a KV cache: a reference video is first encoded
    (``kv_cache_mode="extract"``) to populate a [`WanAnimate2KVCache`], then each denoising step
    (``kv_cache_mode="cached"``) attends jointly over the generation tokens and the cached reference K/V through a flex
    ``BlockMask``. The generation self-attention therefore runs on the ``flex`` attention backend only; every other
    attention in the model works on any backend.

    Args:
        patch_size (`tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        text_len (`int`, defaults to `512`):
            Fixed length for text embeddings.
        in_dim (`int`, defaults to `36`):
            The number of channels in the input (2 * latent_channels + 4 for mask channel).
        dim (`int`, defaults to `5120`):
            The number of channels in the transformer.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        text_dim (`int`, defaults to `4096`):
            Input dimension for text embeddings.
        out_dim (`int`, defaults to `16`):
            The number of channels in the output.
        num_heads (`int`, defaults to `40`):
            The number of attention heads.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        use_img_emb (`bool`, defaults to `True`):
            Whether to use CLIP image embedding.
        refer_offset_t (`int`, defaults to `1`):
            RoPE offset for the temporal dimension of the reference.
        refer_offset_h (`int`, defaults to `0`):
            RoPE offset for the height dimension of the reference.
        refer_offset_w (`int`, defaults to `-1`):
            RoPE offset for the width dimension of the reference. -1 means use the generation grid size.
        refer_stride (`int`, defaults to `1`):
            Stride for RoPE application on the reference.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "img_emb", "norm"]
    _no_split_modules = ["WanAnimate2TransformerBlock"]
    _repeated_blocks = ["WanAnimate2TransformerBlock"]
    _skip_keys = ["kv_cache"]
    _keep_in_fp32_modules = [
        "time_embedding",
        "time_projection",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
        "modulation",
    ]

    @register_to_config
    def __init__(
        self,
        patch_size: tuple = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 36,
        dim: int = 5120,
        ffn_dim: int = 13824,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 40,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        use_img_emb: bool = True,
        refer_offset_t: int = 1,
        refer_offset_h: int = 0,
        refer_offset_w: int = -1,
        refer_stride: int = 1,
    ):
        super().__init__()

        if dim % num_heads != 0 or (dim // num_heads) % 2 != 0:
            raise ValueError(f"`dim` ({dim}) must split into an even head size across `num_heads` ({num_heads}).")

        # [Denoising Transformer]
        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )

        self.timesteps_proj = Timesteps(num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        # blocks
        self.blocks = nn.ModuleList(
            [
                WanAnimate2TransformerBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    cross_attn_norm,
                    eps,
                    refer_stride,
                    use_img_emb=use_img_emb,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        if use_img_emb:
            self.img_emb = MLPProj(1280, dim)

        self.gradient_checkpointing = False
        # Pure memo caches, keyed by geometry — the only state `forward` writes. Both hold
        # deterministic values, so hits and misses produce identical outputs.
        self.block_masks = {}
        self.rope_freqs_cache = {}

    def _rope_freqs(self, offsets: tuple[int, int, int], device: torch.device) -> torch.Tensor:
        """RoPE frequency table for a stream whose (t, h, w) axes start at `offsets`."""
        freqs = self.rope_freqs_cache.get(offsets)
        if freqs is None:
            d = self.config.dim // self.config.num_heads
            freqs = torch.cat(
                [
                    rope_params(512, d - 4 * (d // 6), offset=offsets[0]),
                    rope_params(512, 2 * (d // 6), offset=offsets[1]),
                    rope_params(512, 2 * (d // 6), offset=offsets[2]),
                ],
                dim=1,
            )
        if freqs.device != device:
            freqs = freqs.to(device)
        self.rope_freqs_cache[offsets] = freqs
        return freqs

    def create_mask(self, origin_latent_f, hw, device):
        q_len = (origin_latent_f + 1) * hw
        k_len = origin_latent_f * hw

        q_len_total = math.ceil(q_len / 128) * 128
        k_extra_len_total = math.ceil(k_len / 128) * 128
        k_len_total = q_len_total + k_extra_len_total

        q_limit = q_len
        k_limit = k_len
        q_total = q_len_total

        def attention_mask_logic(b, h, q_idx, kv_idx):
            q_valid = q_idx < q_limit
            is_base_attention = kv_idx < q_limit

            q_frame = q_idx // hw
            is_first_part = kv_idx < q_total

            kv_frame_1 = kv_idx // hw
            kv_is_valid_1 = kv_idx < q_limit

            rel_kv_idx = kv_idx - q_total
            kv_frame_2 = (rel_kv_idx // hw) + 1
            kv_is_valid_2 = rel_kv_idx < k_limit

            kv_frame = torch.where(is_first_part, kv_frame_1, kv_frame_2)
            kv_is_valid = torch.where(is_first_part, kv_is_valid_1, kv_is_valid_2)

            is_cond_attention = (q_frame == kv_frame) & kv_is_valid

            return q_valid & (is_base_attention | is_cond_attention)

        block_mask = create_block_mask(
            attention_mask_logic,
            B=None,
            H=None,
            Q_LEN=q_len_total,
            KV_LEN=k_len_total,
            device=device,
            _compile=True,
        )
        return block_mask

    def forward(
        self,
        hidden_states: list[torch.Tensor],
        timestep: torch.Tensor,
        encoder_hidden_states: list[torch.Tensor],
        condition_latents: list[torch.Tensor],
        kv_cache: WanAnimate2KVCache,
        kv_cache_mode: str,
        seq_len: int,
        encoder_hidden_states_image: torch.Tensor | None = None,
        offset_grid_sizes: torch.Tensor | None = None,
        reference_grid_sizes: torch.Tensor | None = None,
        origin_len: int | None = None,
        origin_area: list[int] | None = None,
        is_uncondtion: bool = False,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[list[torch.Tensor]]:
        r"""
        Args:
            hidden_states (`list[torch.Tensor]`):
                Latents for this pass — the reference latents when `kv_cache_mode="extract"`, the noisy generation
                latents when `kv_cache_mode="cached"`.
            timestep (`torch.Tensor`):
                Denoising timestep. Ignored under `kv_cache_mode="extract"`, which uses a fixed timestep of 1.
            encoder_hidden_states (`list[torch.Tensor]`):
                Text embeddings for this pass.
            condition_latents (`list[torch.Tensor]`):
                Conditioning latents concatenated to `hidden_states` before patch embedding.
            kv_cache (`WanAnimate2KVCache`):
                Written under `kv_cache_mode="extract"`, read under `"cached"`.
            kv_cache_mode (`str`):
                `"extract"` runs the reference pass and populates `kv_cache`; `"cached"` runs a denoising step against
                the cached reference tokens.
            seq_len (`int`):
                Token count each sample must hold after patch embedding.
            encoder_hidden_states_image (`torch.Tensor`, *optional*):
                CLIP image embeddings, used when the model is configured with `use_img_emb`.
            offset_grid_sizes (`torch.Tensor`, *optional*):
                Patch grid of the reference latents, used to resolve any `refer_offset_*` set to -1. Required under
                `kv_cache_mode="extract"`; under `"cached"`, `reference_grid_sizes` describes the same grid and is used
                instead.
            reference_grid_sizes (`torch.Tensor`, *optional*):
                Patch grid of the reference latents, used for the reference rotary embeddings. Required under
                `kv_cache_mode="cached"`.
            origin_len (`int`, *optional*):
                Frame count of the full video, which the in-context block mask is built over. Required under
                `kv_cache_mode="cached"`.
            origin_area (`list[int]`, *optional*):
                Spatial size `[height, width]` of the full video, which the in-context block mask is built over.
                Required under `kv_cache_mode="cached"`.
            is_uncondtion (`bool`, *optional*):
                Whether this is the unconditional branch of classifier-free guidance.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain tuple.

        Returns:
            [`~models.modeling_outputs.Transformer2DModelOutput`] or `tuple(list[torch.Tensor])`:
                The predicted sample per input latent, unpatchified; a plain tuple if `return_dict` is `False`.
        """
        if kv_cache_mode not in ("extract", "cached"):
            raise ValueError(f"`kv_cache_mode` must be either 'extract' or 'cached', got {kv_cache_mode}.")

        device = self.patch_embedding.weight.device

        # 1. Patch embedding. `grid_sizes` describes whichever stream this pass is running over.
        hidden_states = [torch.cat([u, v], dim=0) for u, v in zip(hidden_states, condition_latents)]
        hidden_states = [self.patch_embedding(u.unsqueeze(0)) for u in hidden_states]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in hidden_states])
        hidden_states = [u.flatten(2).transpose(1, 2) for u in hidden_states]
        if any(u.size(1) != seq_len for u in hidden_states):
            raise ValueError(
                f"Each sample must hold exactly `seq_len` ({seq_len}) tokens, got "
                f"{[u.size(1) for u in hidden_states]}. Self-attention here is either dense and unmasked or driven "
                f"by a block mask built from the grid, so a padded sequence would not line up."
            )
        hidden_states = torch.cat(hidden_states)

        # 2. Rotary embeddings for the reference stream. The `refer_offset_*` config values place
        # the reference tokens on a RoPE grid disjoint from the generation tokens; -1 means "use
        # the reference grid size for that axis", resolved per call from the reference grid, which
        # arrives as `offset_grid_sizes` under "extract" and as `reference_grid_sizes` under "cached".
        reference_grid = offset_grid_sizes if kv_cache_mode == "extract" else reference_grid_sizes
        refer_offsets = tuple(
            offset if offset >= 0 else reference_grid[0][axis].item()
            for axis, offset in enumerate(
                (self.config.refer_offset_t, self.config.refer_offset_h, self.config.refer_offset_w)
            )
        )
        freqs_ref = self._rope_freqs(refer_offsets, device)

        # 3. Time and context embeddings. The reference pass is modulated at a fixed timestep.
        timestep_input = timestep * 0 + 1 if kv_cache_mode == "extract" else timestep
        temb = self.time_embedding(
            self.timesteps_proj(timestep_input).to(dtype=next(self.time_embedding.parameters()).dtype)
        )
        timestep_proj = self.time_projection(temb).unflatten(1, (6, self.config.dim))

        encoder_hidden_states = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.config.text_len - u.size(0), u.size(1))])
                    for u in encoder_hidden_states
                ]
            )
        )
        encoder_hidden_states_image = self.img_emb(encoder_hidden_states_image) if self.config.use_img_emb else None

        # 4. Per-mode block arguments.
        if kv_cache_mode == "extract":
            block_kwargs = {
                "rotary_emb": freqs_ref,
                "grid_sizes": grid_sizes,
            }
        else:
            # Latent geometry of the full video, which is what the block mask is built over. The
            # current segment is scattered into a buffer of this size inside the attention processor.
            origin_latent_frames = origin_len // 4 + 1
            origin_latent_hw = origin_area[0] * origin_area[1] // 256

            block_mask_id = (origin_latent_frames, origin_latent_hw)
            if block_mask_id not in self.block_masks:
                self.block_masks[block_mask_id] = self.create_mask(
                    origin_latent_frames, origin_latent_hw, hidden_states.device
                )

            block_kwargs = {
                "rotary_emb": self._rope_freqs((0, 0, 0), device),
                "grid_sizes": grid_sizes,
                "reference_rotary_emb": freqs_ref,
                "reference_grid_sizes": reference_grid_sizes,
                "attention_mask": self.block_masks[block_mask_id],
                "origin_latent_frames": origin_latent_frames,
                "origin_latent_hw": origin_latent_hw,
            }

        block_kwargs.update(
            temb=timestep_proj,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            kv_cache_mode=kv_cache_mode,
        )

        # 5. Transformer blocks.
        for idx, block in enumerate(self.blocks):
            if is_uncondtion and idx == 9:
                continue
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, kv_cache=kv_cache.get(idx), **block_kwargs
                )
            else:
                hidden_states = block(hidden_states, kv_cache=kv_cache.get(idx), **block_kwargs)

        # 6. Output. Under `kv_cache_mode="extract"` the meaningful product is the populated
        # cache; the sample is returned anyway so both modes have the same return type.
        hidden_states = self.head(hidden_states, temb)
        output = [u.float() for u in self.unpatchify(hidden_states, grid_sizes)]

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def unpatchify(self, x, grid_sizes):
        c = self.config.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.config.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.config.patch_size)])
            out.append(u)
        return out
