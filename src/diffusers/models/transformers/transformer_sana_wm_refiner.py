# Copyright 2025 The HuggingFace Team and SANA-WM Authors. All rights reserved.
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

import inspect

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import apply_lora_scale, is_torch_version, logging
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings, PixArtAlphaTextProjection
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.models.transformers.transformer_ltx2.apply_interleaved_rotary_emb
def apply_interleaved_rotary_emb(x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)  # [B, S, C // 2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


# Copied from diffusers.models.transformers.transformer_ltx2.apply_split_rotary_emb
def apply_split_rotary_emb(x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = freqs

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
    split_x = x.reshape(*x.shape[:-1], 2, r).float()  # Explicitly upcast to float
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


# Copied from diffusers.models.transformers.transformer_ltx2.LTX2AdaLayerNormSingle
class SanaWMLTX2AdaLayerNormSingle(nn.Module):
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # No modulation happening here.
        added_cond_kwargs = added_cond_kwargs or {"resolution": None, "aspect_ratio": None}
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


# Copied from diffusers.models.transformers.transformer_ltx2.LTX2AudioVideoAttnProcessor with LTX2->SanaWMLTX2
class SanaWMLTX2AudioVideoAttnProcessor:
    r"""
    Processor for implementing attention (SDPA is used by default if you're using PyTorch 2.0) for the LTX-2.0 model.
    Compared to the LTX-1.0 model, we allow the RoPE embeddings for the queries and keys to be separate so that we can
    support audio-to-video (a2v) and video-to-audio (v2a) cross attention.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if is_torch_version("<", "2.0"):
            raise ValueError(
                "LTX attention processors require a minimum PyTorch version of 2.0. Please upgrade your PyTorch installation."
            )

    def __call__(
        self,
        attn: "SanaWMLTX2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
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
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if query_rotary_emb is not None:
            if attn.rope_type == "interleaved":
                query = apply_interleaved_rotary_emb(query, query_rotary_emb)
                key = apply_interleaved_rotary_emb(
                    key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                )
            elif attn.rope_type == "split":
                query = apply_split_rotary_emb(query, query_rotary_emb)
                key = apply_split_rotary_emb(key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb)

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


# Copied from diffusers.models.transformers.transformer_ltx2.LTX2PerturbedAttnProcessor with LTX2->SanaWMLTX2
class SanaWMLTX2PerturbedAttnProcessor:
    r"""
    Processor which implements attention with perturbation masking and per-head gating for LTX-2.X models.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if is_torch_version("<", "2.0"):
            raise ValueError(
                "LTX attention processors require a minimum PyTorch version of 2.0. Please upgrade your PyTorch installation."
            )

    def __call__(
        self,
        attn: "SanaWMLTX2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool | None = None,
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

        value = attn.to_v(encoder_hidden_states)
        if all_perturbed is None:
            all_perturbed = torch.all(perturbation_mask == 0) if perturbation_mask is not None else False

        if all_perturbed:
            # Skip attention, use the value projection value
            hidden_states = value
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)

            query = attn.norm_q(query)
            key = attn.norm_k(key)

            if query_rotary_emb is not None:
                if attn.rope_type == "interleaved":
                    query = apply_interleaved_rotary_emb(query, query_rotary_emb)
                    key = apply_interleaved_rotary_emb(
                        key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                    )
                elif attn.rope_type == "split":
                    query = apply_split_rotary_emb(query, query_rotary_emb)
                    key = apply_split_rotary_emb(
                        key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                    )

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


# Copied from diffusers.models.transformers.transformer_ltx2.LTX2Attention with LTX2->SanaWMLTX2
class SanaWMLTX2Attention(torch.nn.Module, AttentionModuleMixin):
    r"""
    Attention class for all LTX-2.0 attention layers. Compared to LTX-1.0, this supports specifying the query and key
    RoPE embeddings separately for audio-to-video (a2v) and video-to-audio (v2a) cross-attention.
    """

    _default_processor_cls = SanaWMLTX2AudioVideoAttnProcessor
    _available_processors = [SanaWMLTX2AudioVideoAttnProcessor, SanaWMLTX2PerturbedAttnProcessor]

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


# Copied from diffusers.models.transformers.transformer_ltx2.LTX2AudioVideoRotaryPosEmbed
class SanaWMLTX2AudioVideoRotaryPosEmbed(nn.Module):
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
        freqs_dtype = torch.float64 if self.double_precision else torch.float32
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


# ``kv_cache_mode`` values accepted by [`SanaWMLTX2RefinerTransformer3DModel`] and
# [`SanaWMLTX2RefinerTransformerBlock`]. See [`SanaWMRefinerKVCache`] for the AR contract they implement.
class SanaWMRefinerKVLayerCache:
    r"""
    Per-layer KV cache for the SANA-WM stage-2 chunk-causal AR refiner.

    Holds the two halves of the sliding-window prefix that the refiner's self-attention attends to, plus a slot for
    reading back the K/V that the last forward captured. All tensors are `(batch_size, num_tokens, inner_dim)` (i.e.
    before the per-head unflatten), matching the layout the refiner's self-attention concatenates in.

    * ``sink_k_pre`` / ``sink_v``: **pre**-RoPE K/V of the attention-sink frames, captured once from the raw stage-1
      latents. They are stored pre-RoPE so each AR window can re-apply RoPE at its own shifted sink offset
      (``SanaWMRefinerKVCache.sink_pe``).
    * ``history_k`` / ``history_v``: **post**-RoPE K/V of the already refined recent frames, ready to be concatenated
      as-is.
    * ``captured_k_pre`` / ``captured_v_pre`` and ``captured_k_post`` / ``captured_v_post``: readback slots written by
      the capture ``kv_cache_mode``s.
    """

    def __init__(self):
        self.sink_k_pre: torch.Tensor | None = None
        self.sink_v: torch.Tensor | None = None
        self.history_k: torch.Tensor | None = None
        self.history_v: torch.Tensor | None = None
        self.captured_k_pre: torch.Tensor | None = None
        self.captured_v_pre: torch.Tensor | None = None
        self.captured_k_post: torch.Tensor | None = None
        self.captured_v_post: torch.Tensor | None = None

    def store_sink(self, sink_k_pre: torch.Tensor, sink_v: torch.Tensor) -> None:
        """Store the pre-RoPE sink K/V."""
        self.sink_k_pre = sink_k_pre
        self.sink_v = sink_v

    def get_sink(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the pre-RoPE sink K/V, or `None` if it has not been captured (or is empty)."""
        if self.sink_k_pre is None or self.sink_v is None or self.sink_k_pre.shape[1] == 0:
            return None
        return self.sink_k_pre, self.sink_v

    def store_history(self, history_k: torch.Tensor, history_v: torch.Tensor) -> None:
        """Store the post-RoPE recent-history K/V."""
        self.history_k = history_k
        self.history_v = history_v

    def get_history(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the post-RoPE recent-history K/V, or `None` if empty."""
        if self.history_k is None or self.history_v is None or self.history_k.shape[1] == 0:
            return None
        return self.history_k, self.history_v

    def store_captured_pre_rope(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store the pre-RoPE K/V produced by the current forward."""
        self.captured_k_pre = key
        self.captured_v_pre = value

    def get_captured_pre_rope(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pop the pre-RoPE K/V captured by the last forward."""
        if self.captured_k_pre is None:
            raise RuntimeError("No pre-RoPE K/V was captured. Run a forward with `kv_cache_mode='capture_pre_rope'`.")
        key, value = self.captured_k_pre, self.captured_v_pre
        # Release the references so the caller owns the only handle.
        self.captured_k_pre = self.captured_v_pre = None
        return key, value

    def store_captured_post_rope(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store the post-RoPE K/V produced by the current forward."""
        self.captured_k_post = key
        self.captured_v_post = value

    def get_captured_post_rope(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pop the post-RoPE K/V captured by the last forward."""
        if self.captured_k_post is None:
            raise RuntimeError(
                "No post-RoPE K/V was captured. Run a forward with `kv_cache_mode='inject_and_capture_post_rope'`."
            )
        key, value = self.captured_k_post, self.captured_v_post
        self.captured_k_post = self.captured_v_post = None
        return key, value

    def clear(self) -> None:
        self.sink_k_pre = None
        self.sink_v = None
        self.history_k = None
        self.history_v = None
        self.captured_k_pre = None
        self.captured_v_pre = None
        self.captured_k_post = None
        self.captured_v_post = None


class SanaWMRefinerKVCache:
    r"""
    Container holding one [`SanaWMRefinerKVLayerCache`] per transformer block, plus the shared sink RoPE.

    This implements the ``rf_shifted_sink`` KV-cache contract the SANA-WM stage-2 refiner was trained with. Refinement
    is chunk-causal: `block_size` latent frames are denoised at a time while attending to a bounded window of
    `[attention sink + recent history + active block]` K/V.

    * ``sink_pe``: the `(cos, sin)` RoPE tuple for the sink frames, rebuilt per AR window at the sliding
      ``sink_rope_offset`` so the sink sits immediately before the bounded working cache. Shared across layers because
      RoPE does not depend on the layer.

    Args:
        num_layers (`int`):
            Number of transformer blocks to allocate a per-layer cache for.
    """

    def __init__(self, num_layers: int):
        self.layer_caches = [SanaWMRefinerKVLayerCache() for _ in range(num_layers)]
        self.sink_pe: tuple[torch.Tensor, torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self.layer_caches)

    def get(self, layer_idx: int) -> SanaWMRefinerKVLayerCache:
        return self.layer_caches[layer_idx]

    def clear(self) -> None:
        for layer_cache in self.layer_caches:
            layer_cache.clear()
        self.sink_pe = None


class SanaWMLTX2RefinerAttnProcessor:
    """Self-attention over `[sink + history + current]` K/V for the sliding-window AR refiner.

    Unlike the plain LTX-2 processors this one is cache-aware: `kv_cache_mode` decides whether the layer cache's sink
    and recent-history K/V are prepended before the single SDPA call, and whether the current block's K/V is written
    back for the next window.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: SanaWMLTX2Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        sink_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: SanaWMRefinerKVLayerCache | None = None,
        kv_cache_mode: str | None = None,
    ) -> torch.Tensor:
        """LTX-2 self-attention over `[sink + history + current]` K/V.

        The queries always come from the active block only. Depending on `kv_cache_mode`, the layer cache's pre-RoPE
        sink K/V (re-RoPE'd here with `sink_rotary_emb`) and post-RoPE recent-history K/V are prepended to the current
        K/V before a single SDPA call, and/or the current K/V is written back to the cache.
        """
        del encoder_hidden_states, attention_mask, key_rotary_emb

        gate_logits = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Capture PRE-RoPE (post-norm) K/V so a future window can re-apply RoPE at its shifted sink offset.
        if kv_cache_mode == "capture_pre_rope":
            kv_cache.store_captured_pre_rope(key.detach().clone(), value.detach().clone())

        if attn.rope_type == "interleaved":
            query = apply_interleaved_rotary_emb(query, query_rotary_emb)
            key = apply_interleaved_rotary_emb(key, query_rotary_emb)
        elif attn.rope_type == "split":
            query = apply_split_rotary_emb(query, query_rotary_emb)
            key = apply_split_rotary_emb(key, query_rotary_emb)
        else:
            raise ValueError(f"Unsupported LTX-2 RoPE type: {attn.rope_type}")

        # Capture POST-RoPE K/V so the next window can concatenate the recent history directly. Deliberately taken
        # before the prefix is prepended, so only the current block's tokens are recorded.
        if kv_cache_mode == "inject_and_capture_post_rope":
            kv_cache.store_captured_post_rope(key.detach().clone(), value.detach().clone())

        if kv_cache_mode in ("inject", "inject_and_capture_post_rope"):
            prefix_k_parts: list[torch.Tensor] = []
            prefix_v_parts: list[torch.Tensor] = []
            sink_kv = kv_cache.get_sink()
            if sink_kv is not None:
                if sink_rotary_emb is None:
                    raise ValueError("Injecting the attention sink requires the `sink_pe` RoPE tuple on the KV cache.")
                sink_k_pre, sink_v = sink_kv
                sink_k_pre = sink_k_pre.to(key.dtype)
                if attn.rope_type == "interleaved":
                    sink_k = apply_interleaved_rotary_emb(sink_k_pre, sink_rotary_emb)
                else:
                    sink_k = apply_split_rotary_emb(sink_k_pre, sink_rotary_emb)
                prefix_k_parts.append(sink_k)
                prefix_v_parts.append(sink_v.to(value.dtype))
            history_kv = kv_cache.get_history()
            if history_kv is not None:
                prefix_k_parts.append(history_kv[0].to(key.dtype))
                prefix_v_parts.append(history_kv[1].to(value.dtype))
            if prefix_k_parts:
                key = torch.cat([*prefix_k_parts, key], dim=1)
                value = torch.cat([*prefix_v_parts, value], dim=1)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if gate_logits is not None:
            hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
            gates = 2.0 * torch.sigmoid(gate_logits)
            hidden_states = hidden_states * gates.unsqueeze(-1)
            hidden_states = hidden_states.flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class SanaWMLTX2RefinerTransformerBlock(nn.Module):
    r"""
    Video-only, streaming-attention variant of [`LTX2VideoTransformerBlock`] used by the SANA-WM stage-2 refiner.

    The submodule structure is copied verbatim from [`LTX2VideoTransformerBlock`] (so LTX-2 checkpoints load as-is);
    only [`~SanaWMLTX2RefinerTransformerBlock.forward`] differs. It runs the video stream only (self-attn -> prompt
    cross-attn -> feed-forward), skipping the audio and audio/video cross-attention branches, and routes the
    self-attention through a KV-cached sliding window instead of plain full self-attention.
    """

    # Copied from diffusers.models.transformers.transformer_ltx2.LTX2VideoTransformerBlock.__init__ with LTX2->SanaWMLTX2
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
    ):
        super().__init__()

        self.perturbed_attn = perturbed_attn
        if perturbed_attn:
            attn_processor_cls = SanaWMLTX2PerturbedAttnProcessor
        else:
            attn_processor_cls = SanaWMLTX2AudioVideoAttnProcessor

        # 1. Self-Attention (video and audio)
        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = SanaWMLTX2Attention(
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

        self.audio_norm1 = RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_attn1 = SanaWMLTX2Attention(
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
        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = SanaWMLTX2Attention(
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

        self.audio_norm2 = RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_attn2 = SanaWMLTX2Attention(
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
        self.audio_to_video_norm = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_to_video_attn = SanaWMLTX2Attention(
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
        self.video_to_audio_norm = RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.video_to_audio_attn = SanaWMLTX2Attention(
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
        self.norm3 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.ff = FeedForward(dim, activation_fn=activation_fn, bias=ff_bias)

        self.audio_norm3 = RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
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
    # Copied from diffusers.models.transformers.transformer_ltx2.LTX2VideoTransformerBlock.get_mod_params
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
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        video_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        encoder_attention_mask: torch.Tensor | None = None,
        sink_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: SanaWMRefinerKVLayerCache | None = None,
        kv_cache_mode: str | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)

        # 1. Video self-attention over the KV-cached sliding window
        norm_hidden_states = self.norm1(hidden_states)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.get_mod_params(
            self.scale_shift_table, temb, batch_size
        )
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_hidden_states = self.attn1(
            norm_hidden_states,
            query_rotary_emb=video_rotary_emb,
            sink_rotary_emb=sink_rotary_emb,
            kv_cache=kv_cache,
            kv_cache_mode=kv_cache_mode,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        # 2. Prompt cross-attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_hidden_states = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_rotary_emb=None,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + attn_hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + self.ff(norm_hidden_states) * gate_mlp
        return hidden_states


class SanaWMLTX2RefinerTransformer3DModel(
    ModelMixin, ConfigMixin, AttentionMixin, FromOriginalModelMixin, PeftAdapterMixin, CacheMixin
):
    r"""
    The chunk-causal autoregressive refiner transformer used as SANA-WM stage 2.

    Architecturally identical to [`LTX2VideoTransformer3DModel`] — same config arguments, same submodules, same
    parameter names — so a released LTX-2 checkpoint loads into it unchanged. What differs is the forward pass:

    * only the video stream is run (the audio and audio/video cross-attention branches are skipped),
    * self-attention runs against an explicit sliding-window KV cache ([`SanaWMRefinerKVCache`]) holding the attention
      sink plus recent refined history, so refinement cost is bounded per AR block and scales linearly with video
      length,
    * the caller supplies the video RoPE, which lets each AR window keep every frame's absolute index in the source
      video (see [`~SanaWMLTX2RefinerTransformer3DModel.build_rotary_emb_for_absolute_positions`]).

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
        rope_type (`str`, defaults to `"interleaved"`):
            Which RoPE application to use (`"interleaved"` or `"split"`).

    The remaining arguments mirror [`LTX2VideoTransformer3DModel`] one-for-one. The audio-side arguments and submodules
    are kept purely so the checkpoint's audio weights round-trip; they are not used by the refiner forward.
    """

    _skip_layerwise_casting_patterns = ["norm"]
    _repeated_blocks = ["SanaWMLTX2RefinerTransformerBlock"]
    _skip_keys = ["kv_cache"]

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
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        audio_out_channels = audio_out_channels or audio_in_channels
        inner_dim = num_attention_heads * attention_head_dim
        audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim

        # 1. Patchification input projections
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.audio_proj_in = nn.Linear(audio_in_channels, audio_inner_dim)

        if use_keyframes_abs_pos_embedding:
            self.keyframes_abs_pos_embedding = nn.Parameter(torch.zeros(1, inner_dim))

        # 2. Prompt embeddings
        if use_prompt_embeddings:
            self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)
            self.audio_caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels, hidden_size=audio_inner_dim
            )

        # 3. Timestep Modulation Params and Embedding
        self.prompt_modulation = cross_attn_mod or audio_cross_attn_mod

        # 3.1. Global Timestep Modulation Parameters (except for cross-attention) and timestep + size embedding
        video_time_emb_mod_params = 9 if cross_attn_mod else 6
        audio_time_emb_mod_params = 9 if audio_cross_attn_mod else 6
        self.time_embed = SanaWMLTX2AdaLayerNormSingle(
            inner_dim, num_mod_params=video_time_emb_mod_params, use_additional_conditions=False
        )
        self.audio_time_embed = SanaWMLTX2AdaLayerNormSingle(
            audio_inner_dim, num_mod_params=audio_time_emb_mod_params, use_additional_conditions=False
        )

        # 3.2. Global Cross Attention Modulation Parameters
        self.av_cross_attn_video_scale_shift = SanaWMLTX2AdaLayerNormSingle(
            inner_dim, num_mod_params=4, use_additional_conditions=False
        )
        self.av_cross_attn_audio_scale_shift = SanaWMLTX2AdaLayerNormSingle(
            audio_inner_dim, num_mod_params=4, use_additional_conditions=False
        )
        self.av_cross_attn_video_a2v_gate = SanaWMLTX2AdaLayerNormSingle(
            inner_dim, num_mod_params=1, use_additional_conditions=False
        )
        self.av_cross_attn_audio_v2a_gate = SanaWMLTX2AdaLayerNormSingle(
            audio_inner_dim, num_mod_params=1, use_additional_conditions=False
        )

        # 3.3. Output Layer Scale/Shift Modulation parameters
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.audio_scale_shift_table = nn.Parameter(torch.randn(2, audio_inner_dim) / audio_inner_dim**0.5)

        # 3.4. Prompt Scale/Shift Modulation parameters (LTX-2.3)
        if self.prompt_modulation and use_prompt_adaln_single:
            self.prompt_adaln = SanaWMLTX2AdaLayerNormSingle(
                inner_dim, num_mod_params=2, use_additional_conditions=False
            )
            self.audio_prompt_adaln = SanaWMLTX2AdaLayerNormSingle(
                audio_inner_dim, num_mod_params=2, use_additional_conditions=False
            )

        # 4. Rotary Positional Embeddings (RoPE)
        self.rope = SanaWMLTX2AudioVideoRotaryPosEmbed(
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
        self.audio_rope = SanaWMLTX2AudioVideoRotaryPosEmbed(
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
        self.cross_attn_rope = SanaWMLTX2AudioVideoRotaryPosEmbed(
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
        self.cross_attn_audio_rope = SanaWMLTX2AudioVideoRotaryPosEmbed(
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
                SanaWMLTX2RefinerTransformerBlock(
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
                )
                for _ in range(num_layers)
            ]
        )
        # The blocks are built by LTX-2's `__init__`, which installs LTX-2's own self-attention processor. Swap in
        # the KV-cached sliding-window one the AR refiner needs; the module and its weights are otherwise identical.
        for block in self.transformer_blocks:
            block.attn1.set_processor(SanaWMLTX2RefinerAttnProcessor())

        # 6. Output layers
        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)

        self.audio_norm_out = nn.LayerNorm(audio_inner_dim, eps=1e-6, elementwise_affine=False)
        self.audio_proj_out = nn.Linear(audio_inner_dim, audio_out_channels)

        self.gradient_checkpointing = False

    def build_rotary_emb_for_absolute_positions(
        self,
        batch_size: int,
        frame_positions: list[int],
        height: int,
        width: int,
        device: torch.device,
        fps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Build the video RoPE for an explicit list of absolute latent-frame indices.

        [`SanaWMLTX2AudioVideoRotaryPosEmbed.prepare_video_coords`] assumes a contiguous `torch.arange(num_frames)`,
        which is fine for bidirectional inference. The sliding-window AR refiner instead needs to keep each frame's
        absolute index in the source video, so RoPE captures the correct temporal phase across the `[sink + recent +
        active]` window.

        Args:
            batch_size (`int`):
                Batch size to broadcast the coordinates to.
            frame_positions (`list[int]`):
                Absolute latent-frame indices covered by this window.
            height (`int`), width (`int`):
                Latent spatial resolution.
            device (`torch.device`):
                Device to build the coordinates on.
            fps (`float`):
                Video frame rate, which drives LTX-2's temporal RoPE scaling.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the `(cos, sin)` RoPE tuple.
        """
        rope = self.rope
        patch_size_t = int(rope.patch_size_t)
        patch_size = int(rope.patch_size)
        f_positions = torch.tensor(frame_positions, dtype=torch.float32, device=device)
        if patch_size_t > 1:
            # Each patch covers ``patch_size_t`` latent frames; pick the start of each patch.
            f_positions = f_positions[::patch_size_t]
        grid_h = torch.arange(start=0, end=height, step=patch_size, dtype=torch.float32, device=device)
        grid_w = torch.arange(start=0, end=width, step=patch_size, dtype=torch.float32, device=device)
        grid = torch.meshgrid(f_positions, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)

        patch_size_delta = torch.tensor((patch_size_t, patch_size, patch_size), dtype=grid.dtype, device=device)
        patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)
        latent_coords = torch.stack([grid, patch_ends], dim=-1)
        latent_coords = latent_coords.flatten(1, 3).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        scale_tensor = torch.tensor(rope.scale_factors, device=device)
        broadcast_shape = [1] * latent_coords.ndim
        broadcast_shape[1] = -1
        pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)
        pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + rope.causal_offset - rope.scale_factors[0]).clamp(min=0)
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / float(fps)
        return rope(pixel_coords, device=device)

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        video_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        encoder_attention_mask: torch.Tensor | None = None,
        kv_cache: SanaWMRefinerKVCache | None = None,
        kv_cache_mode: str | None = None,
        attention_kwargs: dict | None = None,
        return_dict: bool = True,
    ):
        r"""
        Video-only forward pass over a single AR block.

        Args:
            hidden_states (`torch.Tensor`):
                Patchified video latents of the active block, of shape `(batch_size, num_video_tokens, in_channels)`.
            encoder_hidden_states (`torch.Tensor`):
                Text embeddings of shape `(batch_size, text_seq_len, caption_channels)`.
            timestep (`torch.Tensor`):
                Timestep of shape `(batch_size, num_video_tokens)`, already scaled by
                `self.config.timestep_scale_multiplier`.
            video_rotary_emb (`tuple[torch.Tensor, torch.Tensor]`):
                The `(cos, sin)` RoPE for the active block's absolute frame positions, as returned by
                [`~SanaWMLTX2RefinerTransformer3DModel.build_rotary_emb_for_absolute_positions`].
            encoder_attention_mask (`torch.Tensor`, *optional*):
                Multiplicative text attention mask of shape `(batch_size, text_seq_len)`.
            kv_cache (`SanaWMRefinerKVCache`, *optional*):
                Sliding-window KV cache holding the per-layer attention sink and recent refined history, plus the
                shared `sink_pe` RoPE. Required whenever `kv_cache_mode` is set.
            kv_cache_mode (`str`, *optional*):
                One of:

                - `"inject"`: attend to `[sink + history + current]` K/V (the denoising steps).
                - `"capture_pre_rope"`: no prefix; record the pre-RoPE K/V of this forward into the cache (used once to
                  seed the attention sink from the raw stage-1 latents).
                - `"inject_and_capture_post_rope"`: attend to `[sink + history + current]` K/V and record this block's
                  post-RoPE K/V into the cache so it can be appended to the history.

                When `None`, the block runs plain full self-attention over the current tokens only.
            attention_kwargs (`dict`, *optional*):
                Optional kwargs forwarded to the LoRA scale handling.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.modeling_outputs.Transformer2DModelOutput`] instead of a plain tuple.

        Returns:
            [`~models.modeling_outputs.Transformer2DModelOutput`] or `tuple`: the predicted velocity for the active
            block, of shape `(batch_size, num_video_tokens, out_channels)`.
        """
        if kv_cache_mode is not None:
            if kv_cache_mode not in ("inject", "capture_pre_rope", "inject_and_capture_post_rope"):
                raise ValueError(
                    "`kv_cache_mode` must be one of 'inject', 'capture_pre_rope', "
                    f"'inject_and_capture_post_rope' or `None`, got {kv_cache_mode!r}."
                )
            if kv_cache is None:
                raise ValueError(f"`kv_cache_mode={kv_cache_mode!r}` requires a `SanaWMRefinerKVCache`.")
            if len(kv_cache) != len(self.transformer_blocks):
                raise ValueError(
                    f"`kv_cache` holds {len(kv_cache)} layer caches but the model has "
                    f"{len(self.transformer_blocks)} transformer blocks."
                )

        batch_size = hidden_states.size(0)

        # Convert encoder_attention_mask to an additive bias.
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Patchification input projection
        hidden_states = self.proj_in(hidden_states)

        # 2. Timestep embedding and modulation parameters
        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        # 3. Prompt embeddings
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

        # 4. Transformer blocks
        sink_rotary_emb = kv_cache.sink_pe if kv_cache is not None else None
        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                video_rotary_emb=video_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                sink_rotary_emb=sink_rotary_emb,
                kv_cache=kv_cache.get(i) if kv_cache is not None else None,
                kv_cache_mode=kv_cache_mode,
            )

        # 5. Output norm and projection
        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
