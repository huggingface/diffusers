# Copyright 2026 Black Forest Labs and The HuggingFace Team. All rights reserved.
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

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import apply_lora_scale, logging
from ...utils.torch_utils import maybe_adjust_dtype_for_device
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..embeddings import TimestepEmbedding, Timesteps, apply_rotary_emb, get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.models.transformers.transformer_flux2.Flux2SwiGLU with Flux2->Flux3
class Flux3SwiGLU(nn.Module):
    """
    Flux 2 uses a SwiGLU-style activation in the transformer feedforward sub-blocks, but with the linear projection
    layer fused into the first linear layer of the FF sub-block. Thus, this module has no trainable parameters.
    """

    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x = self.gate_fn(x[..., :half]) * x[..., half:]
        return x


# Copied from diffusers.models.transformers.transformer_flux2.Flux2FeedForward with Flux2->Flux3
class Flux3FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 3.0,
        inner_dim: int | None = None,
        bias: bool = False,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        # Flux3SwiGLU will reduce the dimension by half
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.act_fn = Flux3SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x


class Flux3Modulation(nn.Module):
    """Produces one set of per-token shift/scale/gate modulation parameters from a pre-activated per-token
    conditioning embedding of shape `(batch_size, seq_len, dim)`. The per-mode and the joint heads of a token mode
    share one activated embedding (activated once by the caller), so this module only holds the projection."""

    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 3, bias=bias)

    def forward(self, temb: torch.Tensor) -> torch.Tensor:
        return self.linear(temb)


def _get_projections(attn: "Flux3Attention", hidden_states: torch.Tensor):
    return attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)


def _get_fused_projections(attn: "Flux3Attention", hidden_states: torch.Tensor):
    return attn.to_qkv(hidden_states).chunk(3, dim=-1)


def _get_qkv_projections(attn: "Flux3Attention", hidden_states: torch.Tensor):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states)
    return _get_projections(attn, hidden_states)


class Flux3AttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "Flux3Attention",
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query, key, value = _get_qkv_projections(attn, hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query.float())
        key = attn.norm_k(key.float())

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        query, key = query.to(value.dtype), key.to(value.dtype)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class Flux3Attention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = Flux3AttnProcessor
    _available_processors = [Flux3AttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        eps: float = 1e-6,
        processor=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.heads = heads
        self.dropout = dropout
        self.use_bias = False

        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=False)

        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=True)

        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, query_dim, bias=False))
        self.to_out.append(torch.nn.Dropout(dropout))

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, attention_mask, image_rotary_emb)


class Flux3ModeTransformerBlock(nn.Module):
    """
    Per-mode transformer block: parallel attention + feedforward computed from the same modulated input, with
    self-attention restricted to a single token mode (the image, the reference images, or the text context).
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Flux3Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            eps=eps,
        )
        self.ff = Flux3FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb_mod: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # modulation is split here rather than in Flux3Modulation, to avoid passing tuples into checkpoints https://github.com/huggingface/diffusers/issues/12776
        mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim=-1)

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        attn_output = self.attn(hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb)
        ff_output = self.ff(norm_hidden_states)

        return hidden_states + mod_gate * (attn_output + ff_output)


class Flux3SingleTransformerBlock(nn.Module):
    """
    Joint transformer block: text and content tokens are concatenated into a single sequence for parallel attention +
    feedforward, with separate per-token modulation for the text and content tokens.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Flux3Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            eps=eps,
        )
        self.ff = Flux3FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod: torch.Tensor,
        temb_mod_context: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # modulation is split here rather than in Flux3Modulation, to avoid passing tuples into checkpoints https://github.com/huggingface/diffusers/issues/12776
        mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim=-1)
        c_mod_shift, c_mod_scale, c_mod_gate = torch.chunk(temb_mod_context, 3, dim=-1)

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        norm_encoder_hidden_states = self.norm(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_mod_scale) * norm_encoder_hidden_states + c_mod_shift

        combined = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)

        attn_output = self.attn(hidden_states=combined, image_rotary_emb=image_rotary_emb)
        ff_output = self.ff(combined)
        output = attn_output + ff_output

        context_output, content_output = torch.split(
            output, [encoder_hidden_states.shape[1], hidden_states.shape[1]], dim=1
        )
        encoder_hidden_states = encoder_hidden_states + c_mod_gate * context_output
        hidden_states = hidden_states + mod_gate * content_output

        return encoder_hidden_states, hidden_states


# Copied from diffusers.models.transformers.transformer_flux2.Flux2PosEmbed with Flux2->Flux3
class Flux3PosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # Expected ids shape: [S, len(self.axes_dim)]
        cos_out = []
        sin_out = []
        pos = ids.float()
        freqs_dtype = maybe_adjust_dtype_for_device(torch.float64, ids.device)
        # Unlike Flux 1, loop over len(self.axes_dim) rather than ids.shape[-1]
        for i in range(len(self.axes_dim)):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[..., i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class Flux3AdaLayerNorm(nn.Module):
    """Output adaptive layer norm conditioned on a per-token embedding of shape `(batch_size, seq_len, dim)`."""

    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.act_fn = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2, bias=bias)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.act_fn(temb))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        return self.norm(hidden_states) * (1 + scale) + shift


class Flux3Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, AttentionMixin):
    """
    The Transformer model introduced in FLUX.3 for text-to-image and image-to-image generation.

    Three token modes share one joint sequence: the noisy image tokens (`hidden_states`), the optional VAE-encoded
    reference image tokens (`image_cond_hidden_states`) and the text context (`encoder_hidden_states`). Each mode first
    passes through its own per-mode transformer blocks (`mode_blocks` for the image modes, `txt_mode_blocks` for the
    text context), then the text context and the active image modes are concatenated for the joint single blocks, and
    only the image tokens pass through the output layers. The modulation is per mode: the image tokens are modulated by
    the flow timestep, the reference image tokens and the text context by timestep `0`. A flow-map checkpoint
    (`flow_map=True`, the step-distilled models) additionally embeds the timestep each denoising step jumps to and adds
    it to every mode's modulation.

    Args:
        in_channels (`int`, defaults to `128`):
            Channel width of the image tokens (the packed VAE latents, `latent_channels * 4`).
        cond_in_channels (`int`, defaults to `128`):
            Channel width of the reference image tokens.
        num_layers (`int`, defaults to `5`):
            The number of per-mode transformer blocks.
        num_single_layers (`int`, defaults to `28`):
            The number of joint single transformer blocks.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `20480`):
            The embedding/channel dimension of `encoder_hidden_states` (stacked Qwen3-VL hidden states).
        timestep_channels (`int`, defaults to `256`):
            The number of channels of the sinusoidal timestep embedding.
        mlp_ratio (`float`, defaults to `3.0`):
            The expansion ratio of the feedforward sub-blocks.
        axes_dims_rope (`tuple[int]`, defaults to `(32, 32, 32, 32)`):
            The dimensions to use for the rotary positional embeddings, per `(t, h, w, l)` position axis.
        rope_theta (`int`, defaults to `10000`):
            The base frequency of the rotary positional embeddings.
        eps (`float`, defaults to `1e-6`):
            The epsilon of the layer norms and the QK norms.
        flow_map (`bool`, defaults to `False`):
            Whether the model is a flow map (step-distilled): adds the `target_timestep_embedder` and requires
            `target_timestep` in the forward pass.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Flux3ModeTransformerBlock", "Flux3SingleTransformerBlock"]
    # QK norm and RoPE run in fp32 with a single cast back, matching the BFL implementation
    _keep_in_fp32_modules = ["norm_q", "norm_k"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["Flux3ModeTransformerBlock", "Flux3SingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        cond_in_channels: int = 128,
        num_layers: int = 5,
        num_single_layers: int = 28,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 20480,
        timestep_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 10000,
        eps: float = 1e-6,
        flow_map: bool = False,
    ):
        super().__init__()

        self.inner_dim = num_attention_heads * attention_head_dim
        in_channels_per_mode = {"image": in_channels, "image_cond": cond_in_channels}
        modes = list(in_channels_per_mode.keys())

        # 1. Sinusoidal positional embedding for RoPE on all token modes
        self.pos_embed = Flux3PosEmbed(theta=rope_theta, axes_dim=list(axes_dims_rope))

        # 2. Per-token timestep embedding
        self.time_proj = Timesteps(num_channels=timestep_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=timestep_channels, time_embed_dim=self.inner_dim, sample_proj_bias=False
        )
        # flow map (step-distilled checkpoints): a second embedder for the timestep each step jumps to
        self.target_timestep_embedder = (
            TimestepEmbedding(in_channels=timestep_channels, time_embed_dim=self.inner_dim, sample_proj_bias=False)
            if flow_map
            else None
        )

        # 3. Modulation: one head per mode (+ text) for the per-mode blocks and for the joint blocks
        self.early_stream_modulations = nn.ModuleDict(
            {name: Flux3Modulation(self.inner_dim, bias=False) for name in [*modes, "txt"]}
        )
        self.single_stream_modulations = nn.ModuleDict(
            {name: Flux3Modulation(self.inner_dim, bias=False) for name in [*modes, "txt"]}
        )

        # 4. Input projections
        self.x_embedders = nn.ModuleDict(
            {name: nn.Linear(channels, self.inner_dim, bias=False) for name, channels in in_channels_per_mode.items()}
        )
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        # 5. Per-mode transformer blocks
        self.mode_blocks = nn.ModuleDict(
            {
                name: nn.ModuleList(
                    [
                        Flux3ModeTransformerBlock(
                            dim=self.inner_dim,
                            num_attention_heads=num_attention_heads,
                            attention_head_dim=attention_head_dim,
                            mlp_ratio=mlp_ratio,
                            eps=eps,
                        )
                        for _ in range(num_layers)
                    ]
                )
                for name in modes
            }
        )
        self.txt_mode_blocks = nn.ModuleList(
            [
                Flux3ModeTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Joint single transformer blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux3SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 7. Output layers (image tokens only)
        self.norm_out = Flux3AdaLayerNorm(self.inner_dim, eps=eps)
        self.proj_out = nn.Linear(self.inner_dim, in_channels, bias=False)

        self.gradient_checkpointing = False

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        image_cond_hidden_states: torch.Tensor | None = None,
        image_cond_ids: torch.Tensor | None = None,
        target_timestep: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        """
        The [`Flux3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_seq_len, in_channels)`):
                Noisy image latent tokens.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_seq_len, joint_attention_dim)`):
                Text context embeddings (stacked Qwen3-VL hidden states).
            timestep (`torch.Tensor` of shape `(batch_size,)`):
                Flow timestep in `[0, 1]` of the image tokens. The reference image tokens and the text context sit at
                timestep `0`.
            img_ids (`torch.Tensor` of shape `(image_seq_len, 4)`):
                Position ids `(t, h, w, l)` of the image tokens, used for the rotary positional embeddings and shared
                by all samples in the batch.
            txt_ids (`torch.Tensor` of shape `(text_seq_len, 4)`):
                Position ids of the text context tokens.
            image_cond_hidden_states (`torch.Tensor` of shape `(batch_size, cond_seq_len, cond_in_channels)`, *optional*):
                VAE-encoded reference image tokens (clean conditioning content, for image-to-image generation).
            image_cond_ids (`torch.Tensor` of shape `(cond_seq_len, 4)`, *optional*):
                Position ids of the reference image tokens. Required with `image_cond_hidden_states`.
            target_timestep (`torch.Tensor` of shape `(batch_size,)`, *optional*):
                Flow timestep in `[0, 1]` that the current denoising step jumps to. Required by, and only by, a
                flow-map model (`flow_map=True`).
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary whose `scale` entry, if specified, scales the LoRA layers for this forward pass.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the predicted velocity of the image tokens, of the same shape as
            `hidden_states`.
        """
        if image_cond_hidden_states is not None and image_cond_ids is None:
            raise ValueError("`image_cond_hidden_states` requires `image_cond_ids`.")
        if image_cond_hidden_states is None and image_cond_ids is not None:
            raise ValueError("`image_cond_ids` was passed without `image_cond_hidden_states`.")
        if (target_timestep is not None) != bool(self.config.flow_map):
            raise ValueError(
                "`target_timestep` is required by, and only by, a flow-map transformer (`flow_map=True`)."
            )

        batch_size, image_seq_len = hidden_states.shape[:2]
        dtype = hidden_states.dtype

        # 1. Timestep embeddings: the image tokens at `timestep`, the text context and the reference tokens at 0.
        #    Each is a per-token embedding (batch_size, seq_len, dim) since the modulation is per token.
        image_temb = self.time_proj(timestep.to(dtype) * 1000)
        image_temb = self.timestep_embedder(image_temb.to(dtype))[:, None, :]
        zero_temb = self.time_proj(torch.zeros(batch_size, device=hidden_states.device, dtype=dtype))
        zero_temb = self.timestep_embedder(zero_temb.to(dtype))[:, None, :]
        if target_timestep is not None:
            # flow map: the embedding of the timestep this step jumps to is added to every mode's embedding, so it
            # reaches the per-mode blocks, the joint blocks and the output layer
            target_temb = self.time_proj(target_timestep.to(dtype) * 1000)
            target_temb = self.target_timestep_embedder(target_temb.to(dtype))[:, None, :]
            image_temb = image_temb + target_temb
            zero_temb = zero_temb + target_temb
        image_temb = image_temb.expand(batch_size, image_seq_len, -1)
        zero_temb_act = F.silu(zero_temb)

        # 2. Text context: the two modulation sets, input projection and RoPE
        txt_temb_act = zero_temb_act.expand(batch_size, encoder_hidden_states.shape[1], -1)
        txt_early_mod = self.early_stream_modulations["txt"](txt_temb_act)
        txt_single_mod = self.single_stream_modulations["txt"](txt_temb_act)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        txt_rotary_emb = self.pos_embed(txt_ids)

        # 3. Image tokens
        image_temb_act = F.silu(image_temb)
        image_early_mod = self.early_stream_modulations["image"](image_temb_act)
        image_single_mod = self.single_stream_modulations["image"](image_temb_act)
        hidden_states = self.x_embedders["image"](hidden_states)
        image_rotary_emb = self.pos_embed(img_ids)

        # 4. Reference image tokens (optional)
        if image_cond_hidden_states is not None:
            cond_temb_act = zero_temb_act.expand(batch_size, image_cond_hidden_states.shape[1], -1)
            cond_early_mod = self.early_stream_modulations["image_cond"](cond_temb_act)
            cond_single_mod = self.single_stream_modulations["image_cond"](cond_temb_act)
            image_cond_hidden_states = self.x_embedders["image_cond"](image_cond_hidden_states)
            cond_rotary_emb = self.pos_embed(image_cond_ids)

        # 5. Per-mode transformer blocks (each mode self-attends within itself only)
        for block in self.txt_mode_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states = self._gradient_checkpointing_func(
                    block, encoder_hidden_states, txt_early_mod, txt_rotary_emb
                )
            else:
                encoder_hidden_states = block(encoder_hidden_states, txt_early_mod, txt_rotary_emb)

        for block in self.mode_blocks["image"]:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, image_early_mod, image_rotary_emb
                )
            else:
                hidden_states = block(hidden_states, image_early_mod, image_rotary_emb)

        if image_cond_hidden_states is not None:
            for block in self.mode_blocks["image_cond"]:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    image_cond_hidden_states = self._gradient_checkpointing_func(
                        block, image_cond_hidden_states, cond_early_mod, cond_rotary_emb
                    )
                else:
                    image_cond_hidden_states = block(image_cond_hidden_states, cond_early_mod, cond_rotary_emb)

        # 6. Joint single transformer blocks over [txt, image, image_cond]
        if image_cond_hidden_states is not None:
            content_states = torch.cat([hidden_states, image_cond_hidden_states], dim=1)
            content_mod = torch.cat([image_single_mod, cond_single_mod], dim=1)
            joint_rotary_emb = (
                torch.cat([txt_rotary_emb[0], image_rotary_emb[0], cond_rotary_emb[0]], dim=0),
                torch.cat([txt_rotary_emb[1], image_rotary_emb[1], cond_rotary_emb[1]], dim=0),
            )
        else:
            content_states = hidden_states
            content_mod = image_single_mod
            joint_rotary_emb = (
                torch.cat([txt_rotary_emb[0], image_rotary_emb[0]], dim=0),
                torch.cat([txt_rotary_emb[1], image_rotary_emb[1]], dim=0),
            )

        for block in self.single_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, content_states = self._gradient_checkpointing_func(
                    block, content_states, encoder_hidden_states, content_mod, txt_single_mod, joint_rotary_emb
                )
            else:
                encoder_hidden_states, content_states = block(
                    hidden_states=content_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod=content_mod,
                    temb_mod_context=txt_single_mod,
                    image_rotary_emb=joint_rotary_emb,
                )

        # 7. Output layers: the image tokens sit first in the content sequence; the reference tokens produce no output
        hidden_states = content_states[:, :image_seq_len]
        hidden_states = self.norm_out(hidden_states, image_temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
