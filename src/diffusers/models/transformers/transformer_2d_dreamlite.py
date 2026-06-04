# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
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
"""DreamLite 2D transformer.

This module is intentionally self-contained: it defines

* ``BasicTransformerBlockDreamLite`` — a DreamLite-flavoured variant of
  :class:`~diffusers.models.attention.BasicTransformerBlock` with four additional knobs (``use_self_attention``,
  ``qk_norm``, ``num_kv_heads``, ``ff_mult``); and
* ``DreamLiteTransformer2DModel`` — a continuous-input-only counterpart of
  :class:`~diffusers.models.transformers.transformer_2d.Transformer2DModel` that wires those knobs all the way down to
  each block.

Keeping everything here means the DreamLite integration never touches the upstream ``attention.py`` /
``transformer_2d.py``, which is the convention followed by other ported pipelines (SD3, Flux, Chroma, …).

The numerical behaviour mirrors the original DreamLite reference implementation at ``dreamlite/models/{attention.py,
transformers/transformer_2d.py}`` — specifically, when ``use_self_attention=False`` the block keeps ``norm1``'s output
as the post-self-attn hidden state instead of running ``attn1``, matching the "Remove self-attention" path used by
DreamLite's ``CrossAttnDownRemoveSelfAttnBlock2D`` and ``CrossAttnUpRemoveSelfAttnBlock2DV1``.
"""

from typing import Any

import torch
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..attention import FeedForward, GatedSelfAttentionDense, _chunked_feed_forward
from ..attention_processor import Attention
from ..embeddings import SinusoidalPositionalEmbedding
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero
from .transformer_2d import Transformer2DModelOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class BasicTransformerBlockDreamLite(nn.Module):
    r"""DreamLite variant of :class:`BasicTransformerBlock`.

    Adds four constructor knobs on top of the upstream block:

    * ``use_self_attention`` — when ``False``, ``attn1`` is *not* instantiated and the self-attention residual branch
      in ``forward`` is replaced by ``norm1``'s output (no add-residual). This implements DreamLite's "Remove
      self-attention" trick used inside ``DreamLiteCrossAttnNoSelfAttnDownBlock2D`` /
      ``DreamLiteCrossAttnNoSelfAttnUpBlock2D``.
    * ``qk_norm`` — propagated to both attention layers' ``qk_norm``.
    * ``num_kv_heads`` — propagated to both attention layers' ``kv_heads`` (enables Grouped-Query Attention).
    * ``ff_mult`` — propagated to :class:`FeedForward.mult` (DreamLite uses a non-default expansion factor).

    Only the ``norm_type`` values actually exercised by DreamLite are supported in detail (``layer_norm`` and
    ``ada_norm``); the other branches are preserved verbatim from the upstream block so that callers writing new
    variants do not have to re-port them.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: int | None = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: str | None = None,
        num_positional_embeddings: int | None = None,
        ada_norm_continous_conditioning_embedding_dim: int | None = None,
        ada_norm_bias: int | None = None,
        ff_inner_dim: int | None = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        use_self_attention: bool = True,
        qk_norm: str | None = None,
        num_kv_heads: int | None = None,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.double_self_attention = double_self_attention
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.only_cross_attention = only_cross_attention
        self.use_self_attention = use_self_attention

        if not use_self_attention and norm_type in ("ada_norm_zero", "ada_norm_single"):
            raise ValueError(
                f"`use_self_attention=False` is incompatible with `norm_type={norm_type}` because "
                "the gate/shift/scale modulation tuple is derived from `norm1`. "
                "Use `norm_type='layer_norm'` or `'ada_norm'` instead."
            )

        # Backward-compatible boolean flags (kept for parity with BasicTransformerBlock).
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. "
                f"Please make sure to define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # 1. Self-Attn (or its replacement)
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_zero":
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_continuous":
            self.norm1 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        if use_self_attention:
            self.attn1 = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                qk_norm=qk_norm,
                kv_heads=num_kv_heads,
            )
        else:
            self.attn1 = None

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            if norm_type == "ada_norm":
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif norm_type == "ada_norm_continuous":
                self.norm2 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            else:
                self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                qk_norm=qk_norm,
                kv_heads=num_kv_heads,
            )
        else:
            if norm_type == "ada_norm_single":
                self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
            else:
                self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if norm_type == "ada_norm_continuous":
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )
        elif norm_type in ["ada_norm_zero", "ada_norm", "layer_norm"]:
            self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        elif norm_type == "layer_norm_i2vgen":
            self.norm3 = None

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
            mult=ff_mult,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha (kept for completeness; DreamLite does not use it).
        if norm_type == "ada_norm_single":
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: int | None, dim: int = 0):
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        cross_attention_kwargs: dict[str, Any] = None,
        class_labels: torch.LongTensor | None = None,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # 0. Self-Attention norm
        batch_size = hidden_states.shape[0]

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. GLIGEN kwargs split
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        if self.use_self_attention:
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

            if self.norm_type == "ada_norm_zero":
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.norm_type == "ada_norm_single":
                attn_output = gate_msa * attn_output

            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)
        else:
            # DreamLite "Remove self-attention" path: drop attn1 entirely and let
            # the normalized state propagate as-is to cross-attn / FF. Matches
            # upstream DreamLite `BasicTransformerBlock.forward` when
            # `use_self_attention=False`.
            hidden_states = norm_hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class DreamLiteTransformer2DModel(ModelMixin, ConfigMixin):
    r"""Continuous-input 2D transformer used by the DreamLite U-Net.

    Equivalent to :class:`Transformer2DModel` restricted to the ``is_input_continuous`` branch (``in_channels`` set,
    ``patch_size`` and ``num_vector_embeds`` both ``None``), with four extra knobs that are propagated into every
    :class:`BasicTransformerBlockDreamLite`:

    * ``use_self_attention`` — set ``False`` from ``CrossAttn*RemoveSelfAttnBlock2D*DreamLite`` to enable DreamLite's
      "Remove self-attention" path.
    * ``qk_norm`` — RMS/LayerNorm applied to Q and K projections.
    * ``num_kv_heads`` — enables Grouped-Query Attention when fewer than ``num_attention_heads``.
    * ``ff_mult`` — feed-forward expansion factor (DreamLite uses a non-default value).
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlockDreamLite"]
    _skip_layerwise_casting_patterns = ["norm"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: int | None = None,
        out_channels: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: int | None = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: int | None = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        use_self_attention: bool = True,
        qk_norm: str | None = None,
        num_kv_heads: int | None = None,
        ff_mult: int = 4,
    ):
        super().__init__()

        if in_channels is None:
            raise ValueError(
                "`DreamLiteTransformer2DModel` only supports continuous inputs; `in_channels` must be provided."
            )

        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False

        self.norm = torch.nn.GroupNorm(
            num_groups=self.config.norm_num_groups, num_channels=self.in_channels, eps=1e-6, affine=True
        )
        if self.use_linear_projection:
            self.proj_in = torch.nn.Linear(self.in_channels, self.inner_dim)
        else:
            self.proj_in = torch.nn.Conv2d(self.in_channels, self.inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlockDreamLite(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    only_cross_attention=self.config.only_cross_attention,
                    double_self_attention=self.config.double_self_attention,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                    use_self_attention=self.config.use_self_attention,
                    qk_norm=self.config.qk_norm,
                    num_kv_heads=self.config.num_kv_heads,
                    ff_mult=self.config.ff_mult,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        if self.use_linear_projection:
            self.proj_out = torch.nn.Linear(self.inner_dim, self.out_channels)
        else:
            self.proj_out = torch.nn.Conv2d(self.inner_dim, self.out_channels, kernel_size=1, stride=1, padding=0)

    def _operate_on_continuous_inputs(self, hidden_states):
        batch, _, height, width = hidden_states.shape
        hidden_states = self.norm(hidden_states)

        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        return hidden_states, inner_dim

    def _get_output_for_continuous_inputs(self, hidden_states, residual, batch_size, height, width, inner_dim):
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        added_cond_kwargs: dict[str, torch.Tensor] = None,
        class_labels: torch.LongTensor | None = None,
        cross_attention_kwargs: dict[str, Any] = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        """Forward pass of :class:`DreamLiteTransformer2DModel`.

        Args:
            hidden_states: Input latent tensor of shape ``(batch, channels, height, width)``.
            encoder_hidden_states: Cross-attention conditioning embeddings.
            timestep: Diffusion timestep(s); broadcast to batch if scalar.
            added_cond_kwargs: Optional extra conditioning (e.g. ``text_embeds``, ``time_ids``).
            class_labels: Optional class labels for class-conditional generation.
            cross_attention_kwargs: Optional kwargs forwarded to the cross-attention processor.
                Note: passing ``scale`` is deprecated and will be ignored.
            attention_mask: Optional self-attention mask; 2D masks are converted to additive biases.
            encoder_attention_mask: Optional cross-attention mask; 2D masks are converted to additive biases.
            return_dict: If ``True``, returns a :class:`Transformer2DModelOutput`; otherwise a 1-tuple ``(sample,)``.

        Returns:
            :class:`~diffusers.models.transformers.transformer_2d.Transformer2DModelOutput` (or a 1-tuple of the
            sample) — kept output-compatible with the upstream class so callers don't have to special-case DreamLite.
        """
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # Convert masks to additive biases (broadcast-friendly).
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size, _, height, width = hidden_states.shape
        residual = hidden_states
        hidden_states, inner_dim = self._operate_on_continuous_inputs(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # 3. Output
        output = self._get_output_for_continuous_inputs(
            hidden_states=hidden_states,
            residual=residual,
            batch_size=batch_size,
            height=height,
            width=width,
            inner_dim=inner_dim,
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
