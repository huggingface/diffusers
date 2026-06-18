# Copyright 2025 Microsoft and The HuggingFace Team. All rights reserved.
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
from ...utils.torch_utils import lru_cache_unless_export, maybe_allow_in_graph
from .._modeling_parallel import ContextParallelInput, ContextParallelOutput
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous, RMSNorm


logger = logging.get_logger(__name__)


def apply_rotary_emb_lens(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_out.type_as(x)


class GateMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LensTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        proj = self.time_proj(timestep)
        return self.timestep_embedder(proj.to(dtype=hidden_states.dtype))


class LensEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope: bool = False) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.register_buffer(
            "pos_freqs",
            torch.cat([self._rope_params(pos_index, d, theta) for d in axes_dim], dim=1),
            persistent=False,
        )
        self.register_buffer(
            "neg_freqs",
            torch.cat([self._rope_params(neg_index, d, theta) for d in axes_dim], dim=1),
            persistent=False,
        )

    @staticmethod
    def _rope_params(index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        assert dim % 2 == 0
        freqs = torch.outer(
            index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).float().div(dim))
        )
        return torch.polar(torch.ones_like(freqs), freqs)

    @lru_cache_unless_export(maxsize=None)
    def _get_device_freqs(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pos_freqs.to(device), self.neg_freqs.to(device)

    def forward(
        self,
        video_fhw: list[tuple[int, int, int]] | tuple[int, int, int],
        txt_seq_len: int | torch.Tensor,
        device: torch.device = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(video_fhw, list) and len(video_fhw) > 1:
            first_fhw = video_fhw[0]
            if not all(fhw == first_fhw for fhw in video_fhw):
                logger.warning(
                    "Batch inference with variable-sized images is not currently supported in LensEmbedRope. "
                    "All images in the batch should have the same dimensions (frame, height, width). "
                    f"Detected sizes: {video_fhw}. Using the first image's dimensions {first_fhw} "
                    "for RoPE computation, which may lead to incorrect results for other images in the batch."
                )

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            video_freq = self._compute_video_freqs(frame, height, width, idx, device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_txt_seq_len_int = int(txt_seq_len)
        pos_freqs_device, _ = self._get_device_freqs(device)
        txt_freqs = pos_freqs_device[max_vid_index : max_vid_index + max_txt_seq_len_int, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @lru_cache_unless_export(maxsize=128)
    def _compute_video_freqs(
        self, frame: int, height: int, width: int, idx: int = 0, device: torch.device = None
    ) -> torch.Tensor:
        seq_lens = frame * height * width
        pos_freqs, neg_freqs = (
            self._get_device_freqs(device) if device is not None else (self.pos_freqs, self.neg_freqs)
        )

        freqs_pos = pos_freqs.split([d // 2 for d in self.axes_dim], dim=1)
        freqs_neg = neg_freqs.split([d // 2 for d in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0
            )
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0
            )
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class LensDoubleStreamAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: "LensJointAttention",
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.FloatTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_img, _ = hidden_states.shape
        seq_txt = encoder_hidden_states.shape[1]

        img_qkv = attn.img_qkv(hidden_states).view(bsz, seq_img, 3, attn.heads, attn.dim_head)
        txt_qkv = attn.txt_qkv(encoder_hidden_states).view(bsz, seq_txt, 3, attn.heads, attn.dim_head)
        img_q, img_k, img_v = img_qkv.unbind(dim=2)
        txt_q, txt_k, txt_v = txt_qkv.unbind(dim=2)

        if attn.norm_q is not None:
            img_q = attn.norm_q(img_q)
        if attn.norm_k is not None:
            img_k = attn.norm_k(img_k)
        if attn.norm_added_q is not None:
            txt_q = attn.norm_added_q(txt_q)
        if attn.norm_added_k is not None:
            txt_k = attn.norm_added_k(txt_k)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            if img_freqs.shape[0] < seq_img:
                raise ValueError(
                    f"Image RoPE length {img_freqs.shape[0]} is shorter than image sequence length {seq_img}."
                )
            img_freqs = img_freqs[:seq_img]
            img_q = apply_rotary_emb_lens(img_q, img_freqs)
            img_k = apply_rotary_emb_lens(img_k, img_freqs)
            if seq_txt > 0:
                if txt_freqs.shape[0] < seq_txt:
                    raise ValueError(
                        f"Text RoPE length {txt_freqs.shape[0]} is shorter than text sequence length {seq_txt}."
                    )
                txt_freqs = txt_freqs[:seq_txt]
                txt_q = apply_rotary_emb_lens(txt_q, txt_freqs)
                txt_k = apply_rotary_emb_lens(txt_k, txt_freqs)

        joint_query = torch.cat([img_q, txt_q], dim=1)
        joint_key = torch.cat([img_k, txt_k], dim=1)
        joint_value = torch.cat([img_v, txt_v], dim=1)

        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        img_attn_output = joint_hidden_states[:, :seq_img, :]
        txt_attn_output = joint_hidden_states[:, seq_img:, :]

        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output


class LensJointAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = LensDoubleStreamAttnProcessor
    _available_processors = [LensDoubleStreamAttnProcessor]

    def __init__(
        self,
        query_dim: int,
        added_kv_proj_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        out_dim: int | None = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = self.inner_dim // dim_head
        self.dim_head = dim_head
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.norm_added_q = RMSNorm(dim_head, eps=eps)
        self.norm_added_k = RMSNorm(dim_head, eps=eps)

        self.img_qkv = nn.Linear(query_dim, 3 * self.inner_dim, bias=True)
        self.txt_qkv = nn.Linear(added_kv_proj_dim, 3 * self.inner_dim, bias=True)

        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, self.out_dim, bias=True), nn.Identity()])
        self.to_add_out = nn.Linear(self.inner_dim, query_dim, bias=True)

        self.set_processor(LensDoubleStreamAttnProcessor())

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.processor(
            self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )


@maybe_allow_in_graph
class LensTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        rms_norm: bool = False,
        gate_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.attn = LensJointAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            eps=eps,
        )

        norm_cls = (lambda d: RMSNorm(d, eps=eps)) if rms_norm else (
            lambda d: nn.LayerNorm(d, elementwise_affine=False, eps=eps)
        )
        if gate_mlp:
            mlp_cls = lambda: GateMLP(dim, int(dim / 3 * 8))
        else:
            mlp_cls = lambda: FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.img_norm1 = norm_cls(dim)
        self.img_norm2 = norm_cls(dim)
        self.img_mlp = mlp_cls()

        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.txt_norm1 = norm_cls(dim)
        self.txt_norm2 = norm_cls(dim)
        self.txt_mlp = mlp_cls()

    @staticmethod
    def _modulate(x: torch.Tensor, mod_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_mod1, img_mod2 = self.img_mod(temb).chunk(2, dim=-1)
        txt_mod1, txt_mod2 = self.txt_mod(temb).chunk(2, dim=-1)

        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1)
        txt_modulated, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)

        joint_attention_kwargs = joint_attention_kwargs or {}
        block_attention_mask = joint_attention_kwargs.pop("attention_mask", attention_mask)
        img_attn, txt_attn = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            attention_mask=block_attention_mask,
            **joint_attention_kwargs,
        )

        hidden_states = hidden_states + img_gate1 * img_attn
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2)
        hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated2)

        txt_modulated2, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2)

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class LensTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    """
    The Transformer model introduced in Lens.

    Reference: https://huggingface.co/microsoft/Lens

    Args:
        patch_size (`int`, defaults to `2`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `128`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `32`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `48`):
            The number of layers of dual stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `64`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        inner_dim (`int`, defaults to `1536`):
            The inner dimension of the transformer. If not specified, it defaults to
            `num_attention_heads * attention_head_dim`.
        enc_hidden_dim (`int`, defaults to `2880`):
            The hidden dimension of the text encoder outputs.
        axes_dims_rope (`tuple[int, int, int]`, defaults to `(8, 28, 28)`):
            The dimensions to use for the rotary positional embeddings for frame, height, and width axes.
        gate_mlp (`bool`, defaults to `True`):
            Whether to use a gated MLP (SwiGLU) in the transformer blocks.
        rms_norm (`bool`, defaults to `True`):
            Whether to use RMS normalization instead of LayerNorm in the transformer blocks.
        multi_layer_encoder_feature (`bool`, defaults to `True`):
            Whether to use multi-layer text encoder features by selecting specific layers.
        selected_layer_index (`tuple[int, ...]`, defaults to `(5, 11, 17, 23)`):
            The indices of the text encoder layers to select when `multi_layer_encoder_feature` is True.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["LensTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["LensTransformerBlock"]
    _cp_plan = {
        "transformer_blocks.0": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "pos_embed": {
            0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
            1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 128,
        out_channels: int | None = 32,
        num_layers: int = 48,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        inner_dim: int = 1536,
        enc_hidden_dim: int = 2880,
        axes_dims_rope: tuple[int, int, int] = (8, 28, 28),
        gate_mlp: bool = True,
        rms_norm: bool = True,
        multi_layer_encoder_feature: bool = True,
        selected_layer_index: tuple[int, ...] = (5, 11, 17, 23),
    ) -> None:
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.multi_layer_encoder_feature = multi_layer_encoder_feature
        self.selected_layer_index = list(selected_layer_index)

        self.pos_embed = LensEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)
        self.time_text_embed = LensTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        if self.multi_layer_encoder_feature:
            self.txt_norm = nn.ModuleList(
                [RMSNorm(enc_hidden_dim, eps=1e-5) for _ in self.selected_layer_index]
            )
            self.txt_in = nn.Linear(enc_hidden_dim * len(self.selected_layer_index), self.inner_dim)
        else:
            self.txt_norm = RMSNorm(enc_hidden_dim, eps=1e-5)
            self.txt_in = nn.Linear(enc_hidden_dim, self.inner_dim)

        self.img_in = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                LensTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    rms_norm=rms_norm,
                    gate_mlp=gate_mlp,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        encoder_hidden_states_mask: torch.Tensor | None = None,
        timestep: torch.LongTensor = None,
        img_shapes: list[tuple[int, int, int]] | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,

    ) -> torch.Tensor | Transformer2DModelOutput:
        """
        The [`LensTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` or `list[torch.Tensor]`):
                Conditional embeddings computed from the input conditions such as prompts. When
                `multi_layer_encoder_feature` is True, a list of per-layer text tensors is expected.
            encoder_hidden_states_mask (`torch.Tensor`, *optional*):
                Boolean mask for the encoder hidden states, where `True` indicates valid tokens.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            img_shapes (`list[tuple[int, int, int]]`, *optional*):
                List of (frame, height, width) tuples for each image in the batch, used to compute
                rotary positional embeddings.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        bsz, img_len, _ = hidden_states.shape

        if self.multi_layer_encoder_feature:
            if not isinstance(encoder_hidden_states, (list, tuple)):
                raise ValueError(
                    "multi_layer_encoder_feature=True expects a list of per-layer text tensors."
                )
            if len(encoder_hidden_states) != len(self.selected_layer_index):
                raise ValueError(
                    f"Expected {len(self.selected_layer_index)} text feature layers, "
                    f"got {len(encoder_hidden_states)}."
                )
            text_seq_len = encoder_hidden_states[0].shape[1]
        else:
            if not isinstance(encoder_hidden_states, torch.Tensor):
                raise ValueError(
                    "multi_layer_encoder_feature=False expects a single text feature tensor."
                )
            text_seq_len = encoder_hidden_states.shape[1]

        attention_mask = None
        if encoder_hidden_states_mask is not None:
            attention_mask = self._build_joint_attention_mask(encoder_hidden_states_mask, img_len)

        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        if self.multi_layer_encoder_feature:
            normed = [
                self.txt_norm[i](encoder_hidden_states[i])
                for i in range(len(self.selected_layer_index))
            ]
            encoder_hidden_states = torch.cat(normed, dim=-1)
        else:
            encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        temb = self.time_text_embed(timestep, hidden_states)

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_len=text_seq_len, device=hidden_states.device)

        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        if attention_mask is not None:
            block_attention_kwargs["attention_mask"] = attention_mask

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    None,
                    block_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=block_attention_kwargs,
                )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    @staticmethod
    def _build_joint_attention_mask(
        text_mask: torch.Tensor, img_len: int
    ) -> torch.Tensor:
        if text_mask.dtype != torch.bool:
            text_mask = text_mask.bool()
        bsz = text_mask.shape[0]
        img_ones = torch.ones((bsz, img_len), dtype=torch.bool, device=text_mask.device)
        joint = torch.cat([img_ones, text_mask], dim=1)
        additive = torch.zeros_like(joint, dtype=torch.float32)
        additive.masked_fill_(~joint, float("-inf"))
        return additive[:, None, None, :]
