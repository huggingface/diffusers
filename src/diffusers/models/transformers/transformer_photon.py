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

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.nn.functional import fold, unfold

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ..attention_processor import Attention, AttentionProcessor
from ..embeddings import get_timestep_embedding
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


logger = logging.get_logger(__name__)


def get_image_ids(batch_size: int, height: int, width: int, patch_size: int, device: torch.device) -> Tensor:
    r"""
    Generates 2D patch coordinate indices for a batch of images.

    Parameters:
        batch_size (`int`):
            Number of images in the batch.
        height (`int`):
            Height of the input images (in pixels).
        width (`int`):
            Width of the input images (in pixels).
        patch_size (`int`):
            Size of the square patches that the image is divided into.
        device (`torch.device`):
            The device on which to create the tensor.

    Returns:
        `torch.Tensor`:
            Tensor of shape `(batch_size, num_patches, 2)` containing the (row, col) coordinates of each patch in the
            image grid.
    """

    img_ids = torch.zeros(height // patch_size, width // patch_size, 2, device=device)
    img_ids[..., 0] = torch.arange(height // patch_size, device=device)[:, None]
    img_ids[..., 1] = torch.arange(width // patch_size, device=device)[None, :]
    return img_ids.reshape((height // patch_size) * (width // patch_size), 2).unsqueeze(0).repeat(batch_size, 1, 1)


def apply_rope(xq: Tensor, freqs_cis: Tensor) -> Tensor:
    r"""
    Applies rotary positional embeddings (RoPE) to a query tensor.

    Parameters:
        xq (`torch.Tensor`):
            Input tensor of shape `(..., dim)` representing the queries.
        freqs_cis (`torch.Tensor`):
            Precomputed rotary frequency components of shape `(..., dim/2, 2)` containing cosine and sine pairs.

    Returns:
        `torch.Tensor`:
            Tensor of the same shape as `xq` with rotary embeddings applied.
    """
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq)

class PhotonAttnProcessor2_0:
    r"""
    Processor for implementing Photon-style attention with multi-source tokens and RoPE. Properly integrates with
    diffusers Attention module while handling Photon-specific logic.
    """

    def __init__(self):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("PhotonAttnProcessor2_0 requires PyTorch 2.0, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: "Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply Photon attention using standard diffusers interface.

        Expected tensor formats from PhotonBlock.attn_forward():
        - hidden_states: Image queries with RoPE applied [B, H, L_img, D]
        - encoder_hidden_states: Packed key+value tensors [B, H, L_all, 2*D] (concatenated keys and values from text +
          image + spatial conditioning)
        - attention_mask: Custom attention mask [B, H, L_img, L_all] or None
        """

        if encoder_hidden_states is None:
            raise ValueError(
                "PhotonAttnProcessor2_0 requires 'encoder_hidden_states' containing packed key+value tensors. "
                "This should be provided by PhotonBlock.attn_forward()."
            )

        # Unpack the combined key+value tensor
        # encoder_hidden_states is [B, H, L_all, 2*D] containing [keys, values]
        key, value = encoder_hidden_states.chunk(2, dim=-1)  # Each [B, H, L_all, D]

        # Apply scaled dot-product attention with Photon's processed tensors
        # hidden_states is image queries [B, H, L_img, D]
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            hidden_states.contiguous(), key.contiguous(), value.contiguous(), attn_mask=attention_mask
        )

        # Reshape from [B, H, L_img, D] to [B, L_img, H*D]
        batch_size, num_heads, seq_len, head_dim = attn_output.shape
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)

        # Apply output projection
        attn_output = attn.to_out[0](attn_output)
        if len(attn.to_out) > 1:
            attn_output = attn.to_out[1](attn_output)  # dropout if present

        return attn_output
class EmbedND(nn.Module):
    r"""
    N-dimensional rotary positional embedding.

    This module creates rotary embeddings (RoPE) across multiple axes, where each axis can have its own embedding
    dimension. The embeddings are combined and returned as a single tensor

    Parameters:
        dim (int):
        Base embedding dimension (must be even).
        theta (int):
        Scaling factor that controls the frequency spectrum of the rotary embeddings.
        axes_dim (list[int]):
        List of embedding dimensions for each axis (each must be even).
    """

    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        self.rope_rearrange = Rearrange("b n d (i j) -> b n d i j", i=2, j=2)

    def rope(self, pos: Tensor, dim: int, theta: int) -> Tensor:
        assert dim % 2 == 0
        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
        omega = 1.0 / (theta**scale)
        out = pos.unsqueeze(-1) * omega.unsqueeze(0)
        out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
        out = self.rope_rearrange(out)
        return out.float()

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [self.rope(ids[:, :, i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    r"""
    A simple 2-layer MLP used for embedding inputs.

    Parameters:
        in_dim (`int`):
            Dimensionality of the input features.
        hidden_dim (`int`):
            Dimensionality of the hidden and output embedding space.

    Returns:
        `torch.Tensor`:
            Tensor of shape `(..., hidden_dim)` containing the embedded representations.
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class QKNorm(torch.nn.Module):
    r"""
    Applies RMS normalization to query and key tensors separately before attention which can help stabilize training
    and improve numerical precision.

    Parameters:
        dim (`int`):
            Dimensionality of the query and key vectors.

    Returns:
        (`torch.Tensor`, `torch.Tensor`):
            A tuple `(q, k)` where both are normalized and cast to the same dtype as the value tensor `v`.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim, eps=1e-6)
        self.key_norm = RMSNorm(dim, eps=1e-6)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    r"""
    Modulation network that generates scale, shift, and gating parameters.

    Given an input vector, the module projects it through a linear layer to produce six chunks, which are grouped into
    two `ModulationOut` objects.

    Parameters:
        dim (`int`):
            Dimensionality of the input vector. The output will have `6 * dim` features internally.

    Returns:
        (`ModulationOut`, `ModulationOut`):
            A tuple of two modulation outputs. Each `ModulationOut` contains three components (e.g., scale, shift,
            gate).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 6 * dim, bias=True)
        nn.init.constant_(self.lin.weight, 0)
        nn.init.constant_(self.lin.bias, 0)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(6, dim=-1)
        return ModulationOut(*out[:3]), ModulationOut(*out[3:])


class PhotonBlock(nn.Module):
    r"""
    Multimodal transformer block with text–image cross-attention, modulation, and MLP.

    Parameters:
        hidden_size (`int`):
            Dimension of the hidden representations.
        num_heads (`int`):
            Number of attention heads.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Expansion ratio for the hidden dimension inside the MLP.
        qk_scale (`float`, *optional*):
            Scale factor for queries and keys. If not provided, defaults to ``head_dim**-0.5``.

    Attributes:
        img_pre_norm (`nn.LayerNorm`):
            Pre-normalization applied to image tokens before QKV projection.
        img_qkv_proj (`nn.Linear`):
            Linear projection to produce image queries, keys, and values.
        qk_norm (`QKNorm`):
            RMS normalization applied separately to image queries and keys.
        txt_kv_proj (`nn.Linear`):
            Linear projection to produce text keys and values.
        k_norm (`RMSNorm`):
            RMS normalization applied to text keys.
        attention (`Attention`):
            Multi-head attention module for cross-attention between image, text, and optional spatial conditioning
            tokens.
        post_attention_layernorm (`nn.LayerNorm`):
            Normalization applied after attention.
        gate_proj / up_proj / down_proj (`nn.Linear`):
            Feedforward layers forming the gated MLP.
        mlp_act (`nn.GELU`):
            Nonlinear activation used in the MLP.
        modulation (`Modulation`):
            Produces scale/shift/gating parameters for modulated layers.

    Methods:
        attn_forward(img, txt, pe, modulation, spatial_conditioning=None, attention_mask=None):
            Compute cross-attention between image and text tokens, with optional spatial conditioning and attention
            masking.

            Parameters:
                img (`torch.Tensor`):
                    Image tokens of shape `(B, L_img, hidden_size)`.
                txt (`torch.Tensor`):
                    Text tokens of shape `(B, L_txt, hidden_size)`.
                pe (`torch.Tensor`):
                    Rotary positional embeddings to apply to queries and keys.
                modulation (`ModulationOut`):
                    Scale and shift parameters for modulating image tokens.
                spatial_conditioning (`torch.Tensor`, *optional*):
                    Extra conditioning tokens of shape `(B, L_cond, hidden_size)`.
                attention_mask (`torch.Tensor`, *optional*):
                    Boolean mask of shape `(B, L_txt)` where 0 marks padding.

            Returns:
                `torch.Tensor`:
                    Attention output of shape `(B, L_img, hidden_size)`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.hidden_size = hidden_size

        # img qkv
        self.img_pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.qk_norm = QKNorm(self.head_dim)

        # txt kv
        self.txt_kv_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.attention = Attention(
            query_dim=hidden_size,
            heads=num_heads,
            dim_head=self.head_dim,
            bias=False,
            out_bias=False,
            processor=PhotonAttnProcessor2_0(),
        )

        # mlp
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.gate_proj = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.mlp_hidden_dim, hidden_size, bias=False)
        self.mlp_act = nn.GELU(approximate="tanh")

        self.modulation = Modulation(hidden_size)

    def _attn_forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        modulation: ModulationOut,
        spatial_conditioning: None | Tensor = None,
        attention_mask: None | Tensor = None,
    ) -> Tensor:
        # image tokens proj and norm
        img_mod = (1 + modulation.scale) * self.img_pre_norm(img) + modulation.shift

        img_qkv = self.img_qkv_proj(img_mod)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.qk_norm(img_q, img_k, img_v)

        # txt tokens proj and norm
        txt_kv = self.txt_kv_proj(txt)
        txt_k, txt_v = rearrange(txt_kv, "B L (K H D) -> K B H L D", K=2, H=self.num_heads)
        txt_k = self.k_norm(txt_k)

        # compute attention
        img_q, img_k = apply_rope(img_q, pe), apply_rope(img_k, pe)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        # build multiplicative 0/1 mask for provided attention_mask over [cond?, text, image] keys
        attn_mask: Tensor | None = None
        if attention_mask is not None:
            bs, _, l_img, _ = img_q.shape
            l_txt = txt_k.shape[2]

            if attention_mask.dim() != 2:
                raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")
            if attention_mask.shape[-1] != l_txt:
                raise ValueError(f"attention_mask last dim {attention_mask.shape[-1]} must equal text length {l_txt}")

            device = img_q.device

            ones_img = torch.ones((bs, l_img), dtype=torch.bool, device=device)

            mask_parts = [
                attention_mask.to(torch.bool),
                ones_img,
            ]
            joint_mask = torch.cat(mask_parts, dim=-1)  # (B, L_all)

            # repeat across heads and query positions
            attn_mask = joint_mask[:, None, None, :].expand(-1, self.num_heads, l_img, -1)  # (B,H,L_img,L_all)

        kv_packed = torch.cat([k, v], dim=-1)

        attn = self.attention(
            hidden_states=img_q,
            encoder_hidden_states=kv_packed,
            attention_mask=attn_mask,
        )

        return attn

    def _ffn_forward(self, x: Tensor, modulation: ModulationOut) -> Tensor:
        x = (1 + modulation.scale) * self.post_attention_layernorm(x) + modulation.shift
        return self.down_proj(self.mlp_act(self.gate_proj(x)) * self.up_proj(x))

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        spatial_conditioning: Tensor | None = None,
        attention_mask: Tensor | None = None,
        **_: dict[str, Any],
    ) -> Tensor:
        r"""
        Runs modulation-gated cross-attention and MLP, with residual connections.

        Parameters:
            img (`torch.Tensor`):
                Image tokens of shape `(B, L_img, hidden_size)`.
            txt (`torch.Tensor`):
                Text tokens of shape `(B, L_txt, hidden_size)`.
            vec (`torch.Tensor`):
                Conditioning vector used by `Modulation` to produce scale/shift/gates, shape `(B, hidden_size)` (or
                broadcastable).
            pe (`torch.Tensor`):
                Rotary positional embeddings applied inside attention.
            spatial_conditioning (`torch.Tensor`, *optional*):
                Extra conditioning tokens of shape `(B, L_cond, hidden_size)`. Used only if spatial conditioning is
                enabled in the block.
            attention_mask (`torch.Tensor`, *optional*):
                Boolean mask for text tokens of shape `(B, L_txt)`, where `0` marks padding.
            **_:
                Ignored additional keyword arguments for API compatibility.

        Returns:
            `torch.Tensor`:
                Updated image tokens of shape `(B, L_img, hidden_size)`.
        """

        mod_attn, mod_mlp = self.modulation(vec)

        img = img + mod_attn.gate * self._attn_forward(
            img,
            txt,
            pe,
            mod_attn,
            spatial_conditioning=spatial_conditioning,
            attention_mask=attention_mask,
        )
        img = img + mod_mlp.gate * self._ffn_forward(img, mod_mlp)
        return img


class LastLayer(nn.Module):
    r"""
    Final projection layer with adaptive LayerNorm modulation.

    This layer applies a normalized and modulated transformation to input tokens and projects them into patch-level
    outputs.

    Parameters:
        hidden_size (`int`):
            Dimensionality of the input tokens.
        patch_size (`int`):
            Size of the square image patches.
        out_channels (`int`):
            Number of output channels per pixel (e.g. RGB = 3).

    Forward Inputs:
        x (`torch.Tensor`):
            Input tokens of shape `(B, L, hidden_size)`, where `L` is the number of patches.
        vec (`torch.Tensor`):
            Conditioning vector of shape `(B, hidden_size)` used to generate shift and scale parameters for adaptive
            LayerNorm.

    Returns:
        `torch.Tensor`:
            Projected patch outputs of shape `(B, L, patch_size * patch_size * out_channels)`.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


def img2seq(img: Tensor, patch_size: int) -> Tensor:
    r"""
    Flattens an image tensor into a sequence of non-overlapping patches.

    Parameters:
        img (`torch.Tensor`):
            Input image tensor of shape `(B, C, H, W)`.
        patch_size (`int`):
            Size of each square patch. Must evenly divide both `H` and `W`.

    Returns:
        `torch.Tensor`:
            Flattened patch sequence of shape `(B, L, C * patch_size * patch_size)`, where `L = (H // patch_size) * (W
            // patch_size)` is the number of patches.
    """
    return unfold(img, kernel_size=patch_size, stride=patch_size).transpose(1, 2)


def seq2img(seq: Tensor, patch_size: int, shape: Tensor) -> Tensor:
    r"""
    Reconstructs an image tensor from a sequence of patches (inverse of `img2seq`).

    Parameters:
        seq (`torch.Tensor`):
            Patch sequence of shape `(B, L, C * patch_size * patch_size)`, where `L = (H // patch_size) * (W //
            patch_size)`.
        patch_size (`int`):
            Size of each square patch.
        shape (`tuple` or `torch.Tensor`):
            The original image spatial shape `(H, W)`. If a tensor is provided, the first two values are interpreted as
            height and width.

    Returns:
        `torch.Tensor`:
            Reconstructed image tensor of shape `(B, C, H, W)`.
    """
    if isinstance(shape, tuple):
        shape = shape[-2:]
    elif isinstance(shape, torch.Tensor):
        shape = (int(shape[0]), int(shape[1]))
    else:
        raise NotImplementedError(f"shape type {type(shape)} not supported")
    return fold(seq.transpose(1, 2), shape, kernel_size=patch_size, stride=patch_size)


class PhotonTransformer2DModel(ModelMixin, ConfigMixin):
    r"""
    Transformer-based 2D model for text to image generation. It supports attention processor injection and LoRA
    scaling.

    Parameters:
        in_channels (`int`, *optional*, defaults to 16):
            Number of input channels in the latent image.
        patch_size (`int`, *optional*, defaults to 2):
            Size of the square patches used to flatten the input image.
        context_in_dim (`int`, *optional*, defaults to 2304):
            Dimensionality of the text conditioning input.
        hidden_size (`int`, *optional*, defaults to 1792):
            Dimension of the hidden representation.
        mlp_ratio (`float`, *optional*, defaults to 3.5):
            Expansion ratio for the hidden dimension inside MLP blocks.
        num_heads (`int`, *optional*, defaults to 28):
            Number of attention heads.
        depth (`int`, *optional*, defaults to 16):
            Number of transformer blocks.
        axes_dim (`list[int]`, *optional*):
            List of dimensions for each positional embedding axis. Defaults to `[32, 32]`.
        theta (`int`, *optional*, defaults to 10000):
            Frequency scaling factor for rotary embeddings.
        time_factor (`float`, *optional*, defaults to 1000.0):
            Scaling factor applied in timestep embeddings.
        time_max_period (`int`, *optional*, defaults to 10000):
            Maximum frequency period for timestep embeddings.

    Attributes:
        pe_embedder (`EmbedND`):
            Multi-axis rotary embedding generator for positional encodings.
        img_in (`nn.Linear`):
            Projection layer for image patch tokens.
        time_in (`MLPEmbedder`):
            Embedding layer for timestep embeddings.
        txt_in (`nn.Linear`):
            Projection layer for text conditioning.
        blocks (`nn.ModuleList`):
            Stack of transformer blocks (`PhotonBlock`).
        final_layer (`LastLayer`):
            Projection layer mapping hidden tokens back to patch outputs.

    Methods:
        attn_processors:
            Returns a dictionary of all attention processors in the model.
        set_attn_processor(processor):
            Replaces attention processors across all attention layers.
        process_inputs(image_latent, txt):
            Converts inputs into patch tokens, encodes text, and produces positional encodings.
        compute_timestep_embedding(timestep, dtype):
            Creates a timestep embedding of dimension 256, scaled and projected.
        forward_transformers(image_latent, cross_attn_conditioning, timestep, time_embedding, attention_mask,
        **block_kwargs):
            Runs the sequence of transformer blocks over image and text tokens.
        forward(image_latent, timestep, cross_attn_conditioning, micro_conditioning, cross_attn_mask=None,
        attention_kwargs=None, return_dict=True):
            Full forward pass from latent input to reconstructed output image.

    Returns:
        `Transformer2DModelOutput` if `return_dict=True` (default), otherwise a tuple containing:
            - `sample` (`torch.Tensor`): Reconstructed image of shape `(B, C, H, W)`.
    """

    config_name = "config.json"
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        patch_size: int = 2,
        context_in_dim: int = 2304,
        hidden_size: int = 1792,
        mlp_ratio: float = 3.5,
        num_heads: int = 28,
        depth: int = 16,
        axes_dim: list = None,
        theta: int = 10000,
        time_factor: float = 1000.0,
        time_max_period: int = 10000,
    ):
        super().__init__()

        if axes_dim is None:
            axes_dim = [32, 32]

        # Store parameters directly
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.out_channels = self.in_channels * self.patch_size**2

        self.time_factor = time_factor
        self.time_max_period = time_max_period

        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")

        pe_dim = hidden_size // num_heads

        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.img_in = nn.Linear(self.in_channels * self.patch_size**2, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.txt_in = nn.Linear(context_in_dim, self.hidden_size)

        self.blocks = nn.ModuleList(
            [
                PhotonBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(depth)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def _process_inputs(self, image_latent: Tensor, txt: Tensor, **_: Any) -> tuple[Tensor, Tensor, Tensor]:
        txt = self.txt_in(txt)
        img = img2seq(image_latent, self.patch_size)
        bs, _, h, w = image_latent.shape
        img_ids = get_image_ids(bs, h, w, patch_size=self.patch_size, device=image_latent.device)
        pe = self.pe_embedder(img_ids)
        return img, txt, pe

    def _compute_timestep_embedding(self, timestep: Tensor, dtype: torch.dtype) -> Tensor:
        return self.time_in(
            get_timestep_embedding(
                timesteps=timestep,
                embedding_dim=256,
                max_period=self.time_max_period,
                scale=self.time_factor,
                flip_sin_to_cos=True,  # Match original cos, sin order
            ).to(dtype)
        )

    def _forward_transformers(
        self,
        image_latent: Tensor,
        cross_attn_conditioning: Tensor,
        timestep: Optional[Tensor] = None,
        time_embedding: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        **block_kwargs: Any,
    ) -> Tensor:
        img = self.img_in(image_latent)

        if time_embedding is not None:
            vec = time_embedding
        else:
            if timestep is None:
                raise ValueError("Please provide either a timestep or a timestep_embedding")
            vec = self._compute_timestep_embedding(timestep, dtype=img.dtype)

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                img = self._gradient_checkpointing_func(
                    block.__call__,
                    img,
                    cross_attn_conditioning,
                    vec,
                    block_kwargs.get("pe"),
                    block_kwargs.get("spatial_conditioning"),
                    attention_mask,
                )
            else:
                img = block(
                    img=img, txt=cross_attn_conditioning, vec=vec, attention_mask=attention_mask, **block_kwargs
                )

        img = self.final_layer(img, vec)
        return img

    def forward(
        self,
        image_latent: Tensor,
        timestep: Tensor,
        cross_attn_conditioning: Tensor,
        micro_conditioning: Tensor,
        cross_attn_mask: None | Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, ...], Transformer2DModelOutput]:
        r"""
        Forward pass of the PhotonTransformer2DModel.

        The latent image is split into patch tokens, combined with text conditioning, and processed through a stack of
        transformer blocks modulated by the timestep. The output is reconstructed into the latent image space.

        Parameters:
            image_latent (`torch.Tensor`):
                Input latent image tensor of shape `(B, C, H, W)`.
            timestep (`torch.Tensor`):
                Timestep tensor of shape `(B,)` or `(1,)`, used for temporal conditioning.
            cross_attn_conditioning (`torch.Tensor`):
                Text conditioning tensor of shape `(B, L_txt, context_in_dim)`.
            micro_conditioning (`torch.Tensor`):
                Extra conditioning vector (currently unused, reserved for future use).
            cross_attn_mask (`torch.Tensor`, *optional*):
                Boolean mask of shape `(B, L_txt)`, where `0` marks padding in the text sequence.
            attention_kwargs (`dict`, *optional*):
                Additional arguments passed to attention layers. If using the PEFT backend, the key `"scale"` controls
                LoRA scaling (default: 1.0).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a `Transformer2DModelOutput` or a tuple.

        Returns:
            `Transformer2DModelOutput` if `return_dict=True`, otherwise a tuple:

                - `sample` (`torch.Tensor`): Output latent image of shape `(B, C, H, W)`.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )
        img_seq, txt, pe = self._process_inputs(image_latent, cross_attn_conditioning)
        img_seq = self._forward_transformers(img_seq, txt, timestep, pe=pe, attention_mask=cross_attn_mask)
        output = seq2img(img_seq, self.patch_size, image_latent.shape)
        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
