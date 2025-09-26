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
from typing import Any, Dict, Optional, Union, Tuple
import torch
import math
from torch import Tensor, nn
from torch.nn.functional import fold, unfold
from einops import rearrange
from einops.layers.torch import Rearrange

from ...configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin
from ..modeling_outputs import Transformer2DModelOutput
from ..attention_processor import Attention, AttentionProcessor, MirageAttnProcessor2_0
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ..normalization import RMSNorm


logger = logging.get_logger(__name__)


def get_image_ids(bs: int, h: int, w: int, patch_size: int, device: torch.device) -> Tensor:
    img_ids = torch.zeros(h // patch_size, w // patch_size, 2, device=device)
    img_ids[..., 0] = torch.arange(h // patch_size, device=device)[:, None]
    img_ids[..., 1] = torch.arange(w // patch_size, device=device)[None, :]
    return img_ids.reshape((h // patch_size) * (w // patch_size), 2).unsqueeze(0).repeat(bs, 1, 1)


def apply_rope(xq: Tensor, freqs_cis: Tensor) -> Tensor:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq)


class EmbedND(nn.Module):
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


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> Tensor:
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))




class QKNorm(torch.nn.Module):
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
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 6 * dim, bias=True)
        nn.init.constant_(self.lin.weight, 0)
        nn.init.constant_(self.lin.bias, 0)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(6, dim=-1)
        return ModulationOut(*out[:3]), ModulationOut(*out[3:])


class MirageBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()

        self._fsdp_wrap = True
        self._activation_checkpointing = True

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
            processor=MirageAttnProcessor2_0(),
        )


        # mlp
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.gate_proj = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.mlp_hidden_dim, hidden_size, bias=False)
        self.mlp_act = nn.GELU(approximate="tanh")

        self.modulation = Modulation(hidden_size)
        self.spatial_cond_kv_proj: None | nn.Linear = None

    def attn_forward(
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

        # optional spatial conditioning tokens
        cond_len = 0
        if self.spatial_cond_kv_proj is not None:
            assert spatial_conditioning is not None
            cond_kv = self.spatial_cond_kv_proj(spatial_conditioning)
            cond_k, cond_v = rearrange(cond_kv, "B L (K H D) -> K B H L D", K=2, H=self.num_heads)
            cond_k = apply_rope(cond_k, pe)
            cond_len = cond_k.shape[2]
            k = torch.cat((cond_k, k), dim=2)
            v = torch.cat((cond_v, v), dim=2)

        # build multiplicative 0/1 mask for provided attention_mask over [cond?, text, image] keys
        attn_mask: Tensor | None = None
        if attention_mask is not None:
            bs, _, l_img, _ = img_q.shape
            l_txt = txt_k.shape[2]

            assert attention_mask.dim() == 2, f"Unsupported attention_mask shape: {attention_mask.shape}"
            assert (
                attention_mask.shape[-1] == l_txt
            ), f"attention_mask last dim {attention_mask.shape[-1]} must equal text length {l_txt}"

            device = img_q.device

            ones_img = torch.ones((bs, l_img), dtype=torch.bool, device=device)
            cond_mask = torch.ones((bs, cond_len), dtype=torch.bool, device=device)

            mask_parts = [
                cond_mask,
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

    def ffn_forward(self, x: Tensor, modulation: ModulationOut) -> Tensor:
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
        mod_attn, mod_mlp = self.modulation(vec)

        img = img + mod_attn.gate * self.attn_forward(
            img,
            txt,
            pe,
            mod_attn,
            spatial_conditioning=spatial_conditioning,
            attention_mask=attention_mask,
        )
        img = img + mod_mlp.gate * self.ffn_forward(img, mod_mlp)
        return img


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


@dataclass
class MirageParams:
    in_channels: int
    patch_size: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    axes_dim: list[int]
    theta: int
    time_factor: float = 1000.0
    time_max_period: int = 10_000
    conditioning_block_ids: list[int] | None = None


def img2seq(img: Tensor, patch_size: int) -> Tensor:
    """Flatten an image into a sequence of patches"""
    return unfold(img, kernel_size=patch_size, stride=patch_size).transpose(1, 2)


def seq2img(seq: Tensor, patch_size: int, shape: Tensor) -> Tensor:
    """Revert img2seq"""
    if isinstance(shape, tuple):
        shape = shape[-2:]
    elif isinstance(shape, torch.Tensor):
        shape = (int(shape[0]), int(shape[1]))
    else:
        raise NotImplementedError(f"shape type {type(shape)} not supported")
    return fold(seq.transpose(1, 2), shape, kernel_size=patch_size, stride=patch_size)


class MirageTransformer2DModel(ModelMixin, ConfigMixin):
    """Mirage Transformer model with IP-Adapter support."""

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
        conditioning_block_ids: list = None,
        **kwargs
    ):
        super().__init__()

        if axes_dim is None:
            axes_dim = [32, 32]

        # Create MirageParams from the provided arguments
        params = MirageParams(
            in_channels=in_channels,
            patch_size=patch_size,
            context_in_dim=context_in_dim,
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            depth=depth,
            axes_dim=axes_dim,
            theta=theta,
            time_factor=time_factor,
            time_max_period=time_max_period,
            conditioning_block_ids=conditioning_block_ids,
        )

        self.params = params
        self.in_channels = params.in_channels
        self.patch_size = params.patch_size
        self.out_channels = self.in_channels * self.patch_size**2

        self.time_factor = params.time_factor
        self.time_max_period = params.time_max_period

        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")

        pe_dim = params.hidden_size // params.num_heads

        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")

        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels * self.patch_size**2, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        conditioning_block_ids: list[int] = params.conditioning_block_ids or list(range(params.depth))

        self.blocks = nn.ModuleList(
            [
                MirageBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                )
                for i in range(params.depth)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

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

    def process_inputs(self, image_latent: Tensor, txt: Tensor, **_: Any) -> tuple[Tensor, Tensor, Tensor]:
        """Timestep independent stuff"""
        txt = self.txt_in(txt)
        img = img2seq(image_latent, self.patch_size)
        bs, _, h, w = image_latent.shape
        img_ids = get_image_ids(bs, h, w, patch_size=self.patch_size, device=image_latent.device)
        pe = self.pe_embedder(img_ids)
        return img, txt, pe

    def compute_timestep_embedding(self, timestep: Tensor, dtype: torch.dtype) -> Tensor:
        return self.time_in(
            timestep_embedding(
                t=timestep, dim=256, max_period=self.time_max_period, time_factor=self.time_factor
            ).to(dtype)
        )

    def forward_transformers(
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
            vec = self.compute_timestep_embedding(timestep, dtype=img.dtype)

        for block in self.blocks:
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
        img_seq, txt, pe = self.process_inputs(image_latent, cross_attn_conditioning)
        img_seq = self.forward_transformers(img_seq, txt, timestep, pe=pe, attention_mask=cross_attn_mask)
        output = seq2img(img_seq, self.patch_size, image_latent.shape)
        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
