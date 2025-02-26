# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.loaders import FromOriginalModelMixin

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ..attention import FeedForward
from ..attention_processor import Attention, AttentionProcessor
from ..cache_utils import CacheMixin
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        grid_sizes: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # i2v task
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]


        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
    
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
    
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        if grid_sizes is not None and freqs is not None:
            query = rope_apply(query, grid_sizes, freqs)
            key = rope_apply(key, grid_sizes, freqs)
        
        query = query.transpose(1, 2)

        # i2v task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img).view(batch_size, -1, attn.heads, head_dim)
            value_img = attn.add_v_proj(encoder_hidden_states_img).view(batch_size, -1, attn.heads, head_dim)
            key_img = key_img.transpose(1, 2)
            value_img = value_img.transpose(1, 2)
            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)


        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.cuda.amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.cuda.amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class WanBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 added_kv_proj_dim=None):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        # self attn
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm="rms_norm_across_heads" if qk_norm else None,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanAttnProcessor2_0(),
        )

        # cross attn
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm="rms_norm_across_heads" if qk_norm else None,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=WanAttnProcessor2_0(),
        )

        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states,
        e,
        encoder_hidden_states,
        seq_lens,
        grid_sizes,
        freqs,
        context_lens,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        assert e.dtype == torch.float32
        with torch.cuda.amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        attn_hidden_states = self.norm1(hidden_states).float() * (1 + e[1]) + e[0]

        attn_hidden_states = self.attn1(
            hidden_states=attn_hidden_states,
            grid_sizes=grid_sizes,
            freqs=freqs,
        )
        with torch.cuda.amp.autocast(dtype=torch.float32):
            hidden_states = hidden_states + attn_hidden_states * e[2]

        # cross-attention
        attn_hidden_states = self.norm3(hidden_states)
        attn_hidden_states = self.attn2(
            hidden_states=attn_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            grid_sizes=None,
            freqs=None,
        )
        hidden_states = hidden_states + attn_hidden_states

        # ffn
        ffn_hidden_states = self.norm2(hidden_states).float() * (1 + e[4]) + e[3]
        ffn_hidden_states = self.ffn(ffn_hidden_states)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            hidden_states = hidden_states + ffn_hidden_states * e[5]
        
        return hidden_states


class WanHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        assert e.dtype == torch.float32
        with torch.cuda.amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "text_embedding", "time_embedding", "time_projection"]
    _no_split_modules = [
        "WanBlock",
        "WanHead",
    ]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 512,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        window_size: Tuple[int] = (-1, -1),
        cross_attn_norm: bool = True,
        qk_norm: bool = True,
        eps: float = 1e-6,
        add_img_emb: bool = False,
        added_kv_proj_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.freq_dim = freq_dim
        self.out_channels = out_channels
        self.patch_size = patch_size

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, inner_dim), nn.GELU(approximate='tanh'),
            nn.Linear(inner_dim, inner_dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, inner_dim), nn.SiLU(), nn.Linear(inner_dim, inner_dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(inner_dim, inner_dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanBlock(inner_dim, ffn_dim, num_attention_heads,
                      window_size, qk_norm, cross_attn_norm, eps,
                      added_kv_proj_dim)
            for _ in range(num_layers)
        ])

        # head
        self.head = WanHead(inner_dim, out_channels, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert attention_head_dim % 2 == 0
        self.freqs = torch.cat([
            rope_params(1024, attention_head_dim - 4 * (attention_head_dim // 6)),
            rope_params(1024, 2 * (attention_head_dim // 6)),
            rope_params(1024, 2 * (attention_head_dim // 6))
        ],
        dim=1)

        self.add_img_emb = add_img_emb
        if add_img_emb:
            self.img_emb = MLPProj(1280, inner_dim)

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

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        seq_len: int,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
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

        if self.freqs.device != hidden_states.device:
            self.freqs = self.freqs.to(hidden_states.device)
        
        hidden_states = self.patch_embedding(hidden_states)

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[1:], dtype=torch.long) for u in hidden_states]
            )
        hidden_states = hidden_states.flatten(2).transpose(1, 2) # (b, l, c)
        seq_lens = torch.tensor([u.size(0) for u in hidden_states], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        hidden_states = torch.cat([
            torch.cat([u.unsqueeze(0), u.new_zeros(1, seq_len - u.size(0), u.size(1))],
                    dim=1) for u in hidden_states
        ])

        with torch.cuda.amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, timestep).float())
            e0 = self.time_projection(e).unflatten(1, (6, -1))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
        
        context_lens = None
        encoder_hidden_states = self.text_embedding(encoder_hidden_states)
        if self.add_img_emb:
            img_encoder_hidden_states = kwargs.get('img_encoder_hidden_states', None)
            if img_encoder_hidden_states is None:
                raise ValueError('`add_img_emb` is set but `img_encoder_hidden_states` is not provided.')
            img_encoder_hidden_states = self.img_emb(img_encoder_hidden_states)
            encoder_hidden_states = torch.concat([img_encoder_hidden_states, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    e0,
                    encoder_hidden_states,
                    seq_lens,
                    grid_sizes,
                    self.freqs,
                    context_lens,
                    attention_kwargs,
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    e0,
                    encoder_hidden_states,
                    seq_lens,
                    grid_sizes,
                    self.freqs,
                    context_lens,
                    attention_kwargs,
                )

        # Output projection
        hidden_states = self.head(hidden_states, e)

        # 5. Unpatchify
        hidden_states = self.unpatchify(hidden_states, grid_sizes)
        hidden_states = torch.stack(hidden_states)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)


    def unpatchify(self, hidden_states, grid_sizes):
        c = self.out_channels
        out = []
        for u, v in zip(hidden_states, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

