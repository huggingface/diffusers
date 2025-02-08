# Copyright 2024 OmniGen team and The HuggingFace Team. All rights reserved.
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
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers
from ..attention_processor import Attention, AttentionProcessor
from ..embeddings import TimestepEmbedding, Timesteps, get_2d_sincos_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNorm, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class OmniGenFeedForward(nn.Module):
    r"""
    A feed-forward layer for OmniGen.

    Parameters:
        hidden_size (`int`):
            The dimensionality of the hidden layers in the model. This parameter determines the width of the model's
            hidden representations.
        intermediate_size (`int`): The intermediate dimension of the feedforward layer.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.activation_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)


class OmniGenPatchEmbed(nn.Module):
    """2D Image to Patch Embedding with support for OmniGen."""

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        embed_dim: int = 768,
        bias: bool = True,
        interpolation_scale: float = 1,
        pos_embed_max_size: int = 192,
        base_size: int = 64,
    ):
        super().__init__()

        self.output_image_proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        self.input_image_proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )

        self.patch_size = patch_size
        self.interpolation_scale = interpolation_scale
        self.pos_embed_max_size = pos_embed_max_size

        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, self.pos_embed_max_size, base_size=base_size, interpolation_scale=self.interpolation_scale, output_type="pt"
        )
        self.register_buffer("pos_embed", pos_embed.float().unsqueeze(0), persistent=True)

    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed

    def patch_embeddings(self, latent, is_input_image: bool):
        if is_input_image:
            latent = self.input_image_proj(latent)
        else:
            latent = self.output_image_proj(latent)
        latent = latent.flatten(2).transpose(1, 2)
        return latent

    def forward(self, latent: torch.Tensor, is_input_image: bool, padding_latent: torch.Tensor = None):
        """
        Args:
            latent: encoded image latents
            is_input_image: use input_image_proj or output_image_proj
            padding_latent:
                When sizes of target images are inconsistent, use `padding_latent` to maintain consistent sequence
                length.

        Returns: torch.Tensor

        """
        if isinstance(latent, list):
            if padding_latent is None:
                padding_latent = [None] * len(latent)
            patched_latents = []
            for sub_latent, padding in zip(latent, padding_latent):
                height, width = sub_latent.shape[-2:]
                sub_latent = self.patch_embeddings(sub_latent, is_input_image)
                pos_embed = self.cropped_pos_embed(height, width)
                sub_latent = sub_latent + pos_embed
                if padding is not None:
                    sub_latent = torch.cat([sub_latent, padding.to(sub_latent.device)], dim=-2)
                patched_latents.append(sub_latent)
        else:
            height, width = latent.shape[-2:]
            pos_embed = self.cropped_pos_embed(height, width)
            latent = self.patch_embeddings(latent, is_input_image)
            patched_latents = latent + pos_embed

        return patched_latents


class OmniGenSuScaledRotaryEmbedding(nn.Module):
    def __init__(
        self, dim, max_position_embeddings=131072, original_max_position_embeddings=4096, base=10000, rope_scaling=None
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

        self.short_factor = rope_scaling["short_factor"]
        self.long_factor = rope_scaling["long_factor"]
        self.original_max_position_embeddings = original_max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=x.device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=x.device)

        inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)

            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    
    cos, sin = freqs_cis  # [S, D]
    if len(cos.shape) == 2:
        cos = cos[None, None]
        sin = sin[None, None]
    elif len(cos.shape) == 3:
        cos = cos[:, None]
        sin = sin[:, None]
    cos, sin = cos.to(x.device), sin.to(x.device)

    # Rotates half the hidden dims of the input. this rorate function is widely used in LLM, e.g. Llama, Phi3, etc.
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    x_rotated = torch.cat((-x2, x1), dim=-1)
       
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out
    

class OmniGenAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the OmniGen model.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        bsz, q_len, query_dim = query.size()
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query, key = query.to(dtype), key.to(dtype)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        hidden_states = hidden_states.transpose(1, 2).to(dtype)
        hidden_states = hidden_states.reshape(bsz, q_len, attn.out_dim)
        hidden_states = attn.to_out[0](hidden_states)
        return hidden_states


class OmniGenBlock(nn.Module):
    """
    A LuminaNextDiTBlock for LuminaNextDiT2DModel.

    Parameters:
        hidden_size (`int`): Embedding dimension of the input features.
        num_attention_heads (`int`): Number of attention heads.
        num_key_value_heads (`int`):
            Number of attention heads in key and value features (if using GQA), or set to None for the same as query.
        intermediate_size (`int`): size of intermediate layer.
        rms_norm_eps (`float`): The eps for norm layer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=hidden_size,
            dim_head=hidden_size // num_attention_heads,
            heads=num_attention_heads,
            kv_heads=num_key_value_heads,
            bias=False,
            out_dim=hidden_size,
            out_bias=False,
            processor=OmniGenAttnProcessor2_0(),
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = OmniGenFeedForward(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
    ):
        """
        Perform a forward pass through the LuminaNextDiTBlock.

        Parameters:
            hidden_states (`torch.Tensor`): The input of hidden_states for LuminaNextDiTBlock.
            attention_mask (`torch.Tensor): The input of hidden_states corresponse attention mask.
            image_rotary_emb (`torch.Tensor`): Precomputed cosine and sine frequencies.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = residual + attn_outputs

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class OmniGenTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    The Transformer model introduced in OmniGen.

    Reference: https://arxiv.org/pdf/2409.11340

    Parameters:
        hidden_size (`int`, *optional*, defaults to 3072):
            The dimensionality of the hidden layers in the model. This parameter determines the width of the model's
            hidden representations.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5): eps for RMSNorm layer.
        num_attention_heads (`int`, *optional*, defaults to 32):
            The number of attention heads in each attention layer. This parameter specifies how many separate attention
            mechanisms are used.
        num_kv_heads (`int`, *optional*, defaults to 32):
            The number of key-value heads in the attention mechanism, if different from the number of attention heads.
            If None, it defaults to num_attention_heads.
        intermediate_size (`int`, *optional*, defaults to 8192): dimension of the intermediate layer in FFN
        num_layers (`int`, *optional*, default to 32):
            The number of layers in the model. This defines the depth of the neural network.
        pad_token_id (`int`, *optional*, default to 32000):
            id for pad token
        vocab_size (`int`, *optional*, default to 32064):
            size of vocabulary
        patch_size (`int`, defaults to 2): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input.
        pos_embed_max_size (`int`, *optional*, defaults to 192): The max size of pos emb.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["OmniGenBlock"]

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 3072,
        rms_norm_eps: float = 1e-05,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        intermediate_size: int = 8192,
        num_layers: int = 32,
        pad_token_id: int = 32000,
        vocab_size: int = 32064,
        max_position_embeddings: int = 131072,
        original_max_position_embeddings: int = 4096,
        rope_base: int = 10000,
        rope_scaling: Dict = None,
        patch_size=2,
        in_channels=4,
        pos_embed_max_size: int = 192,
        time_step_dim: int = 256,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: int = 0,
        timestep_activation_fn: str = "silu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size

        self.patch_embedding = OmniGenPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
            pos_embed_max_size=pos_embed_max_size,
        )

        self.time_proj = Timesteps(time_step_dim, flip_sin_to_cos, downscale_freq_shift)
        self.time_token = TimestepEmbedding(time_step_dim, hidden_size, timestep_activation_fn)
        self.t_embedder = TimestepEmbedding(time_step_dim, hidden_size, timestep_activation_fn)

        self.norm_out = AdaLayerNorm(hidden_size, norm_elementwise_affine=False, norm_eps=1e-6, chunk_dim=1)
        self.proj_out = nn.Linear(hidden_size, patch_size * patch_size * self.out_channels, bias=True)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, pad_token_id)
        self.rotary_emb = OmniGenSuScaledRotaryEmbedding(
            hidden_size // num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            original_max_position_embeddings=original_max_position_embeddings,
            base=rope_base,
            rope_scaling=rope_scaling,
        )

        self.layers = nn.ModuleList(
            [
                OmniGenBlock(
                    hidden_size,
                    num_attention_heads,
                    num_key_value_heads,
                    intermediate_size,
                    rms_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.gradient_checkpointing = False

    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C) imgs: (N, H, W, C)
        """
        c = self.out_channels

        x = x.reshape(
            shape=(x.shape[0], h // self.patch_size, w // self.patch_size, self.patch_size, self.patch_size, c)
        )
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h, w))
        return imgs

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
    def set_attn_processor(self, processor: Union[OmniGenAttnProcessor2_0, Dict[str, AttentionProcessor]]):
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

    def get_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        input_img_latents: List[torch.Tensor],
        input_image_sizes: Dict,
    ):
        """
        get the multi-modal conditional embeddings

        Args:
            input_ids: a sequence of text id
            input_img_latents: continues embedding of input images
            input_image_sizes: the index of the input image in the input_ids sequence.

        Returns: torch.Tensor

        """
        input_img_latents = [x.to(self.dtype) for x in input_img_latents]
        condition_tokens = None
        if input_ids is not None:
            condition_tokens = self.embed_tokens(input_ids)
            input_img_inx = 0
            if input_img_latents is not None:
                input_image_tokens = self.patch_embedding(input_img_latents, is_input_image=True)

                for b_inx in input_image_sizes.keys():
                    for start_inx, end_inx in input_image_sizes[b_inx]:
                        # replace the placeholder in text tokens with the image embedding.
                        condition_tokens[b_inx, start_inx:end_inx] = input_image_tokens[input_img_inx].to(
                            condition_tokens.dtype
                        )
                        input_img_inx += 1

        return condition_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.FloatTensor],
        input_ids: torch.Tensor,
        input_img_latents: List[torch.Tensor],
        input_image_sizes: Dict[int, List[int]],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        """
        The [`OmniGenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            timestep (`torch.FloatTensor`):
                Used to indicate denoising step.
            input_ids (`torch.LongTensor`):
                token ids
            input_img_latents (`torch.Tensor`):
                encoded image latents by VAE
            input_image_sizes (`dict`):
                the indices of the input_img_latents in the input_ids
            attention_mask (`torch.Tensor`):
                mask for self-attention
            position_ids (`torch.LongTensor`):
                id to represent position
            past_key_values (`transformers.cache_utils.Cache`):
                previous key and value states
            offload_transformer_block (`bool`, *optional*, defaults to `True`):
                offload transformer block to cpu
            attention_kwargs: (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`OmniGen2DModelOutput`] instead of a plain tuple.

        Returns:
            If `return_dict` is True, an [`OmniGen2DModelOutput`] is returned, otherwise a `tuple` where the first
            element is the sample tensor.

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
        height, width = hidden_states.size()[-2:]
        hidden_states = self.patch_embedding(hidden_states, is_input_image=False)
        num_tokens_for_output_image = hidden_states.size(1)

        time_token = self.time_token(self.time_proj(timestep).to(hidden_states.dtype)).unsqueeze(1)

        condition_tokens = self.get_multimodal_embeddings(
            input_ids=input_ids,
            input_img_latents=input_img_latents,
            input_image_sizes=input_image_sizes,
        )
        if condition_tokens is not None:
            inputs_embeds = torch.cat([condition_tokens, time_token, hidden_states], dim=1)
        else:
            inputs_embeds = torch.cat([time_token, hidden_states], dim=1)

        batch_size, seq_length = inputs_embeds.shape[:2]
        position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is not None and attention_mask.dim() == 3:
            dtype = inputs_embeds.dtype
            min_dtype = torch.finfo(dtype).min
            attention_mask = (1 - attention_mask) * min_dtype
            attention_mask = attention_mask.unsqueeze(1).to(inputs_embeds.dtype)
        else:
            raise Exception("attention_mask parameter was unavailable or invalid")

        hidden_states = inputs_embeds

        image_rotary_emb = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(decoder_layer, hidden_states, attention_mask, image_rotary_emb)
            else:
                hidden_states = decoder_layer(hidden_states, attention_mask=attention_mask, image_rotary_emb=image_rotary_emb)

        hidden_states = self.norm(hidden_states)

        hidden_states = hidden_states[:, -num_tokens_for_output_image:]
        timestep_proj = self.time_proj(timestep)
        temb = self.t_embedder(timestep_proj.type_as(hidden_states))
        hidden_states = self.norm_out(hidden_states, temb=temb)
        hidden_states = self.proj_out(hidden_states)
        output = self.unpatchify(hidden_states, height, width)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
