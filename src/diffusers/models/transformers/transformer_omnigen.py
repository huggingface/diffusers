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

from typing import Any, Dict, List, Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers
from ..attention import OmniGenFeedForward
from ..attention_processor import Attention, AttentionProcessor, OmniGenAttnProcessor2_0
from ..embeddings import OmniGenPatchEmbed, OmniGenSuScaledRotaryEmbedding, TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNorm, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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
        rotary_emb: torch.Tensor,
    ):
        """
        Perform a forward pass through the LuminaNextDiTBlock.

        Parameters:
            hidden_states (`torch.Tensor`): The input of hidden_states for LuminaNextDiTBlock.
            attention_mask (`torch.Tensor): The input of hidden_states corresponse attention mask.
            rotary_emb (`torch.Tensor`): Precomputed cosine and sine frequencies.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=hidden_states,
            attention_mask=attention_mask,
            query_rotary_emb=rotary_emb,
            key_rotary_emb=rotary_emb,
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

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

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
        timestep: Union[int, float, torch.LongTensor],
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
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            input_ids (`torch.LongTensor`):
                token ids
            input_img_latents (`torch.FloatTensor`):
                encoded image latents by VAE
            input_image_sizes (`dict`):
                the indices of the input_img_latents in the input_ids
            attention_mask (`torch.FloatTensor`):
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

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, attention_mask=attention_mask, rotary_emb=[cos, sin])

        hidden_states = self.norm(hidden_states)

        image_embedding = hidden_states[:, -num_tokens_for_output_image:]
        time_emb = self.t_embedder(self.time_proj(timestep).to(hidden_states.dtype))
        x = self.proj_out(self.norm_out(image_embedding, temb=time_emb))
        output = self.unpatchify(x, height, width)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
