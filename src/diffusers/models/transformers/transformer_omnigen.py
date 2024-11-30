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

from typing import Dict, Optional, Tuple, Union, List

import torch
from torch import nn
import torch.utils.checkpoint

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import Phi3Model, Phi3Config
from transformers.cache_utils import Cache, DynamicCache

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import logging
from ..attention_processor import AttentionProcessor
from ..normalization import AdaLayerNorm, CogVideoXLayerNormZero
from ..embeddings import OmniGenPatchEmbed, OmniGenTimestepEmbed
from ..modeling_utils import ModelMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



class OmniGenBaseTransformer(Phi3Model):
    """
    Transformer used in OmniGen. The transformer block is from Ph3, and only modify the attention mask.
    References: [OmniGen](https://arxiv.org/pdf/2409.11340)

    Parameters:
        config: Phi3Config
    """

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            offload_model: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if attention_mask is not None and attention_mask.dim() == 3:
            dtype = inputs_embeds.dtype
            min_dtype = torch.finfo(dtype).min
            attention_mask = (1 - attention_mask) * min_dtype
            attention_mask = attention_mask.unsqueeze(1).to(inputs_embeds.dtype)
        else:
            raise Exception("attention_mask parameter was unavailable or invalid")

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        layer_idx = -1
        for decoder_layer in self.layers:
            layer_idx += 1

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class OmniGenTransformer(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    The Transformer model introduced in OmniGen.

    Reference: https://arxiv.org/pdf/2409.11340

    Parameters:
        patch_size (`int`, defaults to 2): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            transformer_config: Phi3Config,
            patch_size=2,
            in_channels=4,
            pos_embed_max_size: int = 192,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size

        hidden_size = transformer_config.hidden_size

        self.patch_embedding = OmniGenPatchEmbed(patch_size=patch_size, in_channels=in_channels, embed_dim=hidden_size, pos_embed_max_size=pos_embed_max_size)

        self.time_token = OmniGenTimestepEmbed(hidden_size)
        self.t_embedder = OmniGenTimestepEmbed(hidden_size)

        self.norm_out = AdaLayerNorm(hidden_size, norm_elementwise_affine=False, norm_eps=1e-6, chunk_dim=1)
        self.proj_out = nn.Linear(hidden_size, patch_size * patch_size * self.out_channels, bias=True)

        self.llm = OmniGenBaseTransformer(config=transformer_config)
        self.llm.config.use_cache = False


    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels

        x = x.reshape(
            shape=(x.shape[0], h // self.patch_size, w // self.patch_size, self.patch_size, self.patch_size, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h, w))
        return imgs


    def prepare_condition_embeddings(self, input_ids, input_img_latents, input_image_sizes, padding_latent):
        condition_embeds = None
        if input_img_latents is not None:
            input_latents = self.patch_embedding(input_img_latents, is_input_images=True, padding_latent=padding_latent)
        if input_ids is not None:
            condition_embeds = self.llm.embed_tokens(input_ids).clone()
            input_img_inx = 0
            for b_inx in input_image_sizes.keys():
                for start_inx, end_inx in input_image_sizes[b_inx]:
                    condition_embeds[b_inx, start_inx: end_inx] = input_latents[input_img_inx]
                    input_img_inx += 1
            if input_img_latents is not None:
                assert input_img_inx == len(input_latents)
        return condition_embeds

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

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(self,
                hidden_states,
                timestep,
                input_ids,
                input_img_latents,
                input_image_sizes,
                attention_mask,
                position_ids,
                padding_latent=None,
                past_key_values=None,
                return_past_key_values=True,
                offload_model: bool = False):

        height, width =  hidden_states.size(-2)
        hidden_states = self.patch_embedding(hidden_states, is_input_image=False)
        num_tokens_for_output_image = hidden_states.size(1)

        time_token = self.time_token(timestep, dtype=hidden_states.dtype).unsqueeze(1)

        condition_embeds = self.prepare_condition_embeddings(input_ids, input_img_latents, input_image_sizes, padding_latent)
        if condition_embeds is not None:
            input_emb = torch.cat([condition_embeds, time_token, hidden_states], dim=1)
        else:
            input_emb = torch.cat([time_token, hidden_states], dim=1)
        output = self.llm(inputs_embeds=input_emb, attention_mask=attention_mask, position_ids=position_ids,
                          past_key_values=past_key_values, offload_model=offload_model)
        output, past_key_values = output.last_hidden_state, output.past_key_values

        image_embedding = output[:, -num_tokens_for_output_image:]
        time_emb = self.t_embedder(timestep, dtype=hidden_states.dtype)
        x = self.final_layer(image_embedding, time_emb)
        latents = self.unpatchify(x, height, width)

        if return_past_key_values:
            return latents, past_key_values
        return latents





