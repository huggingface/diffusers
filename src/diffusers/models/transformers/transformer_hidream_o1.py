# Copyright 2026 chinoll and The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextAttention,
    apply_rotary_pos_emb,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import check_model_inputs

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_utils import ModelMixin
from ...utils import BaseOutput
from ..attention import AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn


def _hidream_o1_text_rotary_forward(self, x: torch.Tensor, position_ids: torch.Tensor):
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    inv_freq = self.original_inv_freq
    inv_freq_expanded = inv_freq[None, None, :, None].float().to(device=x.device).expand(
        3, position_ids.shape[1], -1, 1
    )
    position_ids_expanded = position_ids[:, :, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


_hidream_o1_text_rotary_forward = torch.no_grad()(dynamic_rope_update(_hidream_o1_text_rotary_forward))


def _patch_hidream_o1_text_rotary_embedding(rotary_emb):
    if not hasattr(rotary_emb, "original_inv_freq"):
        rotary_emb.original_inv_freq = rotary_emb.inv_freq.detach().float().clone()
    rotary_emb.forward = _hidream_o1_text_rotary_forward.__get__(rotary_emb, type(rotary_emb))


class HiDreamO1AttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("HiDreamO1AttnProcessor requires PyTorch 2.0 or newer.")

    def _attention(self, query, key, value, softmax_scale: float, causal: bool, attention_kwargs: Optional[dict] = None):
        if key.shape[2] != query.shape[2]:
            if query.shape[2] % key.shape[2] != 0:
                raise ValueError(f"Cannot expand key/value heads from {key.shape[2]} to {query.shape[2]}.")
            repeat_factor = query.shape[2] // key.shape[2]
            key = key.repeat_interleave(repeat_factor, dim=2)
            value = value.repeat_interleave(repeat_factor, dim=2)

        return dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
            attention_kwargs=attention_kwargs,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        idx_ar: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        head_dim = attn.head_dim
        hidden_shape = (*input_shape, -1, head_dim)

        query = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape))
        key = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape))
        value = attn.v_proj(hidden_states).view(hidden_shape)

        cos, sin = position_embeddings
        query_rot = query.transpose(1, 2)
        key_rot = key.transpose(1, 2)
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)
        query = query_rot.transpose(1, 2).contiguous()
        key = key_rot.transpose(1, 2).contiguous()
        value = value.contiguous()

        softmax_scale = head_dim**-0.5
        query_ar = query[:, idx_ar].contiguous()
        key_ar = key[:, idx_ar].contiguous()
        value_ar = value[:, idx_ar].contiguous()

        out_ar = self._attention(query_ar, key_ar, value_ar, softmax_scale, causal=True, attention_kwargs=kwargs)
        out_full = self._attention(query, key, value, softmax_scale, causal=False, attention_kwargs=kwargs)
        out_full = out_full.clone()
        out_full[:, idx_ar] = out_ar

        attention_output = out_full.reshape(*input_shape, -1).contiguous()
        return attn.o_proj(attention_output)


class HiDreamO1Attention(Qwen3VLTextAttention, AttentionModuleMixin):
    _default_processor_cls = HiDreamO1AttnProcessor
    _available_processors = [HiDreamO1AttnProcessor]
    _supports_qkv_fusion = False

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.set_processor(self._default_processor_cls())


@dataclass
class HiDreamO1Transformer2DModelOutput(BaseOutput):
    """
    Output of [`HiDreamO1Transformer2DModel`].

    Args:
        sample (`torch.Tensor`):
            Predicted raw RGB pixel patches, with shape `(batch_size, sequence_length, 3 * patch_size * patch_size)`.
        mid_results (`list[torch.Tensor]`, *optional*):
            Optional hidden states returned by selected decoder layers.
        cond_image_embeds (`torch.Tensor`, *optional*):
            Cached conditioning image embeddings for reference-image generation.
        cond_deepstack_image_embeds (`list[torch.Tensor]`, *optional*):
            Cached DeepStack conditioning image embeddings for reference-image generation.
    """

    sample: torch.Tensor
    mid_results: Optional[list[torch.Tensor]] = None
    cond_image_embeds: Optional[torch.Tensor] = None
    cond_deepstack_image_embeds: Optional[list[torch.Tensor]] = None


@dataclass
class HiDreamO1Qwen3VLModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    x_pred: Optional[torch.FloatTensor] = None
    mid_results: Optional[list[torch.Tensor]] = None
    cond_image_embeds: Optional[torch.FloatTensor] = None
    cond_deepstack_image_embeds: Optional[list[torch.Tensor]] = None


@dataclass
class HiDreamO1Qwen3VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    x_pred: Optional[torch.FloatTensor] = None
    mid_results: Optional[list[torch.Tensor]] = None
    cond_image_embeds: Optional[torch.FloatTensor] = None
    cond_deepstack_image_embeds: Optional[list[torch.Tensor]] = None


class HiDreamO1BottleneckPatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 32, in_channels: int = 3, pca_dim: int = 768, embed_dim: int = 768):
        super().__init__()
        self.proj1 = nn.Linear(patch_size * patch_size * in_channels, pca_dim, bias=False)
        self.proj2 = nn.Linear(pca_dim, embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(hidden_states))


class HiDreamO1FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)


class HiDreamO1TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timestep_freq = self.timestep_embedding(timesteps * 1000, self.frequency_embedding_size)
        return self.mlp(timestep_freq.to(dtype=self.mlp[0].weight.dtype))


class HiDreamO1Qwen3VLModel(Qwen3VLModel):
    def __init__(
        self,
        config: Qwen3VLConfig,
        patch_size: int = 32,
        in_channels: int = 3,
        tms_token_id: int = 151673,
    ):
        super().__init__(config)
        _patch_hidream_o1_text_rotary_embedding(self.language_model.rotary_emb)
        for layer_idx, decoder_layer in enumerate(self.language_model.layers):
            decoder_layer.self_attn = HiDreamO1Attention(config.text_config, layer_idx)

        hidden_size = config.text_config.hidden_size
        bottleneck_dim = hidden_size // 4
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.t_embedder1 = HiDreamO1TimestepEmbedder(hidden_size)
        self.x_embedder = HiDreamO1BottleneckPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            pca_dim=bottleneck_dim,
            embed_dim=hidden_size,
        )
        self.t_embedder2 = None
        self.final_layer2 = HiDreamO1FinalLayer(
            hidden_size=hidden_size, patch_size=patch_size, out_channels=in_channels
        )
        self.tms_token_id = tms_token_id

    @property
    def attn_processors(self) -> dict[str, HiDreamO1AttnProcessor]:
        return {
            f"language_model.layers.{layer_idx}.self_attn.processor": decoder_layer.self_attn.processor
            for layer_idx, decoder_layer in enumerate(self.language_model.layers)
        }

    def set_attn_processor(self, processor: HiDreamO1AttnProcessor | dict[str, HiDreamO1AttnProcessor]):
        count = len(self.language_model.layers)
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the "
                f"number of attention layers: {count}. Please pass {count} processor classes."
            )

        for layer_idx, decoder_layer in enumerate(self.language_model.layers):
            if isinstance(processor, dict):
                processor_name = f"language_model.layers.{layer_idx}.self_attn.processor"
                decoder_layer.self_attn.set_processor(processor[processor_name])
            else:
                decoder_layer.self_attn.set_processor(processor)

    def set_default_attn_processor(self):
        self.set_attn_processor(HiDreamO1AttnProcessor())

    def _run_decoder_two_pass_attention(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        token_types: torch.Tensor,
        attention_kwargs: Optional[dict[str, Any]] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        return_mid_results_layers: Optional[list[int]] = None,
    ):
        text_model = self.language_model
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        cos, sin = text_model.rotary_emb(inputs_embeds, position_ids)
        is_gen = token_types[0].bool()
        idx_ar = torch.nonzero(~is_gen, as_tuple=False).squeeze(-1)
        hidden_states = inputs_embeds
        mid_results = [] if return_mid_results_layers is not None else None
        use_gradient_checkpointing = text_model.gradient_checkpointing and torch.is_grad_enabled()
        attention_kwargs = {} if attention_kwargs is None else dict(attention_kwargs)

        def two_pass_layer_forward(hidden_states, decoder_layer, cos, sin, idx_ar):
            residual = hidden_states
            hidden_states = decoder_layer.input_layernorm(hidden_states)
            hidden_states = decoder_layer.self_attn.processor(
                decoder_layer.self_attn,
                hidden_states,
                position_embeddings=(cos, sin),
                idx_ar=idx_ar,
                **attention_kwargs,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            return hidden_states

        for layer_idx, decoder_layer in enumerate(text_model.layers):
            if use_gradient_checkpointing:
                hidden_states = text_model._gradient_checkpointing_func(
                    two_pass_layer_forward,
                    hidden_states,
                    decoder_layer,
                    cos,
                    sin,
                    idx_ar,
                )
            else:
                hidden_states = two_pass_layer_forward(hidden_states, decoder_layer, cos, sin, idx_ar)

            if (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and layer_idx < len(deepstack_visual_embeds)
            ):
                hidden_states = text_model._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

            if return_mid_results_layers is not None and layer_idx in return_mid_results_layers:
                mid_results.append(hidden_states)

        hidden_states = text_model.norm(hidden_states)
        return hidden_states, mid_results

    def _forward_generation(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        vinputs: torch.Tensor,
        timestep: torch.Tensor,
        token_types: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_kwargs: Optional[dict[str, Any]] = None,
        return_mid_results_layers: Optional[list[int]] = None,
        precomputed_image_embeds: Optional[torch.Tensor] = None,
        precomputed_deepstack_image_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ) -> HiDreamO1Qwen3VLModelOutputWithPast:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None
        cond_image_embeds_out = None
        cond_deepstack_image_embeds_out = None

        if pixel_values is not None:
            if precomputed_image_embeds is not None and precomputed_deepstack_image_embeds is not None:
                image_embeds = precomputed_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                deepstack_image_embeds = [
                    embed.to(inputs_embeds.device, inputs_embeds.dtype)
                    for embed in precomputed_deepstack_image_embeds
                ]
            else:
                image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            cond_image_embeds_out = image_embeds
            cond_deepstack_image_embeds_out = deepstack_image_embeds
        elif torch.is_grad_enabled():
            patch_embed = self.visual.patch_embed
            temporal_patch_size = patch_embed.temporal_patch_size
            spatial_merge_size = self.visual.spatial_merge_size
            num_patches = temporal_patch_size * spatial_merge_size * spatial_merge_size
            patch_dim = patch_embed.in_channels * temporal_patch_size * patch_embed.patch_size * patch_embed.patch_size
            fake_pixel_values = torch.zeros(
                num_patches,
                patch_dim,
                device=inputs_embeds.device,
                dtype=patch_embed.proj.weight.dtype,
            )
            fake_grid = torch.tensor(
                [[temporal_patch_size, spatial_merge_size, spatial_merge_size]],
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            fake_image_embeds, fake_deepstack_image_embeds = self.get_image_features(fake_pixel_values, fake_grid)
            fake_total = torch.cat(fake_image_embeds, dim=0).to(inputs_embeds.dtype).sum()
            for fake_deepstack_image_embed in fake_deepstack_image_embeds:
                fake_total = fake_total + fake_deepstack_image_embed.to(inputs_embeds.dtype).sum()
            inputs_embeds = inputs_embeds + fake_total * inputs_embeds.new_zeros([])

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for image_embed, video_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = image_embed.new_zeros(visual_pos_masks.sum(), image_embed.shape[-1]).to(
                    image_embed.device
                )
                embed_joint[image_mask_joint, :] = image_embed
                embed_joint[video_mask_joint, :] = video_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if isinstance(timestep, list):
            timestep = torch.cat(timestep, dim=0)
        timestep = timestep.to(inputs_embeds.device)
        timestep_embeds = self.t_embedder1(timestep)

        tms_mask = input_ids == self.tms_token_id
        tms_mask = tms_mask.unsqueeze(-1).expand_as(inputs_embeds)
        timestep_embeds = timestep_embeds.unsqueeze(1).expand_as(inputs_embeds)
        inputs_embeds = torch.where(tms_mask, timestep_embeds, inputs_embeds)

        if isinstance(vinputs, list):
            vinputs = torch.cat(vinputs, dim=0)
        vinputs = vinputs.to(inputs_embeds.device)
        vinputs_embedded = self.x_embedder(vinputs).to(inputs_embeds.dtype)
        inputs_embeds = torch.cat([inputs_embeds, vinputs_embedded], dim=1)

        batch_size, total_seq_len, _ = inputs_embeds.shape
        if visual_pos_masks is not None:
            vinputs_seq_len = vinputs_embedded.shape[1]
            if visual_pos_masks.shape[0] != batch_size:
                visual_pos_masks = visual_pos_masks.expand(batch_size, -1)
            vinputs_pad = torch.zeros(
                visual_pos_masks.shape[0],
                vinputs_seq_len,
                dtype=visual_pos_masks.dtype,
                device=visual_pos_masks.device,
            )
            visual_pos_masks = torch.cat([visual_pos_masks, vinputs_pad], dim=1)

        if isinstance(token_types, list):
            token_types = torch.cat(token_types, dim=0)
        token_types = token_types.to(inputs_embeds.device)
        if token_types.dim() == 1:
            token_types = token_types.unsqueeze(0)
        elif token_types.dim() == 2 and token_types.shape[-1] == 1 and token_types.shape[0] == total_seq_len:
            token_types = token_types.squeeze(-1).unsqueeze(0)
        if token_types.shape[0] == 1 and batch_size > 1:
            token_types = token_types.expand(batch_size, -1)

        hidden_states, mid_results = self._run_decoder_two_pass_attention(
            inputs_embeds,
            position_ids,
            token_types,
            attention_kwargs=attention_kwargs,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            return_mid_results_layers=return_mid_results_layers,
        )

        x_pred = self.final_layer2(hidden_states)
        return HiDreamO1Qwen3VLModelOutputWithPast(
            last_hidden_state=hidden_states,
            x_pred=x_pred,
            mid_results=mid_results,
            cond_image_embeds=cond_image_embeds_out,
            cond_deepstack_image_embeds=cond_deepstack_image_embeds_out,
        )

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vinputs: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        token_types: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[dict[str, Any]] = None,
        return_mid_results_layers: Optional[list[int]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, HiDreamO1Qwen3VLModelOutputWithPast]:
        if vinputs is not None:
            return self._forward_generation(
                input_ids=input_ids,
                position_ids=position_ids,
                vinputs=vinputs,
                timestep=timestep,
                token_types=token_types,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_kwargs=attention_kwargs,
                return_mid_results_layers=return_mid_results_layers,
                **kwargs,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            **kwargs,
        )


class HiDreamO1ForConditionalGeneration(Qwen3VLPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(
        self,
        config: Qwen3VLConfig,
        patch_size: int = 32,
        in_channels: int = 3,
        tms_token_id: int = 151673,
    ):
        super().__init__(config)
        self.model = HiDreamO1Qwen3VLModel(
            config,
            patch_size=patch_size,
            in_channels=in_channels,
            tms_token_id=tms_token_id,
        )
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    @property
    def attn_processors(self):
        return self.model.attn_processors

    def set_attn_processor(self, processor):
        self.model.set_attn_processor(processor)

    def set_default_attn_processor(self):
        self.model.set_default_attn_processor()

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        vinputs: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        token_types: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[dict[str, Any]] = None,
        return_mid_results_layers: Optional[list[int]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, HiDreamO1Qwen3VLCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            vinputs=vinputs,
            timestep=timestep,
            token_types=token_types,
            attention_kwargs=attention_kwargs,
            return_mid_results_layers=return_mid_results_layers,
            **kwargs,
        )

        if vinputs is not None:
            return HiDreamO1Qwen3VLCausalLMOutputWithPast(
                x_pred=outputs.x_pred,
                mid_results=outputs.mid_results,
                cond_image_embeds=outputs.cond_image_embeds,
                cond_deepstack_image_embeds=outputs.cond_deepstack_image_embeds,
            )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return HiDreamO1Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )


class HiDreamO1Transformer2DModel(ModelMixin, ConfigMixin):
    """
    Diffusers wrapper for the HiDream-O1 raw pixel patch transformer.

    This class is intentionally not compatible with stock Qwen3-VL. HiDream-O1 adds a patch denoising path on top of
    Qwen3-VL (`vinputs`, `token_types`, timestep embedding, and `x_pred`). Use this class to load O1-compatible
    checkpoints and expose them through Diffusers' `ModelMixin` API.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]
    _repeated_blocks = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]
    _skip_layerwise_casting_patterns = ["x_embedder", "t_embedder", "patch_embed", "norm", "rotary_emb"]

    @register_to_config
    def __init__(
        self,
        qwen_config: Optional[dict] = None,
        patch_size: int = 32,
        in_channels: int = 3,
        tms_token_id: int = 151673,
    ):
        super().__init__()

        qwen_config = Qwen3VLConfig().to_dict() if qwen_config is None else qwen_config
        if isinstance(qwen_config, Qwen3VLConfig):
            qwen_config = qwen_config.to_dict()
        self.qwen_config = Qwen3VLConfig.from_dict(qwen_config)
        self.model = HiDreamO1Qwen3VLModel(
            self.qwen_config,
            patch_size=patch_size,
            in_channels=in_channels,
            tms_token_id=tms_token_id,
        )
        self.lm_head = nn.Linear(
            self.qwen_config.text_config.hidden_size,
            self.qwen_config.text_config.vocab_size,
            bias=False,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load HiDream-O1 weights from a Transformers-style checkpoint.

        Official HiDream-O1 checkpoints are Qwen3-VL checkpoints with extra O1 denoising modules. This method uses a
        patched `PreTrainedModel` class to load sharded Transformers weights, then returns a Diffusers `ModelMixin`
        wrapper around the loaded modules.
        """
        try:
            config_dict = cls.load_config(pretrained_model_name_or_path, **kwargs)
        except Exception:
            config_dict = None
        if isinstance(config_dict, dict) and "qwen_config" in config_dict:
            if model_args:
                raise ValueError("Positional model arguments are not supported for Diffusers-format checkpoints.")
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        patch_size = kwargs.pop("patch_size", 32)
        in_channels = kwargs.pop("in_channels", 3)
        tms_token_id = kwargs.pop("tms_token_id", 151673)

        transformer_model = HiDreamO1ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            patch_size=patch_size,
            in_channels=in_channels,
            tms_token_id=tms_token_id,
            **kwargs,
        )
        model = cls(
            qwen_config=transformer_model.config.to_dict(),
            patch_size=patch_size,
            in_channels=in_channels,
            tms_token_id=tms_token_id,
        )
        model.model = transformer_model.model
        model.lm_head = transformer_model.lm_head
        if hasattr(transformer_model, "hf_device_map"):
            model.hf_device_map = transformer_model.hf_device_map
        model.eval()
        return model

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    @property
    def attn_processors(self):
        return self.model.attn_processors

    def set_attn_processor(self, processor):
        self.model.set_attn_processor(processor)

    def set_default_attn_processor(self):
        self.model.set_default_attn_processor()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        vinputs: torch.Tensor,
        timestep: torch.Tensor,
        token_types: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_kwargs: Optional[dict[str, Any]] = None,
        return_mid_results_layers: Optional[list[int]] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], HiDreamO1Transformer2DModelOutput]:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            vinputs=vinputs,
            timestep=timestep,
            token_types=token_types,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_kwargs=attention_kwargs,
            return_mid_results_layers=return_mid_results_layers,
            **kwargs,
        )
        if not return_dict:
            return (outputs.x_pred,)
        return HiDreamO1Transformer2DModelOutput(
            sample=outputs.x_pred,
            mid_results=outputs.mid_results,
            cond_image_embeds=outputs.cond_image_embeds,
            cond_deepstack_image_embeds=outputs.cond_deepstack_image_embeds,
        )
