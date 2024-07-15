# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, List, Optional, Tuple, Union
from math import pi

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import UNet2DConditionLoadersMixin
from ...models.activations import get_activation
from ...models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from ...models.embeddings import (
    TimestepEmbedding,
    Timesteps,
)
from ...models.modeling_utils import ModelMixin
from ...models.resnet import Downsample2D, ResnetBlock2D, Upsample2D
from ...models.transformers.transformer_2d import Transformer2DModel, Transformer2DModelOutput
from ...models.unets.unet_2d_blocks import DownBlock2D, UpBlock2D
from ...models.unets.unet_2d_condition import UNet2DConditionOutput
from ...utils import BaseOutput, is_torch_version, logging
from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...models.attention import BasicTransformerBlock, FeedForward, _chunked_feed_forward
from ...models.attention_processor import Attention, AttentionProcessor, StableAudioAttnProcessor2_0
from ...models.modeling_utils import ModelMixin
from ...models.normalization import AdaLayerNormContinuous
from ...utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph


from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...models.attention import BasicTransformerBlock, FeedForward, _chunked_feed_forward
from ...models.attention_processor import Attention, AttentionProcessor, StableAudioAttnProcessor2_0
from ...models.modeling_utils import ModelMixin
from ...models.normalization import AdaLayerNormContinuous
from ...models.embeddings import GaussianFourierProjection
from ...utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StableAudioPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        times = times[..., None]
        freqs = times * self.weights[None] * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((times, fouriered), dim=-1)
        return fouriered

@dataclass
class StableAudioProjectionModelOutput(BaseOutput):
    """
    Args:
    Class for StableAudio projection layer's outputs.
        text_hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states obtained by linearly projecting the hidden-states for the text encoder.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices, formed by concatenating the attention masks
             for the two text encoders together. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
    """

    text_hidden_states: torch.Tensor
    seconds_start_hidden_states: torch.Tensor
    seconds_end_hidden_states: torch.Tensor
    attention_mask: Optional[torch.LongTensor] = None


class StableAudioNumberConditioner(nn.Module):
    """
    A simple linear projection model to map numbers to a latent space.

    Args:
        number_embedding_dim (`int`):
            Dimensionality of the number embeddings.
        min_value (`int`):
            The minimum value of the seconds number conditioning modules.
        max_value (`int`):
            The maximum value of the seconds number conditioning modules
        internal_dim (`int`):
            Dimensionality of the intermediate number hidden states.
    """

    def __init__(
        self,
        number_embedding_dim,
        min_value,
        max_value,
        internal_dim: Optional[int]=256,
    ):
        super().__init__()
        self.time_positional_embedding = nn.Sequential(
        StableAudioPositionalEmbedding(internal_dim),
        nn.Linear(in_features=internal_dim + 1, out_features=number_embedding_dim),
        )
        
        self.number_embedding_dim = number_embedding_dim 
        self.min_value = min_value
        self.max_value = max_value


    def forward(
        self,
        floats: List[float],
    ):    
        # Cast the inputs to floats
        floats = [float(x) for x in floats]
        floats = torch.tensor(floats).to(self.time_positional_embedding[1].weight.device)

        floats = floats.clamp(self.min_value, self.max_value)

        normalized_floats = (floats - self.min_value) / (self.max_value - self.min_value)

        # Cast floats to same type as embedder
        embedder_dtype = next(self.time_positional_embedding.parameters()).dtype
        normalized_floats = normalized_floats.to(embedder_dtype)

        embedding = self.time_positional_embedding(normalized_floats)
        float_embeds = embedding.view(-1, 1, self.number_embedding_dim)

        return float_embeds


class StableAudioProjectionModel(ModelMixin, ConfigMixin):
    """
    A simple linear projection model to map the conditioning values to a shared latent space.

    Args:
        text_encoder_dim (`int`):
            Dimensionality of the text embeddings from the text encoder (T5).
        conditioning_dim (`int`):
            Dimensionality of the output conditioning tensors.
        min_value (`int`):
            The minimum value of the seconds number conditioning modules.
        max_value (`int`):
            The maximum value of the seconds number conditioning modules
    """

    @register_to_config
    def __init__(
        self,
        text_encoder_dim,
        conditioning_dim,
        min_value,
        max_value
    ):
        super().__init__()
        self.text_projection = nn.Identity() if conditioning_dim == text_encoder_dim else nn.Linear(text_encoder_dim, conditioning_dim)
        self.start_number_conditioner = StableAudioNumberConditioner(conditioning_dim, min_value, max_value)
        self.end_number_conditioner = StableAudioNumberConditioner(conditioning_dim, min_value, max_value)

    def forward(
        self,
        text_hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.LongTensor],
        start_seconds: List[float],
        end_seconds: List[float],
    ):
        text_hidden_states = self.text_projection(text_hidden_states)
        seconds_start_hidden_states = self.start_number_conditioner(start_seconds)
        seconds_end_hidden_states = self.end_number_conditioner(end_seconds)


        return StableAudioProjectionModelOutput(
            text_hidden_states=text_hidden_states,
            attention_mask=attention_mask,
            seconds_start_hidden_states=seconds_start_hidden_states,
            seconds_end_hidden_states=seconds_end_hidden_states,
        )

@maybe_allow_in_graph
class StableAudioDiTBlock(nn.Module):
    r"""
    Transformer block used in Stable Audio model (https://github.com/Stability-AI/stable-audio-tools). Allow skip connection and
    QKNorm

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for the query states.
        num_key_value_attention_heads (`int`): The number of heads to use for the key and value states.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_key_value_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "swiglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = False,
    ):
        super().__init__()
        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            processor=StableAudioAttnProcessor2_0(),
        )

        # 2. Cross-Attn
        self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            kv_heads=num_key_value_attention_heads,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            processor=StableAudioAttnProcessor2_0(),
        )  # is self-attn if encoder_hidden_states is none

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        rotary_embedding: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}

        attn_output = self.attn1(
            norm_hidden_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_embedding,
            **cross_attention_kwargs,
        )

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class StableAudioDiTModel(ModelMixin, ConfigMixin):
    """
    The Diffusion Transformer model introduced in Stable Audio.

    Reference: https://github.com/Stability-AI/stable-audio-tools

    Parameters:
        sample_size ( `int`, *optional*, defaults to 1024): The size of the input sample.
        in_channels (`int`, *optional*, defaults to 64): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 24): The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 24): The number of heads to use for the query states.
        num_key_value_attention_heads (`int`, *optional*, defaults to 12): The number of heads to use for the key and value states.
        out_channels (`int`, defaults to 64): Number of output channels.
        cross_attention_dim ( `int`, *optional*, defaults to 768): Dimension of the cross-attention projection.
        timestep_features_dim ( `int`, *optional*, defaults to 256): Dimension of the timestep inner projection.
        global_states_input_dim ( `int`, *optional*, defaults to 1536): Input dimension of the global hidden states projection. 
        cross_attention_input_dim ( `int`, *optional*, defaults to 768): Input dimension of the cross-attention projection
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 1024,
        in_channels: int = 64,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        num_key_value_attention_heads: int = 12,
        out_channels: int = 64,
        cross_attention_dim: int = 768,
        timestep_features_dim: int = 256,
        global_states_input_dim: int = 1536,
        cross_attention_input_dim: int = 768,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.timestep_features = GaussianFourierProjection(embedding_size=timestep_features_dim//2, flip_sin_to_cos=True, log=False, set_W_to_weight=False, use_stable_audio_implementation=True)

        self.timestep_proj = nn.Sequential(
            nn.Linear(timestep_features_dim, self.inner_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim, bias=True),
        )

        self.global_proj = nn.Sequential(
                nn.Linear(global_states_input_dim, self.inner_dim, bias=False),
                nn.SiLU(),
                nn.Linear(self.inner_dim, self.inner_dim, bias=False)
            )

        self.cross_attention_proj = nn.Sequential(
                nn.Linear(cross_attention_input_dim, cross_attention_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cross_attention_dim, cross_attention_dim, bias=False)
            )
        
        self.preprocess_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False) 
        self.proj_in = nn.Linear(in_channels, self.inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList(
            [
                StableAudioDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    num_key_value_attention_heads=num_key_value_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for i in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(self.out_channels, self.out_channels, 1, bias=False)

        self.gradient_checkpointing = False

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

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
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

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
            
    # Copied from diffusers.models.transformers.hunyuan_transformer_2d.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(StableAudioAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.HunyuanDiT2DModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        global_hidden_states: torch.FloatTensor = None,
        rotary_embedding: torch.FloatTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, in_channels, sequence_len)`):
                Input `hidden_states`.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, encoder_sequence_len, cross_attention_input_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            global_hidden_states (`torch.FloatTensor` of shape `(batch size, global_sequence_len, global_states_input_dim)`):
               Global embeddings that will be prepended to the hidden states.
            rotary_embedding (`torch.Tensor`):
                The rotary embeddings to apply on query and key tensors during attention calculation.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_len)`, *optional*):
                Mask to avoid performing attention on padding token indices, formed by concatenating the attention masks
                    for the two text encoders together. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_len)`, *optional*):
                Mask to avoid performing attention on padding token cross-attention indices, formed by concatenating the attention masks
                    for the two text encoders together. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """                
        cross_attention_hidden_states = self.cross_attention_proj(encoder_hidden_states)
        global_hidden_states = self.global_proj(global_hidden_states)
        time_hidden_states = self.timestep_proj(self.timestep_features(timestep.to(self.dtype)))
        
        global_hidden_states = global_hidden_states + time_hidden_states.unsqueeze(1)
        
        
        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        # (batch_size, dim, sequence_length) -> (batch_size, sequence_length, dim)
        hidden_states = hidden_states.transpose(1,2)
        
        hidden_states = self.proj_in(hidden_states)

        # prepend global states to hidden states
        hidden_states = torch.cat([global_hidden_states, hidden_states], dim=-2)
        if attention_mask is not None:
            prepend_mask = torch.ones((hidden_states.shape[0], 1), device=hidden_states.device, dtype=torch.bool)
            attention_mask = torch.cat([prepend_mask, attention_mask], dim=-1)


        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    cross_attention_hidden_states,
                    encoder_attention_mask,
                    rotary_embedding,
                    joint_attention_kwargs,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states = hidden_states,
                    attention_mask = attention_mask,
                    encoder_hidden_states = cross_attention_hidden_states,
                    encoder_attention_mask = encoder_attention_mask,
                    rotary_embedding = rotary_embedding,
                    cross_attention_kwargs = joint_attention_kwargs,
                )

        hidden_states = self.proj_out(hidden_states)
        
        # (batch_size, sequence_length, dim) -> (batch_size, dim, sequence_length)
        # remove prepend length that has been added by global hidden states
        hidden_states = hidden_states.transpose(1,2)[:, :, 1:]
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states
        

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)