# Copyright 2024 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
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


from typing import Dict, List, Union, Optional, Any

from ..normalization import AdaLayerNormContinuous
from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...models.attention import FeedForward
from ...models.attention_processor import Attention, AttentionProcessor, FusedJointAttnProcessor2_0
from ...models.modeling_utils import ModelMixin
from ...utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from ..embeddings import CogView3PlusImagePatchEmbedding, CogView3CombineTimestepLabelEmbedding
from ..modeling_outputs import Transformer2DModelOutput
import torch
import torch.nn as nn

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def modulate(hidden_states, shift, scale):
    return hidden_states * (scale + 1) + shift


class CogView3PlusTransformerBlock(nn.Module):
    """
    Updated CogView3 Transformer Block to align with AdalnAttentionMixin style, simplified with qk_ln always True.
    """

    def __init__(
            self,
            num_attention_heads: int = 64,
            attention_head_dim: int = 40,
            time_embed_dim: int = 512,
            elementwise_affine: bool = False,
            eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = attention_head_dim * num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.time_embed_dim = time_embed_dim
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.adaln_modules = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 12 * self.dim))

        # LayerNorm for Query and Key normalization (qk_ln is always True)
        self.query_layernorms = nn.ModuleList([
            nn.LayerNorm(self.attention_head_dim, elementwise_affine=elementwise_affine, eps=eps)
            for _ in range(num_attention_heads)
        ])
        self.key_layernorms = nn.ModuleList([
            nn.LayerNorm(self.attention_head_dim, elementwise_affine=elementwise_affine, eps=eps)
            for _ in range(num_attention_heads)
        ])

        # LayerNorm before Attention
        self.input_layernorm = nn.LayerNorm(self.dim, elementwise_affine=elementwise_affine, eps=eps)

        # Attention block
        self.attn = Attention(
            query_dim=self.dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            out_dim=self.dim,
            bias=True,
            eps=eps,
        )

        # Post-Attention LayerNorm
        self.post_attention_layernorm = nn.LayerNorm(self.dim, elementwise_affine=elementwise_affine, eps=eps)

        # MLP layer
        self.mlp = FeedForward(dim=self.dim, dim_out=self.dim, activation_fn="gelu-approximate")

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            emb: torch.FloatTensor,
            text_length: int = 224,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Retrieve Adaln module shifts/scales
        adaln_module = self.adaln_modules
        (
            shift_msa_img, scale_msa_img, gate_msa_img,
            shift_mlp_img, scale_mlp_img, gate_mlp_img,
            shift_msa_txt, scale_msa_txt, gate_msa_txt,
            shift_mlp_txt, scale_mlp_txt, gate_mlp_txt,
        ) = adaln_module(emb).chunk(12, dim=1)

        # Modulate text and image input separately
        attention_input = self.input_layernorm(hidden_states)
        text_attention_input = modulate(attention_input[:, :text_length], shift_msa_txt, scale_msa_txt)
        image_attention_input = modulate(attention_input[:, text_length:], shift_msa_img, scale_msa_img)
        attention_input = torch.cat((text_attention_input, image_attention_input), dim=1)

        # Attention operation
        attention_output = self.attn(attention_input)

        # Post attention, modulate hidden states again with gates
        text_hidden_states, image_hidden_states = hidden_states[:, :text_length], hidden_states[:, text_length:]
        text_attention_output, image_attention_output = attention_output[:, :text_length], attention_output[:,
                                                                                           text_length:]
        text_hidden_states = text_hidden_states + gate_msa_txt.unsqueeze(1) * text_attention_output
        image_hidden_states = image_hidden_states + gate_msa_img.unsqueeze(1) * image_attention_output
        hidden_states = torch.cat((text_hidden_states, image_hidden_states), dim=1)

        # Post-Attention LayerNorm
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP operation
        mlp_input = self.post_attention_layernorm(hidden_states)
        text_mlp_input = modulate(mlp_input[:, :text_length], shift_mlp_txt, scale_mlp_txt)
        image_mlp_input = modulate(mlp_input[:, text_length:], shift_mlp_img, scale_mlp_img)
        mlp_input = torch.cat((text_mlp_input, image_mlp_input), dim=1)

        mlp_output = self.mlp(mlp_input)

        # Apply gates to MLP output
        text_hidden_states, image_hidden_states = hidden_states[:, :text_length], hidden_states[:, text_length:]
        text_mlp_output, image_mlp_output = mlp_output[:, :text_length], mlp_output[:, text_length:]
        text_hidden_states = text_hidden_states + gate_mlp_txt.unsqueeze(1) * text_mlp_output
        image_hidden_states = image_hidden_states + gate_mlp_img.unsqueeze(1) * image_mlp_output
        hidden_states = torch.cat((text_hidden_states, image_hidden_states), dim=1)

        # Final residual connection
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clamp(-65504, 65504)

        return hidden_states

    def modulate(self, hidden_states, shift, scale):
        return hidden_states * (scale + 1) + shift

class CogView3PlusTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The Transformer model introduced in CogView3.

    Reference: https://arxiv.org/abs/2403.05121
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            sample_size: int = 128,
            patch_size: int = 2,
            in_channels: int = 16,
            num_layers: int = 30,
            attention_head_dim: int = 40,
            num_attention_heads: int = 64,
            joint_attention_dim: int = 4096,
            adm_in_channels: int = 1536,
            pooled_projection_dim: int = 4096,
            caption_projection_dim=1536,
            out_channels: int = 16,
            time_embed_dim: int = 512,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.emb = CogView3CombineTimestepLabelEmbedding(
            time_embed_dim=self.config.time_embed_dim,
            label_embed_dim=self.config.adm_in_channels,
            in_channels=self.inner_dim,
        )

        self.pos_embed = CogView3PlusImagePatchEmbedding(
            in_channels=self.config.in_channels,
            hidden_size=self.inner_dim,
            patch_size=self.config.patch_size,
            text_hidden_size=self.config.pooled_projection_dim,

        )

        self.transformer_blocks = nn.ModuleList(
            [
                CogView3PlusTransformerBlock(
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    time_embed_dim=self.config.time_embed_dim,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.config.time_embed_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

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

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking
    def disable_forward_chunking(self):
        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)

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

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedJointAttnProcessor2_0
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

        self.set_attn_processor(FusedJointAttnProcessor2_0())

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
            y: torch.LongTensor = None,
            target_size: List = None,
            block_controlnet_hidden_states: List = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`CogView3PlusTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor`): Input `hidden_states`.
            timestep (`torch.LongTensor`): Indicates denoising step.
            y (`torch.LongTensor`, *optional*): 标签输入，用于获取标签嵌入。
            block_controlnet_hidden_states: (`list` of `torch.Tensor`): A list of tensors for residuals.
            joint_attention_kwargs (`dict`, *optional*): Additional kwargs for the attention processor.
            return_dict (`bool`, *optional*, defaults to `True`): Whether to return a `Transformer2DModelOutput`.

        Returns:
            Output tensor or `Transformer2DModelOutput`.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        emb = self.emb(timestep, y, hidden_dtype=hidden_states.dtype)

        for index_block, block in enumerate(self.transformer_blocks):
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
                    emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    emb=emb,
                )

            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, emb.to(torch.float32))
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
