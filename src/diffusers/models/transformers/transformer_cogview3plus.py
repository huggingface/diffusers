# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
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


from typing import Dict, List, Union
from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...models.attention import FeedForward
from ...models.attention_processor import Attention, AttentionProcessor, FusedJointAttnProcessor2_0
from ...models.modeling_utils import ModelMixin
from ...models.normalization import AdaLayerNormContinuous
from ...utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from ..embeddings import CombinedTimestepTextProjEmbeddings, CogView3PlusPosEmbed, CogView3PlusImagePatchEmbedding
from ..modeling_outputs import Transformer2DModelOutput
import math

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

import torch
import torch.nn as nn
from typing import Optional, Any


def modulate(tensor, shift, scale):
    return tensor * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CogView3PlusTransformerBlock(nn.Module):
    """
    The Transformer block introduced in CogView3, adjusted to match the original implementation.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int = 512,
        elementwise_affine: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.hidden_size_head = attention_head_dim
        self.time_embed_dim = time_embed_dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # Initialize adaln_module similar to AdalnAttentionMixin
        self.adaln_module = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 12 * dim))

        # Initialize LayerNorm for queries and keys
        self.query_layernorm = nn.LayerNorm(self.hidden_size_head, elementwise_affine=elementwise_affine, eps=eps)
        self.key_layernorm = nn.LayerNorm(self.hidden_size_head, elementwise_affine=elementwise_affine, eps=eps)

        # Input LayerNorm
        self.input_layernorm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine, eps=eps)

        # Attention layer
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            qk_norm=None,
            eps=eps,
        )

        # Post-attention LayerNorm
        self.post_attention_layernorm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine, eps=eps)

        # MLP layer
        self.mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        text_length: int,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Get modulation parameters from adaln_module
        adaln_output = self.adaln_module(emb)
        (
            shift_msa_img,
            scale_msa_img,
            gate_msa_img,
            shift_mlp_img,
            scale_mlp_img,
            gate_mlp_img,
            shift_msa_txt,
            scale_msa_txt,
            gate_msa_txt,
            shift_mlp_txt,
            scale_mlp_txt,
            gate_mlp_txt,
        ) = adaln_output.chunk(12, dim=1)

        gate_msa_img = gate_msa_img.unsqueeze(1)
        gate_mlp_img = gate_mlp_img.unsqueeze(1)
        gate_msa_txt = gate_msa_txt.unsqueeze(1)
        gate_mlp_txt = gate_mlp_txt.unsqueeze(1)

        # Input LayerNorm
        attention_input = self.input_layernorm(hidden_states)

        # Modulate attention input
        text_attention_input = modulate(attention_input[:, :text_length], shift_msa_txt, scale_msa_txt)
        image_attention_input = modulate(attention_input[:, text_length:], shift_msa_img, scale_msa_img)
        attention_input = torch.cat((text_attention_input, image_attention_input), dim=1)

        # Attention
        attention_output = self.attention_forward(attention_input, attention_mask, **kwargs)

        # Apply gate to attention output
        text_hidden_states = hidden_states[:, :text_length]
        image_hidden_states = hidden_states[:, text_length:]
        text_attention_output = attention_output[:, :text_length]
        image_attention_output = attention_output[:, text_length:]

        text_hidden_states = text_hidden_states + gate_msa_txt * text_attention_output
        image_hidden_states = image_hidden_states + gate_msa_img * image_attention_output
        hidden_states = torch.cat((text_hidden_states, image_hidden_states), dim=1)

        # MLP
        mlp_input = self.post_attention_layernorm(hidden_states)

        # Modulate MLP input
        text_mlp_input = modulate(mlp_input[:, :text_length], shift_mlp_txt, scale_mlp_txt)
        image_mlp_input = modulate(mlp_input[:, text_length:], shift_mlp_img, scale_mlp_img)
        mlp_input = torch.cat((text_mlp_input, image_mlp_input), dim=1)

        mlp_output = self.mlp_forward(mlp_input, **kwargs)

        # Apply gate to MLP output
        text_mlp_output = mlp_output[:, :text_length]
        image_mlp_output = mlp_output[:, text_length:]

        text_hidden_states = text_hidden_states + gate_mlp_txt * text_mlp_output
        image_hidden_states = image_hidden_states + gate_mlp_img * image_mlp_output
        hidden_states = torch.cat((text_hidden_states, image_hidden_states), dim=1)

        return hidden_states

    def attention_forward(self, hidden_states, attention_mask, **kwargs):
        # Linear projection to get QKV
        qkv = self.attn.to_qkv(hidden_states)
        mixed_query_layer, mixed_key_layer, mixed_value_layer = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        batch_size, seq_length, _ = hidden_states.size()
        query_layer = mixed_query_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_dim)
        key_layer = mixed_key_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_dim)
        value_layer = mixed_value_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_dim)

        # Transpose for attention computation
        query_layer = query_layer.permute(0, 2, 1, 3)  # [batch, heads, seq_length, head_dim]
        key_layer = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)

        # Apply LayerNorm to queries and keys
        query_layer = self.query_layernorm(query_layer)
        key_layer = self.key_layernorm(key_layer)

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_dim)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Compute attention output
        context_layer = torch.matmul(attention_probs, value_layer)

        # Reshape back to [batch_size, seq_length, hidden_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.dim)

        # Output projection
        attention_output = self.attn.to_out(context_layer)

        return attention_output

    def mlp_forward(self, hidden_states, **kwargs):
        return self.mlp(hidden_states)


class CogView3PlusFinalLayer(nn.Module):
    """
    The final layer for the CogView3Plus model to process the transformer outputs into image patches and reconstruct the full image.

    Parameters:
        hidden_size (`int`): The hidden size of the transformer outputs.
        time_embed_dim (`int`): The dimension of the time embedding.
        patch_size (`int`): The patch size to use for generating patches.
        block_size (`int`): The block size for the patchify and unpatchify process.
        out_channels (`int`): The number of output channels (e.g., RGB = 3).
        eps (`float`, defaults to `1e-6`): The epsilon value for numerical stability in LayerNorm.
    """

    def __init__(
        self,
        hidden_size: int,
        time_embed_dim: int,
        patch_size: int,
        block_size: int,
        out_channels: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.block_size = block_size
        self.out_channels = out_channels

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * hidden_size),
        )
        self.linear = nn.Linear(hidden_size, out_channels * patch_size**2)

    def forward(self, logits, emb, text_length, target_size=None, **kwargs):
        # Process the logits and the embedding
        x = logits[:, text_length:]
        shift, scale = self.adaln(emb).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        # Unpatchify
        target_height, target_width = target_size[0]
        assert (
            target_height % self.block_size == 0 and target_width % self.block_size == 0
        ), "target size must be divisible by block size"

        out_height = (target_height // self.block_size) * self.patch_size
        out_width = (target_width // self.block_size) * self.patch_size

        # Calculate the number of patches along height and width
        num_patches_h = out_height // self.patch_size
        num_patches_w = out_width // self.patch_size

        # Step 1: Reshape x to split the patches and their respective dimensions
        # Original shape is (b, (h*w), (c*p1*p2)), so we reshape to (b, h, w, c, p1, p2)
        x = x.view(x.size(0), num_patches_h, num_patches_w, self.out_channels, self.patch_size, self.patch_size)

        # Step 2: Permute to rearrange the dimensions to get the correct shape
        # Moving the patch dimensions back into the spatial dimensions
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()

        # Step 3: Reshape back to (b, c, height, width)
        # This merges the patches back into the original image format
        x = x.view(x.size(0), self.out_channels, out_height, out_width)

        return x

    def reinit(self, parent_model=None):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)


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
        out_channels: int = 16,
        pos_embed_max_size: int = 128,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = CogView3PlusPosEmbed(
            max_height=self.config.sample_size,
            max_width=self.config.sample_size,
            hidden_size=self.inner_dim,
            text_length=224,
            block_size=self.config.patch_size,
        )

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.image_patch_embed = CogView3PlusImagePatchEmbedding(
            in_channels=self.config.in_channels,
            hidden_size=self.inner_dim,
            patch_size=self.config.patch_size,
            text_hidden_size=self.config.pooled_projection_dim,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                CogView3PlusTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    time_embed_dim=self.config.time_embed_dim,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # Final layer for reconstructing the full image
        self.final_layer = CogView3PlusFinalLayer(
            hidden_size=self.inner_dim,
            time_embed_dim=self.config.time_embed_dim,
            patch_size=self.config.patch_size,
            block_size=self.config.patch_size,
            out_channels=self.out_channels,
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    def forward(self, hidden_states, emb, target_size, **kwargs):
        # Transformer processing
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, emb, **kwargs)

        # Final normalization and projection
        hidden_states = self.norm_out(hidden_states)
        logits = self.proj_out(hidden_states)

        # Pass through final layer to reconstruct the image
        output = self.final_layer(logits, emb, text_length=224, target_size=target_size)
        return output

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
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`CogView3PlusTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
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
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

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
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
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
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
