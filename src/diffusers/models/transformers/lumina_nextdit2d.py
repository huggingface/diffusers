# Copyright 2024 Alpha-VLLM Authors and The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..attention import LuminaFeedForward
from ..attention_processor import Attention, LuminaAttnProcessor2_0
from ..embeddings import (
    LuminaCombinedTimestepCaptionEmbedding,
)
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))


class LuminaFinalLayer(nn.Module):
    """
    The final layer of LuminaNextDiT.

    Parameters:
        hidden_size (`int`):
            The dimensionality of the hidden layers in the model. This parameter determines the width of the model's
            hidden representations.
        patch_size (`int`): The patch size of noise.
        out_channels (`int`): The number of output channels.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                min(hidden_size, 1024),
                hidden_size,
                bias=True,
            ),
        )

    def forward(self, x, c):
        """
        Forward pass of the LuminaFinalLayer.

        Args:
            x (torch.Tensor): The input tensor.
            c (torch.Tensor): The conditioning tensor.
        Returns:
            torch.Tensor: The output tensor.
        """
        scale = self.adaLN_modulation(c)

        x = modulate(self.norm_final(x), scale)
        x = self.linear(x)

        return x


class LuminaNextDiTBlock(nn.Module):
    """
    Initialize a LuminaNextDiTBlock.

    Args:
        layer_id (int): Identifier for the layer.
        hidden_size (int): Embedding dimension of the input features.
        num_attention_heads (int): Number of attention heads.
        num_kv_heads (Optional[int]): Number of attention heads in key and
            value features (if using GQA), or set to None for the same as query.
        multiple_of (int):
        ffn_dim_multiplier (float):
        norm_eps (float):

    Attributes:
        num_attention_heads (int): Number of attention heads.
        hidden_size (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        cross_attention_dim: int,
        norm_elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_attention_heads

        self.gate = nn.Parameter(torch.zeros([num_attention_heads]))

        # Self-attention
        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            dim_head=hidden_size // num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=LuminaAttnProcessor2_0(),
        )
        self.attn.to_out = nn.Identity()

        # Cross-attention
        self.cross_attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=encoder_hidden_size,
            dim_head=hidden_size // num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=LuminaAttnProcessor2_0(),
        )

        self.feed_forward = LuminaFeedForward(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        self.attn_norm1 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.ffn_norm1 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        self.attn_norm2 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.ffn_norm2 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                min(hidden_size, 1024),
                4 * hidden_size,
                bias=True,
            ),
        )

        self.attn_encoder_hidden_states_norm = RMSNorm(
            encoder_hidden_size, eps=norm_eps, elementwise_affine=norm_elementwise_affine
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
    ):
        """
        Perform a forward pass through the LuminaNextDiTBlock.

        Args:
            hidden_states (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
        Returns:
            torch.Tensor: Output tensor after applying attention and
                feedforward layers.
        """
        residual = hidden_states

        if adaln_input is not None:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

            # Self-attention
            hidden_states = modulate(self.attn_norm1(hidden_states), scale_msa)
            self_attn_output = self.attn(
                hidden_states=hidden_states,
                encoder_hidden_states=hidden_states,
                attention_mask=attention_mask,
                query_rotary_emb=freqs_cis,
                key_rotary_emb=freqs_cis,
                **cross_attention_kwargs,
            )

            # Cross-attention
            norm_encoder_hidden_states = self.attn_encoder_hidden_states_norm(encoder_hidden_states)
            cross_attn_output = self.cross_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                attention_mask=encoder_mask,
                query_rotary_emb=freqs_cis,
                key_rotary_emb=None,
                **cross_attention_kwargs,
            )
            cross_attn_output = cross_attn_output * self.gate.tanh().view(1, 1, -1, 1)
            mixed_attn_output = self_attn_output + cross_attn_output
            mixed_attn_output = mixed_attn_output.flatten(-2)
            # linear proj
            hidden_states = self.cross_attn.to_out[0](mixed_attn_output)

            hidden_states = residual + gate_msa.unsqueeze(1).tanh() * self.attn_norm2(hidden_states)

            mlp_output = self.feed_forward(
                modulate(self.ffn_norm1(hidden_states), scale_mlp),
            )
            hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)
        else:
            hidden_states = self.attn_norm1(hidden_states)
            # Self-attention
            self_attn_output = self.attn(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                image_rotary_emb=freqs_cis,
                **cross_attention_kwargs,
            )

            # Cross-attention
            cross_attn_output = self.cross_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=self.attn_encoder_hidden_states_norm(encoder_hidden_states),
                attention_mask=encoder_mask,
                image_rotary_emb=None,
                **cross_attention_kwargs,
            )
            hidden_states = residual + self.attn_norm2(cross_attn_output)

            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
            hidden_states = hidden_states + self.ffn_norm2(hidden_states)

        return hidden_states


class LuminaNextDiT2DModel(ModelMixin, ConfigMixin):
    """
    LuminaNextDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`, *optional*, (`int`, *optional*, defaults to 2):
            The size of each patch in the image. This parameter defines the resolution of patches fed into the model.
        in_channels (`int`, *optional*, defaults to 4):
            The number of input channels for the model. Typically, this matches the number of channels in the input
            images.
        hidden_size (`int`, *optional*, defaults to 4096):
            The dimensionality of the hidden layers in the model. This parameter determines the width of the model's
            hidden representations.
        num_layers (`int`, *optional*, default to 32):
            The number of layers in the model. This defines the depth of the neural network.
        num_attention_heads (`int`, *optional*, defaults to 32):
            The number of attention heads in each attention layer. This parameter specifies how many separate attention
            mechanisms are used.
        num_kv_heads (`int`, *optional*, defaults to 8):
            The number of key-value heads in the attention mechanism, if different from the number of attention heads.
            If None, it defaults to num_attention_heads.
        multiple_of (`int`, *optional*, defaults to 256):
            A factor that the hidden size should be a multiple of. This can help optimize certain hardware
            configurations.
        ffn_dim_multiplier (`float`, *optional*):
            A multiplier for the dimensionality of the feed-forward network. If None, it uses a default value based on
            the model configuration.
        norm_eps float = (`float`, *optional*, defaults to 1e-5):
            A small value added to the denominator for numerical stability in normalization layers.
        learn_sigma bool = (`bool`, *optional*, defaults to True):
            Whether the model should learn the sigma parameter, which might be related to uncertainty or variance in
            predictions.
        qk_norm (`bool`, *optional*, defaults to True):
            Indicates if the queries and keys in the attention mechanism should be normalized.
        encoder_hidden_size (`int`, *optional*, defaults to 2048):
            The dimensionality of the text embeddings. This parameter defines the size of the text representations used
            in the model.
        scaling_factor (`float`, *optional*, defaults to 1.0):
            A scaling factor applied to certain parameters or layers in the model. This can be used for adjusting the
            overall scale of the model's operations.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: Optional[int] = 2,
        in_channels: Optional[int] = 4,
        hidden_size: Optional[int] = 2304,
        num_layers: Optional[int] = 32,
        num_attention_heads: Optional[int] = 32,
        num_kv_heads: Optional[int] = None,
        multiple_of: Optional[int] = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: Optional[float] = 1e-5,
        learn_sigma: Optional[bool] = True,
        qk_norm: Optional[bool] = True,
        encoder_hidden_size: Optional[int] = 2048,
        scaling_factor: Optional[float] = 1.0,
    ) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling_factor = scaling_factor

        self.patch_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
            bias=True,
        )

        self.pad_token = nn.Parameter(torch.empty(hidden_size))

        self.time_caption_embed = LuminaCombinedTimestepCaptionEmbedding(
            hidden_size=min(hidden_size, 1024), encoder_hidden_size=encoder_hidden_size
        )

        self.layers = nn.ModuleList(
            [
                LuminaNextDiTBlock(
                    layer_id,
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    encoder_hidden_size,
                )
                for layer_id in range(num_layers)
            ]
        )
        self.final_layer = LuminaFinalLayer(hidden_size, patch_size, self.out_channels)

        assert (hidden_size // num_attention_heads) % 4 == 0, "2d rope needs head dim to be divisible by 4"
        self.freqs_cis = self.precompute_freqs_cis(
            hidden_size // num_attention_heads,
            384,
            scaling_factor=scaling_factor,
        )

    def unpatchify(self, x: torch.Tensor, img_size: List[Tuple[int, int]], return_tensor=False) -> List[torch.Tensor]:
        """
        Reconstructs the original images from the patchified tensor.

        Args:
            x (torch.Tensor): The patchified tensor of shape (N, T, patch_size**2 * C).
            img_size (List[Tuple[int, int]]): The list of image sizes for each image in the batch.
            return_tensor (bool, optional): Whether to return the reconstructed images as a tensor.
                If False, the reconstructed images will be returned as a list of tensors. Defaults to False.

        Returns:
            List[torch.Tensor] or torch.Tensor: The reconstructed images.
                If return_tensor is True, the reconstructed images will be returned as a tensor of shape (N, C, H, W).
                If return_tensor is False, the reconstructed images will be returned as a list of tensors, where each
                tensor has shape (H, W, C).
        """
        pH = pW = self.patch_size
        if return_tensor:
            H, W = img_size[0]
            B = x.size(0)
            L = (H // pH) * (W // pW)
            x = x[:, :L].view(B, H // pH, W // pW, pH, pW, self.out_channels)
            x = x.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)
            return x
        else:
            imgs = []
            for i in range(x.size(0)):
                H, W = img_size[i]
                L = (H // pH) * (W // pW)
                imgs.append(
                    x[i][:L]
                    .view(H // pH, W // pW, pH, pW, self.out_channels)
                    .permute(4, 0, 2, 1, 3)
                    .flatten(3, 4)
                    .flatten(1, 2)
                )
        return imgs

    def patchify_and_embed(
        self, x: Union[List[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]], torch.Tensor]:
        """
        Patchifies and embeds the input tensor(s).

        Args:
            x (List[torch.Tensor] | torch.Tensor): The input tensor(s) to be patchified and embedded.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]], torch.Tensor]: A tuple containing the patchified
            and embedded tensor(s), the mask indicating the valid patches, the original image size(s), and the
            frequency tensor(s).
        """
        self.freqs_cis = self.freqs_cis.to(x[0].device)
        if isinstance(x, torch.Tensor):
            pH = pW = self.patch_size
            B, C, H, W = x.size()
            x = x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 1, 3, 5).flatten(3)
            x = self.patch_embedder(x)
            x = x.flatten(1, 2)

            mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.int32, device=x.device)

            return (
                x,
                mask,
                [(H, W)] * B,
                self.freqs_cis[: H // pH, : W // pW].flatten(0, 1).unsqueeze(0),
            )
        else:
            pH = pW = self.patch_size
            x_embed = []
            freqs_cis = []
            img_size = []
            l_effective_seq_len = []

            for img in x:
                C, H, W = img.size()
                item_freqs_cis = self.freqs_cis[: H // pH, : W // pW]
                freqs_cis.append(item_freqs_cis.flatten(0, 1))
                img_size.append((H, W))
                img = img.view(C, H // pH, pH, W // pW, pW).permute(1, 3, 0, 2, 4).flatten(2)
                img = self.x_embedder(img)
                img = img.flatten(0, 1)
                l_effective_seq_len.append(len(img))
                x_embed.append(img)

            max_seq_len = max(l_effective_seq_len)
            mask = torch.zeros(len(x), max_seq_len, dtype=torch.int32, device=x[0].device)
            padded_x_embed = []
            padded_freqs_cis = []
            for i, (item_embed, item_freqs_cis, item_seq_len) in enumerate(
                zip(x_embed, freqs_cis, l_effective_seq_len)
            ):
                item_embed = torch.cat(
                    [
                        item_embed,
                        self.pad_token.view(1, -1).expand(max_seq_len - item_seq_len, -1),
                    ],
                    dim=0,
                )
                item_freqs_cis = torch.cat(
                    [
                        item_freqs_cis,
                        item_freqs_cis[-1:].expand(max_seq_len - item_seq_len, -1),
                    ],
                    dim=0,
                )
                padded_x_embed.append(item_embed)
                padded_freqs_cis.append(item_freqs_cis)
                mask[i][:item_seq_len] = 1

            x_embed = torch.stack(padded_x_embed, dim=0)
            freqs_cis = torch.stack(padded_freqs_cis, dim=0)

            return x_embed, mask, img_size, freqs_cis

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        scaling_factor: Optional[int] = 1,
        scaling_watershed: Optional[float] = 0.3,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict=True,
    ) -> torch.Tensor:
        """
        Forward pass of LuminaNextDiT.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (N, C, H, W).
            timestep (torch.Tensor): Tensor of diffusion timesteps of shape (N,).
            encoder_hidden_states (torch.Tensor): Tensor of caption features of shape (N, D).
            encoder_mask (torch.Tensor): Tensor of caption masks of shape (N, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W).
        """
        # For inference
        self.freqs_cis = self.precompute_freqs_cis(
            self.head_dim,
            384,
            scaling_factor=scaling_factor,
            scaling_watershed=scaling_watershed,
            timestep=timestep[0].item(),
        )
        x_is_tensor = isinstance(hidden_states, torch.Tensor)
        hidden_states, mask, img_size, freqs_cis = self.patchify_and_embed(hidden_states)
        freqs_cis = freqs_cis.to(hidden_states.device)

        adaln_input = self.time_caption_embed(timestep, encoder_hidden_states, encoder_mask)

        encoder_mask = encoder_mask.bool()
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                mask,
                freqs_cis,
                encoder_hidden_states,
                encoder_mask,
                adaln_input=adaln_input,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        hidden_states = self.final_layer(hidden_states, adaln_input)
        output = self.unpatchify(hidden_states, img_size, return_tensor=x_is_tensor)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def precompute_freqs_cis(
        self,
        dim: int,
        end: int,
        theta: float = 10000.0,
        scaling_factor: float = 1.0,
        scaling_watershed: float = 1.0,
        timestep: float = 1.0,
    ):
        """
        Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

        This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the
        end index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in
        complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            end (int): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation.
                Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex
                exponentials.
        """

        if timestep < scaling_watershed:
            linear_factor = scaling_factor
            ntk_factor = 1.0
        else:
            linear_factor = 1.0
            ntk_factor = scaling_factor

        theta = theta * ntk_factor
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float().cuda() / dim)) / linear_factor

        timestep = torch.arange(end, device=freqs.device, dtype=torch.float)

        freqs = torch.outer(timestep, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

        freqs_cis_h = freqs_cis.view(end, 1, dim // 4, 1).repeat(1, end, 1, 1)
        freqs_cis_w = freqs_cis.view(1, end, dim // 4, 1).repeat(end, 1, 1, 1)
        freqs_cis = torch.cat([freqs_cis_h, freqs_cis_w], dim=-1).flatten(2)

        return freqs_cis
