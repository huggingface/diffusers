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

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..attention import LuminaFeedForward
from ..embeddings import (
    LuminaCombinedTimestepCaptionEmbedding,
)
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))


class Attention(nn.Module):
    """
    Initialize the Multi-head attention module.

    Args:
        hidden_size (int): Number of input dimensions.
        num_attention_heads (int): Number of heads.
        num_kv_heads (Optional[int]): Number of kv heads, if using GQA.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: Optional[int],
        qk_norm: bool,
        caption_dim: int,
    ):
        super().__init__()
        self.num_kv_heads = num_attention_heads if num_kv_heads is None else num_kv_heads
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = self.num_kv_heads
        self.n_rep = self.num_attention_heads // self.num_kv_heads
        self.head_dim = hidden_size // num_attention_heads

        self.wq = nn.Linear(
            hidden_size,
            num_attention_heads * self.head_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.wq.weight)
        self.wk = nn.Linear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.wk.weight)
        self.wv = nn.Linear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.wv.weight)
        if caption_dim > 0:
            self.wk_cap = nn.Linear(
                caption_dim,
                self.num_kv_heads * self.head_dim,
                bias=False,
            )
            nn.init.xavier_uniform_(self.wk_cap.weight)
            self.wv_cap = nn.Linear(
                caption_dim,
                self.num_kv_heads * self.head_dim,
                bias=False,
            )
            nn.init.xavier_uniform_(self.wv_cap.weight)
            self.gate = nn.Parameter(torch.zeros([self.num_attention_heads]))

        self.wo = nn.Linear(
            num_attention_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        nn.init.xavier_uniform_(self.wo.weight)

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.num_attention_heads * self.head_dim)
            self.k_norm = nn.LayerNorm(self.num_kv_heads * self.head_dim)
            if caption_dim > 0:
                self.k_cap_norm = nn.LayerNorm(self.num_kv_heads * self.head_dim)
            else:
                self.k_cap_norm = nn.Identity()
        else:
            self.q_norm = self.k_norm = nn.Identity()
            self.k_cap_norm = nn.Identity()

        # for proportional attention computation
        self.base_seqlen = None
        self.proportional_attn = False
        self.use_flash_attn = True

    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        """
        Reshape frequency tensor for broadcasting it with another tensor.

        This function reshapes the frequency tensor to have the same shape as the target tensor 'x' for the purpose of
        broadcasting the frequency tensor during element-wise operations.

        Args:
            freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
            x (torch.Tensor): Target tensor for broadcasting compatibility.

        Returns:
            torch.Tensor: Reshaped frequency tensor.

        Raises:
            AssertionError: If the frequency tensor doesn't match the expected
                shape.
            AssertionError: If the target tensor 'x' doesn't have the expected
                number of dimensions.
        """
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(
        x_in: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensors using the given frequency tensor.

        This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
        frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor is
        reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are returned as
        real tensors.

        Args:
            x_in (torch.Tensor): Query or Key tensor to apply rotary embeddings.
            freqs_cis (torch.Tensor): Precomputed frequency tensor for complex
                exponentials.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor
                and key tensor with rotary embeddings.
        """
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x * freqs_cis).flatten(3)
            return x_out.type_as(x_in)

    # copied from huggingface modeling_llama.py
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        def _get_unpad_data(attention_mask):
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
            return (
                indices,
                cu_seqlens,
                max_seqlen_in_batch,
            )

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_attention_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        caption_feat: torch.Tensor,
        caption_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the attention mechanism to the input tensors.

        Args:
            x (torch.Tensor): Input tensor.
            x_mask (torch.Tensor): Mask for the input tensor.
            freqs_cis (torch.Tensor): Frequency tensor for complex exponentials.
            caption_feat (torch.Tensor): Additional input tensor.
            caption_mask (torch.Tensor): Mask for the additional input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the attention mechanism.
        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.num_attention_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        xq = Attention.apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = Attention.apply_rotary_emb(xk, freqs_cis=freqs_cis)

        xq, xk = xq.to(dtype), xk.to(dtype)

        if self.proportional_attn:
            softmax_scale = math.sqrt(math.log(seqlen, self.base_seqlen) / self.head_dim)
        else:
            softmax_scale = math.sqrt(1 / self.head_dim)

        if self.use_flash_attn and dtype in [torch.float16, torch.bfloat16]:
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(xq, xk, xv, x_mask, seqlen)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=0.0,
                causal=False,
                softmax_scale=softmax_scale,
            )
            output = pad_input(attn_output_unpad, indices_q, bsz, seqlen)

        else:
            n_rep = self.num_attention_heads // self.num_kv_heads
            if n_rep >= 1:
                xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            output = (
                F.scaled_dot_product_attention(
                    xq.permute(0, 2, 1, 3),
                    xk.permute(0, 2, 1, 3),
                    xv.permute(0, 2, 1, 3),
                    attn_mask=x_mask.bool().view(bsz, 1, 1, seqlen).expand(-1, self.num_attention_heads, seqlen, -1),
                    scale=softmax_scale,
                )
                .permute(0, 2, 1, 3)
                .to(dtype)
            )

        if hasattr(self, "wk_cap"):
            yk = self.k_cap_norm(self.wk_cap(caption_feat)).view(bsz, -1, self.num_kv_heads, self.head_dim)
            yv = self.wv_cap(caption_feat).view(bsz, -1, self.num_kv_heads, self.head_dim)

            n_rep = self.num_attention_heads // self.num_kv_heads

            if n_rep >= 1:
                yk = yk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                yv = yv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

            output_caption = F.scaled_dot_product_attention(
                xq.permute(0, 2, 1, 3),
                yk.permute(0, 2, 1, 3),
                yv.permute(0, 2, 1, 3),
                caption_mask.view(bsz, 1, 1, -1).expand(bsz, self.num_attention_heads, seqlen, -1),
            ).permuste(0, 2, 1, 3)

            output_caption = output_caption * self.gate.tanh().view(1, 1, -1, 1)
            output = output + output_caption

        output = output.flatten(-2)

        return self.wo(output)


class FinalLayer(nn.Module):
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
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                min(hidden_size, 1024),
                hidden_size,
                bias=True,
            ),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        """
        Forward pass of the FinalLayer.

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
        layer_id: int,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        caption_dim: int,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_attention_heads
        self.attention = Attention(hidden_size, num_attention_heads, num_kv_heads, qk_norm, caption_dim)
        self.feed_forward = LuminaFeedForward(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm1 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=elementwise_affine)
        self.ffn_norm1 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=elementwise_affine)

        self.attention_norm2 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=elementwise_affine)
        self.ffn_norm2 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=elementwise_affine)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                min(hidden_size, 1024),
                4 * hidden_size,
                bias=True,
            ),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        self.attention_caption_norm = RMSNorm(caption_dim, eps=norm_eps, elementwise_affine=elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        caption_feat: torch.Tensor,
        caption_mask: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass through the LuminaNextDiTBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
        Returns:
            torch.Tensor: Output tensor after applying attention and
                feedforward layers.
        """
        if adaln_input is not None:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(
                self.attention(
                    modulate(self.attention_norm1(x), scale_msa),
                    x_mask,
                    freqs_cis,
                    self.attention_caption_norm(caption_feat),
                    caption_mask,
                )
            )
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                self.feed_forward(
                    modulate(self.ffn_norm1(x), scale_mlp),
                )
            )
        else:
            x = x + self.attention_norm2(
                self.attention(
                    self.attention_norm1(x),
                    x_mask,
                    freqs_cis,
                    self.attention_caption_norm(caption_feat),
                    caption_mask,
                )
            )
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


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
        caption_dim (`int`, *optional*, defaults to 2048):
            The dimensionality of the text embeddings. This parameter defines the size of the text representations used
            in the model.
        scale_factor (`float`, *optional*, defaults to 1.0):
            A scaling factor applied to certain parameters or layers in the model. This can be used for adjusting the
            overall scale of the model's operations.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: Optional[int] = 2,
        in_channels: Optional[int] = 4,
        hidden_size: Optional[int] = 4096,
        num_layers: Optional[int] = 32,
        num_attention_heads: Optional[int] = 32,
        num_kv_heads: Optional[int] = None,
        multiple_of: Optional[int] = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: Optional[float] = 1e-5,
        learn_sigma: Optional[bool] = True,
        qk_norm: Optional[bool] = False,
        caption_dim: Optional[int] = 2048,
        scale_factor: Optional[float] = 1.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.scale_factor = scale_factor
        self.learn_sigma = learn_sigma

        self.patch_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
            bias=True,
        )
        nn.init.xavier_uniform_(self.patch_embedder.weight)
        nn.init.constant_(self.patch_embedder.bias, 0.0)

        self.pad_token = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.pad_token, std=0.02)

        self.time_caption_embed = LuminaCombinedTimestepCaptionEmbedding(
            hidden_size=min(hidden_size, 1024), caption_dim=caption_dim
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
                    caption_dim,
                )
                for layer_id in range(num_layers)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        assert (hidden_size // num_attention_heads) % 4 == 0, "2d rope needs head dim to be divisible by 4"
        self.freqs_cis = LuminaNextDiT2DModel.precompute_freqs_cis(
            hidden_size // num_attention_heads,
            384,
            scale_factor=scale_factor,
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
        self, x: List[torch.Tensor] | torch.Tensor
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
        x: torch.Tensor,
        timestep: torch.Tensor,
        caption_feat: torch.Tensor,
        caption_mask: torch.Tensor,
        return_dict=True,
    ) -> torch.Tensor:
        """
        Forward pass of LuminaNextDiT.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            timestep (torch.Tensor): Tensor of diffusion timesteps of shape (N,).
            caption_feat (torch.Tensor): Tensor of caption features of shape (N, D).
            caption_mask (torch.Tensor): Tensor of caption masks of shape (N, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W).
        """
        x_is_tensor = isinstance(x, torch.Tensor)
        x, mask, img_size, freqs_cis = self.patchify_and_embed(x)
        freqs_cis = freqs_cis.to(x.device)

        adaln_input = self.time_caption_embed(timestep, caption_feat, caption_mask)

        caption_mask = caption_mask.bool()
        for layer in self.layers:
            x = layer(x, mask, freqs_cis, caption_feat, caption_mask, adaln_input=adaln_input)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x, img_size, return_tensor=x_is_tensor)

        if self.learn_sigma:
            if x_is_tensor:
                output, _ = x.chunk(2, dim=1)
            else:
                output = [_.chunk(2, dim=0)[0] for _ in x]

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def forward_with_cfg(
        self,
        x,
        timestep,
        caption_feat,
        caption_mask,
        cfg_scale,
        scale_factor=1.0,
        scale_watershed=1.0,
        base_seqlen: Optional[int] = None,
        proportional_attn: bool = False,
        return_dict: bool = True,
    ):
        """
        Forward pass of LuminaNextDiT, but also batches the unconditional forward pass for classifier-free guidance.

        Args:
            x:
                The input tensor, typically representing image data.
            timestep:
                A tensor representing the time steps or sequence positions.
            caption_feat:
                The caption features, which are used for conditioning the model.
            caption_mask:
                A mask for the caption features, indicating which elements are valid.
            cfg_scale:
                The classifier-free guidance scale, used to adjust the influence of the conditioning information.
            scale_factor (float, optional, defaults to 1.0):
                A scaling factor applied to certain operations or parameters within the forward pass.
            scale_watershed (float, optional, defaults to 1.0):
                A specific scaling factor used in conjunction with the main scale factor, potentially for different
                stages of processing.
            base_seqlen (Optional[int], optional):
                The base sequence length, which can be used to set the length of sequences processed by the model. If
                None, a default value is used.
            proportional_attn (bool, optional, defaults to False):
                Whether to use proportional attention mechanisms within the model, adjusting attention weights
                proportionally.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        self.freqs_cis = LuminaNextDiT2DModel.precompute_freqs_cis(
            self.hidden_size // self.num_attention_heads,
            384,
            scale_factor=scale_factor,
            scale_watershed=scale_watershed,
            timestep=timestep[0].item(),
        )

        if proportional_attn:
            assert base_seqlen is not None
            for layer in self.layers:
                layer.attention.base_seqlen = base_seqlen
                layer.attention.proportional_attn = proportional_attn
        else:
            for layer in self.layers:
                layer.attention.base_seqlen = None
                layer.attention.proportional_attn = proportional_attn

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self(combined, timestep, caption_feat, caption_mask)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        output = torch.cat([eps, rest], dim=1)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    @classmethod
    def precompute_freqs_cis(
        cls,
        dim: int,
        end: int,
        theta: float = 10000.0,
        scale_factor: float = 1.0,
        scale_watershed: float = 1.0,
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

        if timestep < scale_watershed:
            linear_factor = scale_factor
            ntk_factor = 1.0
        else:
            linear_factor = 1.0
            ntk_factor = scale_factor

        theta = theta * ntk_factor
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float().cuda() / dim)) / linear_factor

        timestep = torch.arange(end, device=freqs.device, dtype=torch.float)  # type: ignore

        freqs = torch.outer(timestep, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

        freqs_cis_h = freqs_cis.view(end, 1, dim // 4, 1).repeat(1, end, 1, 1)
        freqs_cis_w = freqs_cis.view(1, end, dim // 4, 1).repeat(end, 1, 1, 1)
        freqs_cis = torch.cat([freqs_cis_h, freqs_cis_w], dim=-1).flatten(2)

        return freqs_cis

    def parameter_count(self) -> int:
        total_params = 0

        def _recursive_count_params(module):
            nonlocal total_params
            for param in module.parameters(recurse=False):
                total_params += param.numel()
            for submodule in module.children():
                _recursive_count_params(submodule)

        _recursive_count_params(self)
        return total_params


# todo: delete it after creating hf format model
def LuminaNextDiT2DModel_2B_patch2(**kwargs):
    return LuminaNextDiT2DModel(patch_size=2, dim=2304, n_layers=24, num_attention_heads=32, **kwargs)


def LuminaNextDiT2DModel_2B_GQA_patch2(**kwargs):
    return LuminaNextDiT2DModel(patch_size=2, dim=2304, n_layers=24, num_attention_heads=32, num_kv_heads=8, **kwargs)
