# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import logging
import math
from typing import Optional, Tuple, List

from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin
from ...models.transformers.transformer_2d import Transformer2DModelOutput
from ...utils import logging

logger = logging.getLogger(__name__)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


#############################################################################
#             Embedding Layers for Timesteps and Class Labels               #
#############################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class ParallelLabelEmbedder(nn.Module):
    r"""Embeds class labels into vector representations. Also handles label
    dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
            drop_ids = drop_ids.cuda()
            drop_ids = drop_ids.to(labels.device)
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#############################################################################
#                               Core NextDiT Model                              #
#############################################################################


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        qk_norm: bool,
        y_dim: int,
    ):
        """
        Initialize the Attention module.

        Args:
            dim (int): Number of input dimensions.
            n_heads (int): Number of heads.
            n_kv_heads (Optional[int]): Number of kv heads, if using GQA.

        """
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_heads = n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        if y_dim > 0:
            self.wk_y = nn.Linear(y_dim, self.n_kv_heads * self.head_dim, bias=False)
            self.wv_y = nn.Linear(y_dim, self.n_kv_heads * self.head_dim, bias=False)
            self.gate = nn.Parameter(torch.zeros([self.n_heads]))

        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
            self.k_norm = nn.LayerNorm(self.n_kv_heads * self.head_dim)
            if y_dim > 0:
                self.ky_norm = nn.LayerNorm(self.n_kv_heads * self.head_dim)
            else:
                self.ky_norm = nn.Identity()
        else:
            self.q_norm = self.k_norm = nn.Identity()
            self.ky_norm = nn.Identity()

        # for proportional attention computation
        self.base_seqlen = None
        self.proportional_attn = False

    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        """
        Reshape frequency tensor for broadcasting it with another tensor.

        This function reshapes the frequency tensor to have the same shape as
        the target tensor 'x' for the purpose of broadcasting the frequency
        tensor during element-wise operations.

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
        Apply rotary embeddings to input tensors using the given frequency
        tensor.

        This function applies rotary embeddings to the given query 'xq' and
        key 'xk' tensors using the provided frequency tensor 'freqs_cis'. The
        input tensors are reshaped as complex numbers, and the frequency tensor
        is reshaped for broadcasting compatibility. The resulting tensors
        contain rotary embeddings and are returned as real tensors.

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
    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):

        def _get_unpad_data(attention_mask):
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
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
                query_layer.reshape(batch_size * kv_seq_len, self.n_heads, head_dim),
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
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

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
        y: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> torch.Tensor:
        """

        Args:
            x:
            x_mask:
            freqs_cis:
            y:
            y_mask:

        Returns:

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq = Attention.apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = Attention.apply_rotary_emb(xk, freqs_cis=freqs_cis)

        xq, xk = xq.to(dtype), xk.to(dtype)

        if dtype in [torch.float16, torch.bfloat16]:
            # begin var_len flash attn
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

            if self.proportional_attn:
                softmax_scale = math.sqrt(
                    math.log(seqlen, self.base_seqlen) / self.head_dim
                )
            else:
                softmax_scale = math.sqrt(1 / self.head_dim)

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
            # end var_len_flash_attn

        else:
            output = (
                F.scaled_dot_product_attention(
                    xq.permute(0, 2, 1, 3),
                    xk.permute(0, 2, 1, 3),
                    xv.permute(0, 2, 1, 3),
                    attn_mask=x_mask.bool()
                    .view(bsz, 1, 1, seqlen)
                    .expand(-1, self.n_heads, seqlen, -1),
                )
                .permute(0, 2, 1, 3)
                .to(dtype)
            )

        if hasattr(self, "wk_y"):
            # todo better flash_attn support
            yk = self.ky_norm(self.wk_y(y)).view(
                bsz, -1, self.n_kv_heads, self.head_dim
            )
            yv = self.wv_y(y).view(bsz, -1, self.n_kv_heads, self.head_dim)
            n_rep = self.n_heads // self.n_kv_heads
            if n_rep >= 1:
                yk = yk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                yv = yv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            output_y = F.scaled_dot_product_attention(
                xq.permute(0, 2, 1, 3),
                yk.permute(0, 2, 1, 3),
                yv.permute(0, 2, 1, 3),
                y_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_heads, seqlen, -1),
            ).permute(0, 2, 1, 3)
            output_y = output_y * self.gate.tanh().view(1, 1, -1, 1)
            output = output + output_y

        output = output.flatten(-2)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple
                of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden
                dimension. Defaults to None.

        Attributes:
            w1 (nn.Linear): Linear transformation for the first
                layer.
            w2 (nn.Linear): Linear transformation for the second layer.
            w3 (nn.Linear): Linear transformation for the third
                layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    # @torch.compile
    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        y_dim: int,
    ) -> None:
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            dim (int): Embedding dimension of the input features.
            n_heads (int): Number of attention heads.
            n_kv_heads (Optional[int]): Number of attention heads in key and
                value features (if using GQA), or set to None for the same as
                query.
            multiple_of (int):
            ffn_dim_multiplier (float):
            norm_eps (float):

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm, y_dim)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 6 * dim, bias=True),
        )

        self.attention_y_norm = RMSNorm(y_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        y: torch.Tensor,
        y_mask: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and
                feedforward layers.

        """
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            )

            x = x + self.attention_norm1(
                gate_msa.unsqueeze(1)
                * self.attention(
                    modulate(self.attention_norm(x), shift_msa, scale_msa),
                    x_mask,
                    freqs_cis,
                    self.attention_y_norm(y),
                    y_mask,
                )
            )
            d = x.shape[-1]
            x = x + self.ffn_norm1(
                gate_mlp.unsqueeze(1)
                * self.feed_forward(
                    modulate(self.ffn_norm(x), shift_mlp, scale_mlp).view(-1, d),
                ).view(*x.shape)
            )

        else:
            x = x + self.attention_norm1(
                self.attention(
                    self.attention_norm(x),
                    x_mask,
                    freqs_cis,
                    self.attention_y_norm(y),
                    y_mask,
                )
            )
            # for compatibility with torch.compile because the sequence length changes
            B, L, D = x.shape
            x = x.view(B * L, D)
            x = x + self.ffn_norm1(self.feed_forward(self.ffn_norm(x)))
            x = x.view(B, L, D)

        return x


class FinalLayer(nn.Module):
    """
    The final layer of NextDiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 1024), 2 * hidden_size),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class NextFlagDiffuserTransformer2DModel(ModelMixin, ConfigMixin):
    """
    Diffusion model with a Transformer backbone.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        learn_sigma: bool = True,
        qk_norm: bool = False,
        cap_feat_dim: int = 5120,
        rope_scaling_factor: float = 1.0,
        ntk_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size

        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=dim,
            bias=True,
        )
        nn.init.constant_(self.x_embedder.bias, 0.0)

        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        self.cap_embedder = nn.Sequential(
            nn.LayerNorm(cap_feat_dim),
            nn.Linear(cap_feat_dim, min(dim, 1024), bias=True),
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    cap_feat_dim,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        assert (dim // n_heads) % 4 == 0, "2d rope needs head dim to be divisible by 4"
        self.dim = dim
        self.n_heads = n_heads
        self.freqs_cis = NextDiT.precompute_freqs_cis(
            dim // n_heads,
            384,
            rope_scaling_factor=rope_scaling_factor,
            ntk_factor=ntk_factor,
        )
        self.rope_scaling_factor = rope_scaling_factor
        self.ntk_factor = ntk_factor
        # self.eol_token = nn.Parameter(torch.empty(dim))
        self.pad_token = nn.Parameter(torch.empty(dim))
        # nn.init.normal_(self.eol_token, std=0.02)
        nn.init.normal_(self.pad_token, std=0.02)

    def unpatchify(
        self, x: torch.Tensor, img_size: List[Tuple[int, int]], return_tensor=False
    ) -> List[torch.Tensor]:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
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
        self.freqs_cis = self.freqs_cis.to(x[0].device)
        if isinstance(x, torch.Tensor):
            pH = pW = self.patch_size
            B, C, H, W = x.size()
            x = (
                x.view(B, C, H // pH, pH, W // pW, pW)
                .permute(0, 2, 4, 1, 3, 5)
                .flatten(3)
            )
            x = self.x_embedder(x)
            x = x.flatten(1, 2)

            mask = torch.ones(
                x.shape[0], x.shape[1], dtype=torch.int32, device=x.device
            )
            # leave the first line for text
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
                img = (
                    img.view(C, H // pH, pH, W // pW, pW)
                    .permute(1, 3, 0, 2, 4)
                    .flatten(2)
                )
                img = self.x_embedder(img)
                img = img.flatten(0, 1)
                l_effective_seq_len.append(len(img))
                x_embed.append(img)

            max_seq_len = max(l_effective_seq_len)
            mask = torch.zeros(
                len(x), max_seq_len, dtype=torch.int32, device=x[0].device
            )
            padded_x_embed = []
            padded_freqs_cis = []
            for i, (item_embed, item_freqs_cis, item_seq_len) in enumerate(
                zip(x_embed, freqs_cis, l_effective_seq_len)
            ):
                item_embed = torch.cat(
                    [
                        item_embed,
                        self.pad_token.view(1, -1).expand(
                            max_seq_len - item_seq_len, -1
                        ),
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
        hidden_states,
        timestep=None,
        encoder_hidden_states=None,
        encoder_attn_mask=None,
        return_dict: bool = True,
        hidden_states_is_tensor: bool = False,
        unpatchify: bool = True,
    ):
        """
        Forward pass of NextDiT.
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        # 0. Check inputs
        if not unpatchify and return_dict:
            raise ValueError(
                f"Cannot both define `unpatchify`: {unpatchify} and `return_dict`: {return_dict} since when"
                f" `unpatchify` is {unpatchify} the returned output is of shape (batch_size, seq_len, hidden_dim)"
                " rather than (batch_size, num_channels, height, width)."
            )

        # 1. Patchify and Get embedding
        hidden_states, mask, img_size, freqs_cis = self.patchify_and_embed(hidden_states)
        freqs_cis = freqs_cis.to(hidden_states.device)

        # 2. Get timestep embedding
        timestep = self.t_embedder(timestep)  # (N, D)

        # 3. Get encoder hidden_state
        encoder_attn_mask_float = encoder_attn_mask.float().unsqueeze(-1)
        encoder_hidden_states_pool = (encoder_hidden_states * encoder_attn_mask_float).sum(dim=1) / encoder_attn_mask_float.sum(
            dim=1
        )
        encoder_hidden_states_pool = encoder_hidden_states_pool.to(encoder_hidden_states)
        cap_emb = self.cap_embedder(encoder_hidden_states_pool)
        encoder_attn_mask = encoder_attn_mask.bool()
    
        adaln_input = timestep + cap_emb
        # 4. Get hidden_state from transformer
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask, freqs_cis, encoder_hidden_states, encoder_attn_mask, adaln_input=adaln_input)

        hidden_states = self.final_layer(hidden_states, adaln_input)

        if unpatchify:
            hidden_states = self.unpatchify(hidden_states, img_size, return_tensor=hidden_states_is_tensor)
        else:
            output = hidden_states

        if self.learn_sigma:
            if hidden_states_is_tensor:
                hidden_states, _ = hidden_states.chunk(2, dim=1)
            else:
                hidden_states = [_.chunk(2, dim=0)[0] for _ in hidden_states]

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def forward_with_cfg(
        self,
        x,
        t,
        cap_feats,
        cap_mask,
        cfg_scale,
        rope_scaling_factor=None,
        ntk_factor=None,
        base_seqlen: Optional[int] = None,
        proportional_attn: bool = False,
    ):
        # """
        # Forward pass of NextDiT, but also batches the unconNextditional forward pass
        # for classifier-free guidance.
        # """
        # # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # print(ntk_factor, rope_scaling_factor, self.ntk_factor, self.rope_scaling_factor)
        if rope_scaling_factor is not None or ntk_factor is not None:
            rope_scaling_factor = (
                rope_scaling_factor
                if rope_scaling_factor is not None
                else self.rope_scaling_factor
            )
            ntk_factor = ntk_factor if ntk_factor is not None else self.ntk_factor
            if (
                rope_scaling_factor != self.rope_scaling_factor
                or ntk_factor != self.ntk_factor
            ):
                print(
                    f"override freqs_cis, rope_scaling {rope_scaling_factor}, ntk {ntk_factor}",
                    flush=True,
                )
                self.freqs_cis = NextDiT.precompute_freqs_cis(
                    self.dim // self.n_heads,
                    384,
                    rope_scaling_factor=rope_scaling_factor,
                    ntk_factor=ntk_factor,
                )
                self.rope_scaling_factor = rope_scaling_factor
                self.ntk_factor = ntk_factor

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
        model_out = self.forward(combined, t, cap_feats, cap_mask)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @staticmethod
    def precompute_freqs_cis(
        dim: int,
        end: int,
        theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        ntk_factor: float = 1.0,
    ):
        """
        Precompute the frequency tensor for complex exponentials (cis) with
        given dimensions.

        This function calculates a frequency tensor with complex exponentials
        using the given dimension 'dim' and the end index 'end'. The 'theta'
        parameter scales the frequencies. The returned tensor contains complex
        values in complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            end (int): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation.
                Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex
                exponentials.
        """

        theta = theta * ntk_factor

        logger.info(
            f"theta {theta} rope scaling {rope_scaling_factor} ntk {ntk_factor}"
        )
        freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float().cuda() / dim)
        )
        t = torch.arange(end, device=freqs.device, dtype=torch.float)  # type: ignore
        t = t / rope_scaling_factor
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

        freqs_cis_h = freqs_cis.view(end, 1, dim // 4, 1).repeat(1, end, 1, 1)
        freqs_cis_w = freqs_cis.view(1, end, dim // 4, 1).repeat(end, 1, 1, 1)
        freqs_cis = torch.cat([freqs_cis_h, freqs_cis_w], dim=-1).flatten(2)
        return freqs_cis

    def parameter_count(self) -> int:
        tensor_parallel_module_list = (
            nn.Linear,
            nn.Linear,
            nn.Embedding,
        )
        total_params = 0

        def _recursive_count_params(module):
            nonlocal total_params
            is_tp_module = isinstance(module, tensor_parallel_module_list)
            for param in module.parameters(recurse=False):
                total_params += param.numel()
            for submodule in module.children():
                _recursive_count_params(submodule)

        _recursive_count_params(self)
        return total_params

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)


#############################################################################
#                                 NextDiT Configs                               #
#############################################################################
def NextDiT_2B_patch2(**kwargs):
    return NextDiT(patch_size=2, dim=2304, n_layers=24, n_heads=32, **kwargs)
