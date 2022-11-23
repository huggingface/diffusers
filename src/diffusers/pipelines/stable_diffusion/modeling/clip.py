#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from inspect import isfunction
from typing import Optional

from aitemplate.compiler import ops
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target

# pylint: disable=W0102

USE_CUDA = detect_target().name() == "cuda"


def get_shape(x):
    shape = [it.value() for it in x._attrs["shape"]]
    return shape


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        dtype="float16",
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q_weight = nn.Parameter(shape=[inner_dim, query_dim], dtype=dtype)
        self.to_k_weight = nn.Parameter(shape=[inner_dim, context_dim], dtype=dtype)
        self.to_v_weight = nn.Parameter(shape=[inner_dim, context_dim], dtype=dtype)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, residual=None):
        nheads = self.heads
        d = self.dim_head

        layout = "20314" if USE_CUDA else "m2n3"

        bs, seqlen, _ = get_shape(x)
        q = ops.gemm_rcr_permute(shape=(seqlen, 1, nheads), layout=layout)(
            ops.reshape()(x, [bs * seqlen, -1]), self.to_q_weight.tensor()
        )
        context = default(context, x)

        seqlen = get_shape(context)[1]
        k = ops.gemm_rcr_permute(shape=(seqlen, 1, nheads), layout=layout)(
            ops.reshape()(context, [bs * seqlen, -1]), self.to_k_weight.tensor()
        )
        v = ops.gemm_rcr_permute(shape=(seqlen, 1, nheads), layout=layout)(
            ops.reshape()(context, [bs * seqlen, -1]), self.to_v_weight.tensor()
        )

        if USE_CUDA:
            attn_op = ops.mem_eff_attention(causal=False)
            out = attn_op(
                (ops.reshape()(q, [bs, nheads, -1, d])),
                (ops.reshape()(k, [bs, nheads, -1, d])),
                (ops.reshape()(v, [bs, nheads, -1, d])),
            )
        else:
            OP = ops.bmm_softmax_bmm_permute(shape=(nheads,), scale=self.scale)
            out = OP(
                (ops.reshape()(q, [bs * nheads, -1, d])),
                (ops.reshape()(k, [bs * nheads, -1, d])),
                (ops.reshape()(v, [bs * nheads, -1, d])),
            )
        out = ops.reshape()(out, [bs, -1, nheads * d])
        proj = self.to_out(out)
        proj = ops.reshape()(proj, [bs, -1, nheads * d])
        if residual is not None:
            return proj + residual
        else:
            return proj


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, specialization="mul")
        self.gate = nn.Linear(dim_in, dim_out, specialization="fast_gelu")

    def forward(self, x):
        return self.proj(x, self.gate(x))


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(
                nn.Linear(dim, inner_dim, specialization="fast_gelu"),
            )
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x, residual=None):
        shape = ops.size()(x)
        x = self.net(x)
        x = ops.reshape()(x, shape)
        if residual is not None:
            return x + residual
        else:
            return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

        self.param = (dim, n_heads, d_head, context_dim, gated_ff, checkpoint)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x), residual=x)
        x = self.attn2(self.norm2(x), context=context, residual=x)
        x = self.ff(self.norm3(x), residual=x)
        return x


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)  # Group Norm

        self.proj_in = nn.Conv2dBias(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim
                )
                for d in range(depth)
            ]
        )

        self.proj_out = nn.Conv2dBias(
            inner_dim, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, h, w, c = get_shape(x)
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = ops.reshape()(x, [b, -1, c])
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = ops.reshape()(x, [b, h, w, c])
        x = self.proj_out(x)
        return x + x_in


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        attention_dropout=0.0,
        batch_size=1,
        seq_len=16,
        layer_norm_eps=1e-5,
        hidden_dropout_prob=0.0,
        causal=False,
        mask_seq=0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim=hidden_size,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_attention_heads,
            qkv_bias=True,
            attn_drop=attention_dropout,
            proj_drop=hidden_dropout_prob,
            has_residual=False,
            causal=causal,
            mask_seq=mask_seq,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        causal_attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = False,
        residual: Optional[Tensor] = None,
    ):
        if residual is not None:
            self_output = self.attn(hidden_states, residual)
        else:
            self_output = self.attn(hidden_states)
        return self_output


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, x):
        x1 = x * 1.702
        x1 = ops.sigmoid(x1)
        x = x * x1
        return x


class CLIPMLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer="GELU",
        drop=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(
            in_features,
            hidden_features,
        )
        self.activation_fn = QuickGELUActivation()
        self.fc2 = nn.Linear(hidden_features, out_features, specialization="add")

    def forward(self, x, res):
        shape = get_shape(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x, res)
        return ops.reshape()(x, shape)


class CLIPEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        attention_dropout=0.0,
        mlp_ratio=4.0,
        batch_size=1,
        seq_len=16,
        causal=False,
        mask_seq=0,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.self_attn = nn.MultiheadAttention(
            dim=hidden_size,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_attention_heads,
            qkv_bias=True,
            attn_drop=attention_dropout,
            proj_drop=0,
            has_residual=True,
            causal=causal,
            mask_seq=mask_seq,
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(hidden_size, int(hidden_size * mlp_ratio))
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: Tensor,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, residual)

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states, residual)

        return hidden_states


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].
    Args:
        config: CLIPConfig
    """

    def __init__(
        self,
        num_hidden_layers=12,
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=False,
        hidden_size=768,
        num_attention_heads=12,
        batch_size=1,
        seq_len=64,
        causal=False,
        mask_seq=0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CLIPEncoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    causal=causal,
                    mask_seq=mask_seq,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[Tensor] = None,
        causal_attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        encoder_states = () if output_hidden_states else None
        # all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for _, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(hidden_states)
            hidden_states = layer_outputs

        return hidden_states


class CLIPTextEmbeddings(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        vocab_size=49408,
        max_position_embeddings=77,
        dtype="float16",
    ):
        super().__init__()
        embed_dim = hidden_size

        self.token_embedding = nn.Embedding(shape=[vocab_size, embed_dim], dtype=dtype)
        self.position_embedding = nn.Embedding(
            shape=[max_position_embeddings, embed_dim], dtype=dtype
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Tensor:

        input_shape = ops.size()(input_ids)

        # [B * S]
        input_ids = ops.reshape()(input_ids, [-1])

        position_ids = ops.reshape()(position_ids, [-1])

        if inputs_embeds is None:
            inputs_embeds = ops.batch_gather()(self.token_embedding.tensor(), input_ids)

        position_embeddings = ops.batch_gather()(
            self.position_embedding.tensor(), position_ids
        )

        embeddings = inputs_embeds + position_embeddings

        embeddings = ops.reshape()(embeddings, [input_shape[0], input_shape[1], -1])

        return embeddings


class CLIPTextTransformer(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=False,
        num_hidden_layers=12,
        num_attention_heads=12,
        batch_size=1,
        seq_len=64,
        causal=False,
        mask_seq=0,
    ):
        super().__init__()
        embed_dim = hidden_size
        self.embeddings = CLIPTextEmbeddings()
        self.encoder = CLIPEncoder(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            batch_size=batch_size,
            seq_len=seq_len,
            causal=causal,
            mask_seq=mask_seq,
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
        )

        last_hidden_state = encoder_outputs
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        return last_hidden_state
