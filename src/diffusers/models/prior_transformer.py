import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .attention import BasicTransformerBlock
from .embeddings import TimestepEmbedding, Timesteps
from .modeling_utils import ModelMixin


@dataclass
class PriorTransformerOutput(BaseOutput):
    """
    Args:
        predicted_image_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`):
            The predicted CLIP image embedding conditioned on the CLIP text embedding input.
    """

    predicted_image_embedding: torch.FloatTensor


class PriorTransformer(ModelMixin, ConfigMixin):
    """
    The prior transformer from unCLIP is used to predict CLIP image embeddings from CLIP text embeddings. Note that the
    transformer predicts the image embeddings through a denoising diffusion process.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)

    For more details, see the original paper: https://arxiv.org/abs/2204.06125

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 32): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_layers (`int`, *optional*, defaults to 20): The number of layers of Transformer blocks to use.
        embedding_dim (`int`, *optional*, defaults to 768): The dimension of the CLIP embeddings. Note that CLIP
            image embeddings and text embeddings are both the same dimension.
        num_embeddings (`int`, *optional*, defaults to 77): The max number of clip embeddings allowed. I.e. the
            length of the prompt after it has been tokenized.
        additional_embeddings (`int`, *optional*, defaults to 4): The number of additional tokens appended to the
            projected hidden_states. The actual length of the used hidden_states is `num_embeddings +
            additional_embeddings`.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.

    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_layers: int = 20,
        embedding_dim: int = 768,
        num_embeddings=77,
        additional_embeddings=4,
        dropout: float = 0.0,
        act_fn: str = "silu",
        has_pre_norm: bool = False,
        has_encoder_hidden_states_proj: bool = True,
        has_prd_embedding: bool = True,
        has_post_process: bool = True,
        time_embed_dim: Optional[int] = None,
        clip_embedding_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.additional_embeddings = additional_embeddings

        if time_embed_dim is None:
            time_embed_dim = inner_dim

        if clip_embedding_dim is None:
            clip_embedding_dim = embedding_dim

        if out_dim is None:
            out_dim = embedding_dim

        self.time_proj = Timesteps(inner_dim, True, 0)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, out_dim=inner_dim, act_fn=act_fn)

        self.proj_in = nn.Linear(embedding_dim, inner_dim)

        self.embedding_proj = nn.Linear(clip_embedding_dim, inner_dim)

        if has_encoder_hidden_states_proj:
            self.encoder_hidden_states_proj = nn.Linear(clip_embedding_dim, inner_dim)
        else:
            self.encoder_hidden_states_proj = None

        self.positional_embedding = nn.Parameter(torch.zeros(1, num_embeddings + additional_embeddings, inner_dim))

        if has_prd_embedding:
            self.prd_embedding = nn.Parameter(torch.zeros(1, 1, inner_dim))
        else:
            self.prd_embedding = None

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn="gelu",
                    attention_bias=True,
                )
                for d in range(num_layers)
            ]
        )

        if has_pre_norm:
            self.norm_in = nn.LayerNorm(inner_dim)
        else:
            self.norm_in = None

        self.norm_out = nn.LayerNorm(inner_dim)

        self.proj_to_clip_embeddings = nn.Linear(inner_dim, out_dim)

        causal_attention_mask = torch.full(
            [num_embeddings + additional_embeddings, num_embeddings + additional_embeddings], -10000.0
        )
        causal_attention_mask.triu_(1)
        causal_attention_mask = causal_attention_mask[None, ...]
        self.register_buffer("causal_attention_mask", causal_attention_mask, persistent=False)
        if has_post_process:
            self.clip_mean = nn.Parameter(torch.zeros(1, clip_embedding_dim))
            self.clip_std = nn.Parameter(torch.zeros(1, clip_embedding_dim))
        else:
            self.clip_mean = None
            self.clip_std = None

    def forward(
        self,
        hidden_states,
        timestep: Union[torch.Tensor, float, int],
        proj_embedding: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`):
                x_t, the currently predicted image embeddings.
            timestep (`torch.long`):
                Current denoising step.
            proj_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`):
                Projected embedding vector the denoising process is conditioned on.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_embeddings, embedding_dim)`):
                Hidden states of the text embeddings the denoising process is conditioned on.
            attention_mask (`torch.BoolTensor` of shape `(batch_size, num_embeddings)`):
                Text mask for the text embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.prior_transformer.PriorTransformerOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.prior_transformer.PriorTransformerOutput`] or `tuple`:
            [`~models.prior_transformer.PriorTransformerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        batch_size = hidden_states.shape[0]

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=hidden_states.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(hidden_states.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(batch_size, dtype=timesteps.dtype, device=timesteps.device)

        timesteps_projected = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might be fp16, so we need to cast here.
        timesteps_projected = timesteps_projected.to(dtype=self.dtype)
        time_embeddings = self.time_embedding(timesteps_projected)

        # Rescale the features to have unit variance
        # YiYi TO-DO: It was normalized before during encode_prompt step, move this step to pipeline
        if self.clip_mean is None:
            proj_embedding = math.sqrt(proj_embedding.shape[1]) * proj_embedding
        proj_embeddings = self.embedding_proj(proj_embedding)
        if self.encoder_hidden_states_proj is not None and encoder_hidden_states is not None:
            encoder_hidden_states = self.encoder_hidden_states_proj(encoder_hidden_states)
        elif self.encoder_hidden_states_proj is not None and encoder_hidden_states is None:
            raise ValueError("`encoder_hidden_states_proj` requires `encoder_hidden_states` to be set")

        hidden_states = self.proj_in(hidden_states)

        positional_embeddings = self.positional_embedding.to(hidden_states.dtype)

        tokens = []

        if encoder_hidden_states is not None:
            tokens.append(encoder_hidden_states)

        tokens = tokens + [
            proj_embeddings[:, None, :],
            time_embeddings[:, None, :],
            hidden_states[:, None, :] if len(hidden_states.shape) == 2 else hidden_states,
        ]

        if self.prd_embedding is not None:
            prd_embedding = self.prd_embedding.to(hidden_states.dtype).expand(batch_size, -1, -1)
            tokens.append(prd_embedding)

        hidden_states = torch.cat(
            tokens,
            dim=1,
        )

        # Allow positional_embedding to not include the `addtional_embeddings` and instead pad it with zeros for these additional tokens
        additional_embeddings = 2 + (encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0)
        if positional_embeddings.shape[1] < hidden_states.shape[1]:
            positional_embeddings = F.pad(
                positional_embeddings,
                (0, 0, additional_embeddings, self.prd_embedding.shape[1] if self.prd_embedding is not None else 0),
                value=0.0,
            )

        hidden_states = hidden_states + positional_embeddings

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = F.pad(attention_mask, (0, self.additional_embeddings), value=0.0)
            attention_mask = (attention_mask[:, None, :] + self.causal_attention_mask).to(hidden_states.dtype)
            attention_mask = attention_mask.repeat_interleave(self.config.num_attention_heads, dim=0)

        if self.norm_in is not None:
            hidden_states = self.norm_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm_out(hidden_states)

        if self.prd_embedding is not None:
            hidden_states = hidden_states[:, -1]
        else:
            hidden_states = hidden_states[:, additional_embeddings:]

        predicted_image_embedding = self.proj_to_clip_embeddings(hidden_states)

        if not return_dict:
            return (predicted_image_embedding,)

        return PriorTransformerOutput(predicted_image_embedding=predicted_image_embedding)

    def post_process_latents(self, prior_latents):
        prior_latents = (prior_latents * self.clip_std) + self.clip_mean
        return prior_latents
