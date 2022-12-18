from dataclasses import dataclass
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin
from ..utils import BaseOutput
from .attention import BasicTransformerBlock
from .embeddings import TimestepEmbedding, Timesteps


@dataclass
class PriorTransformerOutput(BaseOutput):
    """
    Args:
        predicted_image_embedding (`torch.FloatTensor` of shape `(batch_size, clip_embeddings_dim)`):
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
        clip_embeddings_dim (`int`, *optional*, defaults to 768): The dimension of the CLIP embeddings. Note that CLIP
            image embeddings and text embeddings are both the same dimension.
        clip_num_embeddings (`int`, *optional*, defaults to 77): The max number of clip embeddings allowed. I.e. the
            length of the prompt after it has been tokenized.
        additional_embeddings (`int`, *optional*, defaults to 4): The number of additional tokens appended to the
            projected hidden_states. The actual length of the used hidden_states is `clip_num_embeddings +
            additional_embeddings`.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        upcast_attention (`bool`, *optional*, defaults to False): In attention blocks, ensures projected query and key
            values are upcast to float32 before matrix multiplication.

    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_layers: int = 20,
        clip_embeddings_dim: int = 768,
        clip_num_embeddings=77,
        additional_embeddings=4,
        dropout: float = 0.0,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.additional_embeddings = additional_embeddings

        self.time_proj = Timesteps(inner_dim, True, 0)
        self.time_embedding = TimestepEmbedding(inner_dim, inner_dim)

        self.proj_in = nn.Linear(clip_embeddings_dim, inner_dim)

        self.text_embeddings_proj = nn.Linear(clip_embeddings_dim, inner_dim)
        self.text_encoder_hidden_states_proj = nn.Linear(clip_embeddings_dim, inner_dim)

        self.positional_embedding = nn.Parameter(
            torch.zeros(1, clip_num_embeddings + additional_embeddings, inner_dim)
        )

        # TODO - better name. I can't tell what this is
        self.prd_embedding = nn.Parameter(torch.zeros(1, 1, inner_dim))

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn="gelu",
                    attention_bias=True,
                    upcast_attention=upcast_attention,
                )
                for d in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(inner_dim)
        self.proj_to_clip_embeddings = nn.Linear(inner_dim, clip_embeddings_dim)

        causal_attention_mask = torch.full(
            [clip_num_embeddings + additional_embeddings, clip_num_embeddings + additional_embeddings], float("-inf")
        )
        causal_attention_mask.triu_(1)
        causal_attention_mask = causal_attention_mask[None, ...]
        self.register_buffer("causal_attention_mask", causal_attention_mask, persistent=False)

        self.clip_mean = nn.Parameter(torch.zeros(1, clip_embeddings_dim))
        self.clip_std = nn.Parameter(torch.zeros(1, clip_embeddings_dim))

    def forward(
        self,
        hidden_states,
        timestep: Union[torch.Tensor, float, int],
        text_embeddings: torch.FloatTensor,
        text_encoder_hidden_states: torch.FloatTensor,
        text_mask: torch.BoolTensor,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, clip_embeddings_dim)`):
                x_t, the currently predicted image embeddings.
            timestep (`torch.long`):
                Current denoising step.
            text_embeddings (`torch.FloatTensor` of shape `(batch_size, clip_embeddings_dim)`):
                Text embeddings the denoising process is conditioned on.
            text_encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, clip_num_embeddings, clip_embeddings_dim)`):
                Hidden states of the text embeddings the denoising process is conditioned on.
            text_mask (`torch.BoolTensor` of shape `(batch_size, clip_num_embeddings)`):
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

        text_embeddings = self.text_embeddings_proj(text_embeddings)
        text_encoder_hidden_states = self.text_encoder_hidden_states_proj(text_encoder_hidden_states)
        hidden_states = self.proj_in(hidden_states)
        prd_embedding = self.prd_embedding.to(hidden_states.dtype).expand(batch_size, -1, -1)
        positional_embeddings = self.positional_embedding.to(hidden_states.dtype)

        hidden_states = torch.cat(
            [
                text_encoder_hidden_states,
                text_embeddings[:, None, :],
                time_embeddings[:, None, :],
                hidden_states[:, None, :],
                prd_embedding,
            ],
            dim=1,
        )

        hidden_states = hidden_states + positional_embeddings

        text_mask = F.pad(text_mask, (0, self.additional_embeddings), value=0.0)
        attention_mask = (text_mask[:, None, :] + self.causal_attention_mask).to(hidden_states.dtype)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states[:, -1]
        predicted_image_embedding = self.proj_to_clip_embeddings(hidden_states)

        if not return_dict:
            return (predicted_image_embedding,)

        return PriorTransformerOutput(predicted_image_embedding=predicted_image_embedding)

    def post_process_latents(self, prior_latents):
        prior_latents = (prior_latents * self.clip_std) + self.clip_mean
        return prior_latents
