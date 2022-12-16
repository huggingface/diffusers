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
    predicted_image_embedding: torch.FloatTensor


class PriorTransformer(ModelMixin, ConfigMixin):
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

        causal_mask = torch.full(
            [clip_num_embeddings + additional_embeddings, clip_num_embeddings + additional_embeddings], float("-inf")
        )
        causal_mask.triu_(1)
        causal_mask = causal_mask[None, ...]
        self.register_buffer("causal_mask", causal_mask, persistent=False)

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
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
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
        mask = (text_mask[:, None, :] + self.causal_mask).to(hidden_states.dtype)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, mask=mask)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states[:, -1]
        predicted_image_embedding = self.proj_to_clip_embeddings(hidden_states)

        if not return_dict:
            return (predicted_image_embedding,)

        return PriorTransformerOutput(predicted_image_embedding=predicted_image_embedding)

    def post_process_latents(self, prior_latents):
        prior_latents = (prior_latents * self.clip_std) + self.clip_mean
        return prior_latents
