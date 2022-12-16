# TODO License

import torch
from torch import nn

from diffusers.modeling_utils import ModelMixin

from ...configuration_utils import ConfigMixin, register_to_config


class UnCLIPTextProjModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        *,
        clip_extra_context_tokens: int = 4,
        clip_embeddings_dim: int = 768,
        time_embed_dim: int,
        cross_attention_dim,
    ):
        super().__init__()

        self.learned_classifier_free_guidance_embeddings = nn.Parameter(torch.zeros(clip_embeddings_dim))

        # parameters for additional clip time embeddings
        self.text_embeddings_proj = nn.Linear(clip_embeddings_dim, time_embed_dim)
        self.clip_image_embeddings_project_to_time_embeddings = nn.Linear(clip_embeddings_dim, time_embed_dim)

        # parameters for encoder hidden states
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.clip_extra_context_tokens_proj = nn.Linear(
            clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim
        )
        self.text_encoder_hidden_states_proj = nn.Linear(clip_embeddings_dim, cross_attention_dim)
        self.text_encoder_hidden_states_norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, *, image_embeddings, text_embeddings, text_encoder_hidden_states, do_classifier_free_guidance):
        if do_classifier_free_guidance:
            # Add the classifier free guidance embeddings to the image embeddings
            image_embeddings_batch_size = image_embeddings.shape[0]
            classifier_free_guidance_embeddings = self.learned_classifier_free_guidance_embeddings.unsqueeze(0)
            classifier_free_guidance_embeddings = classifier_free_guidance_embeddings.expand(
                image_embeddings_batch_size, -1
            )
            image_embeddings = torch.cat([classifier_free_guidance_embeddings, image_embeddings], dim=0)

        # The image embeddings batch size and the text embeddings batch size are equal
        assert image_embeddings.shape[0] == text_embeddings.shape[0]

        batch_size = text_embeddings.shape[0]

        # "Specifically, we modify the architecture described in Nichol et al. (2021) by projecting and
        # adding CLIP embeddings to the existing timestep embedding, ...
        time_projected_text_embeddings = self.text_embeddings_proj(text_embeddings)
        time_projected_image_embeddings = self.clip_image_embeddings_project_to_time_embeddings(image_embeddings)
        additive_clip_time_embeddings = time_projected_image_embeddings + time_projected_text_embeddings

        # ... and by projecting CLIP embeddings into four
        # extra tokens of context that are concatenated to the sequence of outputs from the GLIDE text encoder"
        clip_extra_context_tokens = self.clip_extra_context_tokens_proj(image_embeddings)
        clip_extra_context_tokens = clip_extra_context_tokens.reshape(batch_size, -1, self.clip_extra_context_tokens)

        text_encoder_hidden_states = self.text_encoder_hidden_states_proj(text_encoder_hidden_states)
        text_encoder_hidden_states = self.text_encoder_hidden_states_norm(text_encoder_hidden_states)
        text_encoder_hidden_states = text_encoder_hidden_states.permute(0, 2, 1)
        text_encoder_hidden_states = torch.cat([clip_extra_context_tokens, text_encoder_hidden_states], dim=2)

        return text_encoder_hidden_states, additive_clip_time_embeddings
