import torch
import torch.nn as nn


class TextImageProjection(nn.Module):
    def __init__(
        self,
        text_embed_dim: int = 1024,
        image_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_image_text_embeds: int = 10,
    ):
        super().__init__()

        self.num_image_text_embeds = num_image_text_embeds
        self.image_embeds = nn.Linear(image_embed_dim, self.num_image_text_embeds * cross_attention_dim)
        self.text_proj = nn.Linear(text_embed_dim, cross_attention_dim)

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        batch_size = text_embeds.shape[0]

        # image
        image_text_embeds = self.image_embeds(image_embeds)
        image_text_embeds = image_text_embeds.reshape(batch_size, self.num_image_text_embeds, -1)

        # text
        text_embeds = self.text_proj(text_embeds)

        return torch.cat([image_text_embeds, text_embeds], dim=1)
    

class TextImageTimeEmbedding(nn.Module):
    def __init__(self, text_embed_dim: int = 768, image_embed_dim: int = 768, time_embed_dim: int = 1536):
        super().__init__()
        self.text_proj = nn.Linear(text_embed_dim, time_embed_dim)
        self.text_norm = nn.LayerNorm(time_embed_dim)
        self.image_proj = nn.Linear(image_embed_dim, time_embed_dim)

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        # text
        time_text_embeds = self.text_proj(text_embeds)
        time_text_embeds = self.text_norm(time_text_embeds)

        # image
        time_image_embeds = self.image_proj(image_embeds)

        return time_image_embeds + time_text_embeds
    

class ImageHintTimeEmbedding(nn.Module):
    def __init__(self, image_embed_dim: int = 768, time_embed_dim: int = 1536):
        super().__init__()
        self.image_proj = nn.Linear(image_embed_dim, time_embed_dim)
        self.image_norm = nn.LayerNorm(time_embed_dim)
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(256, 4, 3, padding=1),
        )

    def forward(self, image_embeds: torch.Tensor, hint: torch.Tensor):
        # image
        time_image_embeds = self.image_proj(image_embeds)
        time_image_embeds = self.image_norm(time_image_embeds)
        hint = self.input_hint_block(hint)
        return time_image_embeds, hint


class CombinedTimestepLabelEmbeddings(nn.Module):
    def __init__(self, num_classes, embedding_dim, class_dropout_prob=0.1):
        super().__init__()
        from .timestep import TimestepEmbedding, Timesteps
        from .others import LabelEmbedding

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.class_embedder = LabelEmbedding(num_classes, embedding_dim, class_dropout_prob)

    def forward(self, timestep, class_labels, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        class_labels = self.class_embedder(class_labels)  # (N, D)

        conditioning = timesteps_emb + class_labels  # (N, D)

        return conditioning


class CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()
        from .image_text import PixArtAlphaTextProjection
        from .timestep import TimestepEmbedding, Timesteps

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)

        pooled_projections = self.text_embedder(pooled_projection)

        conditioning = timesteps_emb + pooled_projections

        return conditioning


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """
    For PixArt-Alpha.

    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(self, embedding_dim, size_emb_dim, use_additional_conditions: bool = False):
        super().__init__()
        from .timestep import TimestepEmbedding, Timesteps

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            self.aspect_ratio_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)

    def forward(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        if self.use_additional_conditions:
            resolution_emb = self.additional_condition_proj(resolution.flatten()).to(hidden_dtype)
            resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
            aspect_ratio_emb = self.additional_condition_proj(aspect_ratio.flatten()).to(hidden_dtype)
            aspect_ratio_emb = self.aspect_ratio_embedder(aspect_ratio_emb).reshape(batch_size, -1)
            conditioning = timesteps_emb + torch.cat([resolution_emb, aspect_ratio_emb], dim=1)
        else:
            conditioning = timesteps_emb

        return conditioning
