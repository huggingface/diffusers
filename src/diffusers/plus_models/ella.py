from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from ..diffusers.models.embeddings import TimestepEmbedding, Timesteps

class ELLA(nn.Module):
    def __init__(
        self,
        time_channel=320,
        time_embed_dim=768,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
        width=768,
        layers=6,
        heads=8,
        num_latents=64,
        input_dim=2048,
    ):
        super().__init__()

        self.position = Timesteps(
            time_channel, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=time_channel,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
            out_dim=out_dim,
        )

        self.connector = PerceiverResampler(
            width=width,
            layers=layers,
            heads=heads,
            num_latents=num_latents,
            input_dim=input_dim,
            time_embedding_dim=time_embed_dim,
        )

    def forward(self, text_encode_features, timesteps):
        device = text_encode_features.device
        dtype = text_encode_features.dtype

        ori_time_feature = self.position(timesteps.view(-1)).to(device, dtype=dtype)
        ori_time_feature = (
            ori_time_feature.unsqueeze(dim=1)
            if ori_time_feature.ndim == 2
            else ori_time_feature
        )
        ori_time_feature = ori_time_feature.expand(len(text_encode_features), -1, -1)
        time_embedding = self.time_embedding(ori_time_feature)

        encoder_hidden_states = self.connector(
            text_encode_features, timestep_embedding=time_embedding
        )

        return encoder_hidden_states

class PerceiverAttentionBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, time_embedding_dim: Optional[int] = None
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_2 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
    ):
        normed_latents = self.ln_1(latents, timestep_embedding)
        latents = latents + self.attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_2(x, timestep_embedding)], dim=1),
        )
        latents = latents + self.mlp(self.ln_ff(latents, timestep_embedding))
        return latents


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        width: int = 768,
        layers: int = 6,
        heads: int = 8,
        num_latents: int = 64,
        output_dim=None,
        input_dim=None,
        time_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_latents, width))
        self.time_aware_linear = nn.Linear(
            time_embedding_dim or width, width, bias=True
        )

        if self.input_dim is not None:
            self.proj_in = nn.Linear(input_dim, width)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    width, heads, time_embedding_dim=time_embedding_dim
                )
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(width, output_dim), nn.LayerNorm(output_dim)
            )

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor = None):
        learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1)
        latents = learnable_latents + self.time_aware_linear(
            torch.nn.functional.silu(timestep_embedding)
        )
        if self.input_dim is not None:
            x = self.proj_in(x)
        for p_block in self.perceiver_blocks:
            latents = p_block(x, latents, timestep_embedding=timestep_embedding)

        if self.output_dim is not None:
            latents = self.proj_out(latents)

        return latents

class ELLAProxyUNet(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        # In order to still use the diffusers pipeline, including various workaround

        self.ella = ELLA()
        self.unet = unet
        self.config = unet.config
        self.dtype = unet.dtype
        self.device = unet.device

        self.flexible_max_length_workaround = None

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[dict[str, Any]] = None,
        added_cond_kwargs: Optional[dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        if self.flexible_max_length_workaround is not None:
            time_aware_encoder_hidden_state_list = []
            for i, max_length in enumerate(self.flexible_max_length_workaround):
                time_aware_encoder_hidden_state_list.append(
                    self.ella(encoder_hidden_states[i : i + 1, :max_length], timestep)
                )
            # No matter how many tokens are text features, the ella output must be 64 tokens.
            time_aware_encoder_hidden_states = torch.cat(
                time_aware_encoder_hidden_state_list, dim=0
            )
        else:
            time_aware_encoder_hidden_states = self.ella(
                encoder_hidden_states, timestep
            )

        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=time_aware_encoder_hidden_states,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )

