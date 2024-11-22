import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict

from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.phi3_transformer_config import Phi3Transformer

from transformers import Phi3Config

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

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
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype=torch.float32):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=1):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PatchEmbedMR(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size: int = 2,
            in_chans: int = 4,
            embed_dim: int = 768,
            bias: bool = True,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return x
    
class OmniGenTransformer(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        transformer_config: Phi3Config,
        patch_size=2,
        in_channels=4,
        pe_interpolation: float = 1.0,
        pos_embed_max_size: int = 192,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size

        hidden_size = transformer_config.hidden_size

        self.x_embedder = PatchEmbedMR(patch_size, in_channels, hidden_size, bias=True)
        self.input_x_embedder = PatchEmbedMR(patch_size, in_channels, hidden_size, bias=True)

        self.time_token = TimestepEmbedder(hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        self.pe_interpolation = pe_interpolation
        pos_embed = get_2d_sincos_pos_embed(hidden_size, pos_embed_max_size, interpolation_scale=self.pe_interpolation, base_size=64)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=True)

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        self.llm = Phi3Transformer(config=transformer_config)
        self.llm.config.use_cache = False

    def initialize_weights(self):
        assert not hasattr(self, "llama")

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        w = self.input_x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)


        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.time_token.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_token.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels

        x = x.reshape(shape=(x.shape[0], h//self.patch_size, w//self.patch_size, self.patch_size, self.patch_size, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h, w))
        return imgs


    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        # print(top, top + height, left, left + width, spatial_pos_embed.size())
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed


    def patch_multiple_resolutions(self, latents, padding_latent=None, is_input_images:bool=False):
        if isinstance(latents, list):
            return_list = False
            if padding_latent is None:
                padding_latent = [None] * len(latents)
                return_list = True
            patched_latents, num_tokens, shapes = [], [], []
            for latent, padding in zip(latents, padding_latent):
                height, width = latent.shape[-2:]
                if is_input_images:
                    latent = self.input_x_embedder(latent)
                else:
                    latent = self.x_embedder(latent)
                pos_embed = self.cropped_pos_embed(height, width)    
                latent = latent + pos_embed
                if padding is not None:
                    latent = torch.cat([latent, padding], dim=-2)
                patched_latents.append(latent)

                num_tokens.append(pos_embed.size(1))
                shapes.append([height, width])
            if not return_list:
                latents = torch.cat(patched_latents, dim=0)
            else:
                latents = patched_latents
        else:
            height, width = latents.shape[-2:]
            if is_input_images:
                latents = self.input_x_embedder(latents)
            else:
                latents = self.x_embedder(latents)
            pos_embed = self.cropped_pos_embed(height, width)  
            latents = latents + pos_embed
            num_tokens = latents.size(1)
            shapes = [height, width]
        return latents, num_tokens, shapes

    
    def forward(self, x, timestep, input_ids, input_img_latents, input_image_sizes, attention_mask, position_ids, padding_latent=None, past_key_values=None, return_past_key_values=True, offload_model:bool=False):
        """
        
        """
        input_is_list = isinstance(x, list)
        x, num_tokens, shapes = self.patch_multiple_resolutions(x, padding_latent)
        time_token = self.time_token(timestep, dtype=x[0].dtype).unsqueeze(1)   
        
        if input_img_latents is not None:
            input_latents, _, _ = self.patch_multiple_resolutions(input_img_latents, is_input_images=True)
        if input_ids is not None:
            condition_embeds = self.llm.embed_tokens(input_ids).clone()
            input_img_inx = 0
            for b_inx in input_image_sizes.keys():
                for start_inx, end_inx in input_image_sizes[b_inx]:
                    condition_embeds[b_inx, start_inx: end_inx] = input_latents[input_img_inx]
                    input_img_inx += 1
            if input_img_latents is not None:
                assert input_img_inx == len(input_latents) 

            input_emb = torch.cat([condition_embeds, time_token, x], dim=1)
        else:
            input_emb = torch.cat([time_token, x], dim=1)
        output = self.llm(inputs_embeds=input_emb, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, offload_model=offload_model)
        output, past_key_values = output.last_hidden_state, output.past_key_values
        if input_is_list:
            image_embedding = output[:, -max(num_tokens):]
            time_emb = self.t_embedder(timestep, dtype=x.dtype)
            x = self.final_layer(image_embedding, time_emb)
            latents = []
            for i in range(x.size(0)):
                latent = x[i:i+1, :num_tokens[i]]
                latent = self.unpatchify(latent, shapes[i][0], shapes[i][1])
                latents.append(latent)
        else:
            image_embedding = output[:, -num_tokens:]
            time_emb = self.t_embedder(timestep, dtype=x.dtype)
            x = self.final_layer(image_embedding, time_emb)
            latents = self.unpatchify(x, shapes[0], shapes[1])

        if return_past_key_values:
            return latents, past_key_values
        return latents

    @torch.no_grad()
    def forward_with_cfg(self, x, timestep, input_ids, input_img_latents, input_image_sizes, attention_mask, position_ids, cfg_scale, use_img_cfg, img_cfg_scale, past_key_values, use_kv_cache, offload_model):      
        self.llm.config.use_cache = use_kv_cache
        model_out, past_key_values = self.forward(x, timestep, input_ids, input_img_latents, input_image_sizes, attention_mask, position_ids, past_key_values=past_key_values, return_past_key_values=True, offload_model=offload_model)
        if use_img_cfg:
            cond, uncond, img_cond = torch.split(model_out, len(model_out) // 3, dim=0)
            cond = uncond + img_cfg_scale * (img_cond - uncond) + cfg_scale * (cond - img_cond)
            model_out = [cond, cond, cond]
        else:
            cond, uncond = torch.split(model_out, len(model_out) // 2, dim=0)
            cond = uncond + cfg_scale * (cond - uncond)
            model_out = [cond, cond]
        
        return torch.cat(model_out, dim=0), past_key_values


    @torch.no_grad()
    def forward_with_separate_cfg(self, x, timestep, input_ids, input_img_latents, input_image_sizes, attention_mask, position_ids, cfg_scale, use_img_cfg, img_cfg_scale, past_key_values, use_kv_cache, offload_model):
        self.llm.config.use_cache = use_kv_cache
        if past_key_values is None:
            past_key_values = [None] * len(attention_mask)

        x = torch.split(x, len(x) // len(attention_mask), dim=0)
        timestep = timestep.to(x[0].dtype)
        timestep = torch.split(timestep, len(timestep) // len(input_ids), dim=0)

        model_out, pask_key_values = [], []
        for i in range(len(input_ids)):
            temp_out, temp_pask_key_values = self.forward(x[i], timestep[i], input_ids[i], input_img_latents[i], input_image_sizes[i], attention_mask[i], position_ids[i], past_key_values=past_key_values[i], return_past_key_values=True, offload_model=offload_model)
            model_out.append(temp_out)
            pask_key_values.append(temp_pask_key_values)

        if len(model_out) == 3:
            cond, uncond, img_cond = model_out
            cond = uncond + img_cfg_scale * (img_cond - uncond) + cfg_scale * (cond - img_cond)
            model_out = [cond, cond, cond]
        elif len(model_out) == 2:
            cond, uncond = model_out
            cond = uncond + cfg_scale * (cond - uncond)
            model_out = [cond, cond]
        else:
            return model_out[0]
        
        return torch.cat(model_out, dim=0), pask_key_values
