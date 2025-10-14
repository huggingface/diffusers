# Copyright 2025 Alpha-VLLM Authors and The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..attention_processor import Attention
from ..embeddings import get_2d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LuminaDiTTimestepEmbedder(nn.Module):
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
        """
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(max_period))
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LuminaDiTFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class LuminaDiTBlock(nn.Module):
    """
    A Lumina DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: Optional[int],
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        norm_eps: float,
        qk_norm: bool,
        cross_attention_dim: int,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_attention_heads

        # Self-attention on image tokens
        self.attn1 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=self.head_dim,
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            qk_norm="layer_norm" if qk_norm else None,
            bias=False,
            out_bias=False,
        )

        # Cross-attention to text
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            dim_head=self.head_dim,
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            qk_norm="layer_norm" if qk_norm else None,
            bias=False,
            out_bias=False,
        )

        # Gate for cross-attention
        self.cross_attn_gate = nn.Parameter(torch.zeros([num_attention_heads]))

        # Feed-forward network
        self.ff = LuminaDiTFeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        # Layer norms
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(cross_attention_dim, eps=norm_eps)
        self.norm_ff = RMSNorm(dim, eps=norm_eps)

        # adaLN modulation
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 6 * dim, bias=True),
        )
        nn.init.zeros_(self.adaln_modulation[1].weight)
        nn.init.zeros_(self.adaln_modulation[1].bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
        image_rotary_emb: Optional[torch.Tensor],
        adaln_input: Optional[torch.Tensor] = None,
    ):
        batch_size = hidden_states.shape[0]

        # AdaLN modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_modulation(adaln_input).chunk(6, dim=1)
        )

        # Self-attention with modulation
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = modulate(norm_hidden_states, shift_msa, scale_msa)
        
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        # Cross-attention to text
        norm_encoder_hidden_states = self.norm2(encoder_hidden_states)
        cross_attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        
        # Apply gating with tanh
        gate = self.cross_attn_gate.tanh().view(1, 1, -1, 1)
        cross_attn_output = cross_attn_output * gate
        hidden_states = hidden_states + cross_attn_output.flatten(-2)

        # Feed-forward with modulation
        norm_hidden_states = self.norm_ff(hidden_states)
        norm_hidden_states = modulate(norm_hidden_states, shift_mlp, scale_mlp)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        return hidden_states


class LuminaDiTFinalLayer(nn.Module):
    """
    The final layer of Lumina DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaln_modulation[1].weight)
        nn.init.zeros_(self.adaln_modulation[1].bias)

    def forward(self, x, c):
        shift, scale = self.adaln_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LuminaDiT2DModel(ModelMixin, ConfigMixin):
    """
    Lumina-T2I Diffusion Transformer model with a transformer backbone (DiT-Llama).
    
    Reference: https://arxiv.org/abs/2404.02905

    Parameters:
        patch_size (`int`, defaults to 2):
            The size of the patches to use in the patch embedding layer.
        in_channels (`int`, defaults to 4):
            The number of input channels.
        dim (`int`, defaults to 4096):
            The hidden dimension of the model.
        num_layers (`int`, defaults to 32):
            The number of transformer blocks.
        num_attention_heads (`int`, defaults to 32):
            The number of attention heads.
        num_kv_heads (`Optional[int]`, defaults to None):
            The number of key-value heads for grouped query attention.
        multiple_of (`int`, defaults to 256):
            For feed-forward dimension calculation.
        ffn_dim_multiplier (`Optional[float]`, defaults to None):
            Multiplier for feed-forward hidden dimension.
        norm_eps (`float`, defaults to 1e-5):
            The epsilon for normalization layers.
        learn_sigma (`bool`, defaults to True):
            Whether to learn the sigma parameter.
        qk_norm (`bool`, defaults to False):
            Whether to use query-key normalization.
        cross_attention_dim (`int`, defaults to 5120):
            The dimension of the cross-attention layers (text encoder hidden size).
        sample_size (`int`, defaults to 32):
            The size of the latent image (in patches).
        rope_scaling_factor (`float`, defaults to 1.0):
            Scaling factor for rotary position embeddings.
        ntk_factor (`float`, defaults to 1.0):
            NTK-aware scaling factor for RoPE.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 4096,
        num_layers: int = 32,
        num_attention_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        learn_sigma: bool = True,
        qk_norm: bool = False,
        cross_attention_dim: int = 5120,
        sample_size: int = 32,
        rope_scaling_factor: float = 1.0,
        ntk_factor: float = 1.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_attention_heads = num_attention_heads
        self.dim = dim

        # Patch embedding
        self.x_embedder = nn.Linear(
            patch_size * patch_size * in_channels,
            dim,
            bias=True,
        )
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0.0)

        # Timestep embedding
        self.t_embedder = LuminaDiTTimestepEmbedder(min(dim, 1024))

        # Caption embedding
        self.cap_embedder = nn.Sequential(
            nn.LayerNorm(cross_attention_dim),
            nn.Linear(cross_attention_dim, min(dim, 1024), bias=True),
        )
        nn.init.zeros_(self.cap_embedder[1].weight)
        nn.init.zeros_(self.cap_embedder[1].bias)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                LuminaDiTBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    num_kv_heads=num_kv_heads,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer
        self.final_layer = LuminaDiTFinalLayer(dim, patch_size, self.out_channels)

        # Special tokens for end-of-line and padding
        self.eol_token = nn.Parameter(torch.empty(dim))
        self.pad_token = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.eol_token, std=0.02)
        nn.init.normal_(self.pad_token, std=0.02)

        # Precompute rotary embeddings
        self.rope_scaling_factor = rope_scaling_factor
        self.ntk_factor = ntk_factor
        self.register_buffer(
            "freqs_cis",
            self.precompute_freqs_cis(
                dim // num_attention_heads,
                4096,  # Max sequence length
                rope_scaling_factor=rope_scaling_factor,
                ntk_factor=ntk_factor,
            ),
        )

    @staticmethod
    def precompute_freqs_cis(
        dim: int,
        end: int,
        theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        ntk_factor: float = 1.0,
    ):
        """
        Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
        """
        theta = theta * ntk_factor
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, dtype=torch.float32)
        t = t / rope_scaling_factor
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def patchify_and_embed(
        self, x: Union[List[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        """
        Convert images to patches and embed them.
        """
        if isinstance(x, torch.Tensor):
            pH = pW = self.patch_size
            B, C, H, W = x.shape
            x = x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 1, 3, 5).flatten(3)
            x = self.x_embedder(x)
            
            # Add end-of-line tokens
            x = torch.cat(
                [
                    x,
                    self.eol_token.view(1, 1, 1, -1).expand(B, H // pH, 1, -1),
                ],
                dim=2,
            )
            x = x.flatten(1, 2)

            mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
            return x, mask, [(H, W)] * B
        else:
            # Variable resolution batch (list of tensors)
            pH = pW = self.patch_size
            x_embed = []
            img_sizes = []
            seq_lens = []

            for img in x:
                C, H, W = img.shape
                img_sizes.append((H, W))
                img = img.view(C, H // pH, pH, W // pW, pW).permute(1, 3, 0, 2, 4).flatten(2)
                img = self.x_embedder(img)
                
                # Add end-of-line tokens
                img = torch.cat(
                    [
                        img,
                        self.eol_token.view(1, 1, -1).expand(H // pH, 1, -1),
                    ],
                    dim=1,
                )
                img = img.flatten(0, 1)
                seq_lens.append(len(img))
                x_embed.append(img)

            # Pad to max length
            max_seq_len = max(seq_lens)
            mask = torch.zeros(len(x), max_seq_len, dtype=torch.bool, device=x[0].device)
            padded_x_embed = []
            
            for i, (embed, seq_len) in enumerate(zip(x_embed, seq_lens)):
                embed = torch.cat(
                    [
                        embed,
                        self.pad_token.view(1, -1).expand(max_seq_len - seq_len, -1),
                    ],
                    dim=0,
                )
                padded_x_embed.append(embed)
                mask[i, :seq_len] = True

            x_embed = torch.stack(padded_x_embed, dim=0)
            return x_embed, mask, img_sizes

    def unpatchify(
        self, x: torch.Tensor, img_sizes: List[Tuple[int, int]], return_tensor: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Convert patches back to images.
        """
        pH = pW = self.patch_size
        
        if return_tensor:
            H, W = img_sizes[0]
            B = x.shape[0]
            L = (H // pH) * (W // pW + 1)
            x = x[:, :L].view(B, H // pH, W // pW + 1, pH, pW, self.out_channels)
            x = x[:, :, :-1]  # Remove eol tokens
            x = x.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)
            return x
        else:
            imgs = []
            for i in range(x.shape[0]):
                H, W = img_sizes[i]
                L = (H // pH) * (W // pW + 1)
                img = (
                    x[i, :L]
                    .view(H // pH, W // pW + 1, pH, pW, self.out_channels)[:, :-1, :, :, :]
                    .permute(4, 0, 2, 1, 3)
                    .flatten(3, 4)
                    .flatten(1, 2)
                )
                imgs.append(img)
            return imgs

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass of the Lumina DiT model.

        Args:
            hidden_states: Input latent image (B, C, H, W) or list of variable-size latents.
            timestep: Diffusion timesteps (B,).
            encoder_hidden_states: Text embeddings (B, seq_len, hidden_size).
            encoder_attention_mask: Attention mask for text (B, seq_len).
            return_dict: Whether to return a dict.
        """
        # Patchify and embed
        is_tensor_input = isinstance(hidden_states, torch.Tensor)
        hidden_states, mask, img_sizes = self.patchify_and_embed(hidden_states)

        # Move freqs_cis to correct device if needed
        if self.freqs_cis.device != hidden_states.device:
            self.freqs_cis = self.freqs_cis.to(hidden_states.device)

        # Time and caption embeddings
        t_emb = self.t_embedder(timestep)
        
        # Pool caption embeddings
        if encoder_attention_mask is not None:
            cap_mask_float = encoder_attention_mask.float().unsqueeze(-1)
            cap_pool = (encoder_hidden_states * cap_mask_float).sum(dim=1) / cap_mask_float.sum(dim=1)
        else:
            cap_pool = encoder_hidden_states.mean(dim=1)
        
        cap_emb = self.cap_embedder(cap_pool)
        adaln_input = t_emb + cap_emb

        # Get rotary embeddings
        image_rotary_emb = self.freqs_cis[: hidden_states.shape[1]]

        # Transformer blocks
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    image_rotary_emb,
                    adaln_input,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    image_rotary_emb,
                    adaln_input,
                )

        # Final layer
        hidden_states = self.final_layer(hidden_states, adaln_input)

        # Unpatchify
        output = self.unpatchify(hidden_states, img_sizes, return_tensor=is_tensor_input)

        # Split out sigma if learned
        if self.config.learn_sigma:
            if is_tensor_input:
                output, _ = output.chunk(2, dim=1)
            else:
                output = [out.chunk(2, dim=0)[0] for out in output]

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def forward_with_cfg(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        guidance_scale: float = 1.0,
        use_cfg: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with classifier-free guidance.
        """
        if not use_cfg or guidance_scale == 1.0:
            return self.forward(
                hidden_states,
                timestep,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )[0]

        # Concatenate conditional and unconditional
        half = hidden_states[: len(hidden_states) // 2]
        combined = torch.cat([half, half], dim=0)
        
        model_out = self.forward(
            combined,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            return_dict=False,
        )[0]

        # Apply CFG
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        
        return torch.cat([eps, rest], dim=1)

