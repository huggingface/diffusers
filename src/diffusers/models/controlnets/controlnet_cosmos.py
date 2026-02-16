from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin
from ...utils import BaseOutput, is_torchvision_available, logging
from ..modeling_utils import ModelMixin
from ..transformers.transformer_cosmos import (
    CosmosEmbedding,
    CosmosLearnablePositionalEmbed,
    CosmosPatchEmbed,
    CosmosRotaryPosEmbed,
    CosmosTransformerBlock,
)


if is_torchvision_available():
    from torchvision import transforms

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class CosmosControlNetOutput(BaseOutput):
    """
    Output of [`CosmosControlNetModel`].

    Args:
        control_block_samples (`list[torch.Tensor]`):
            List of control block activations to be injected into transformer blocks.
    """

    control_block_samples: List[torch.Tensor]


class CosmosControlNetModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    ControlNet for Cosmos Transfer2.5.

    This model duplicates the shared embedding modules from the transformer (patch_embed, time_embed,
    learnable_pos_embed, img_context_proj) to enable proper CPU offloading. The forward() method computes everything
    internally from raw inputs.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embed", "patch_embed_base", "time_embed"]
    _no_split_modules = ["CosmosTransformerBlock"]
    _keep_in_fp32_modules = ["learnable_pos_embed"]

    @register_to_config
    def __init__(
        self,
        n_controlnet_blocks: int = 4,
        in_channels: int = 130,
        latent_channels: int = 18,  # base latent channels (latents + condition_mask) + padding_mask
        model_channels: int = 2048,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        max_size: Tuple[int, int, int] = (128, 240, 240),
        rope_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0),
        extra_pos_embed_type: str | None = None,
        img_context_dim_in: int | None = None,
        img_context_dim_out: int = 2048,
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int = 1024,
        encoder_hidden_states_channels: int = 1024,
    ):
        super().__init__()

        self.patch_embed = CosmosPatchEmbed(in_channels, model_channels, patch_size, bias=False)

        self.patch_embed_base = CosmosPatchEmbed(latent_channels, model_channels, patch_size, bias=False)
        self.time_embed = CosmosEmbedding(model_channels, model_channels)

        self.learnable_pos_embed = None
        if extra_pos_embed_type == "learnable":
            self.learnable_pos_embed = CosmosLearnablePositionalEmbed(
                hidden_size=model_channels,
                max_size=max_size,
                patch_size=patch_size,
            )

        self.img_context_proj = None
        if img_context_dim_in is not None and img_context_dim_in > 0:
            self.img_context_proj = nn.Sequential(
                nn.Linear(img_context_dim_in, img_context_dim_out, bias=True),
                nn.GELU(),
            )

        # Cross-attention projection for text embeddings (same as transformer)
        self.crossattn_proj = None
        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, encoder_hidden_states_channels, bias=True),
                nn.GELU(),
            )

        # RoPE for both control and base latents
        self.rope = CosmosRotaryPosEmbed(
            hidden_size=attention_head_dim, max_size=max_size, patch_size=patch_size, rope_scale=rope_scale
        )

        self.control_blocks = nn.ModuleList(
            [
                CosmosTransformerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=text_embed_dim,
                    mlp_ratio=mlp_ratio,
                    adaln_lora_dim=adaln_lora_dim,
                    qk_norm="rms_norm",
                    out_bias=False,
                    img_context=img_context_dim_in is not None and img_context_dim_in > 0,
                    before_proj=(block_idx == 0),
                    after_proj=True,
                )
                for block_idx in range(n_controlnet_blocks)
            ]
        )

        self.gradient_checkpointing = False

    def _expand_conditioning_scale(self, conditioning_scale: float | list[float]) -> List[float]:
        if isinstance(conditioning_scale, list):
            scales = conditioning_scale
        else:
            scales = [conditioning_scale] * len(self.control_blocks)

        if len(scales) < len(self.control_blocks):
            logger.warning(
                "Received %d control scales, but control network defines %d blocks. "
                "Scales will be trimmed or repeated to match.",
                len(scales),
                len(self.control_blocks),
            )
            scales = (scales * len(self.control_blocks))[: len(self.control_blocks)]
        return scales

    def forward(
        self,
        controls_latents: torch.Tensor,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Union[Optional[torch.Tensor], Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
        condition_mask: torch.Tensor,
        conditioning_scale: float | list[float] = 1.0,
        padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        fps: int | None = None,
        return_dict: bool = True,
    ) -> Union[CosmosControlNetOutput, Tuple[List[torch.Tensor]]]:
        """
        Forward pass for the ControlNet.

        Args:
            controls_latents: Control signal latents [B, C, T, H, W]
            latents: Base latents from the noising process [B, C, T, H, W]
            timestep: Diffusion timestep tensor
            encoder_hidden_states: Tuple of (text_context, img_context) or text_context
            condition_mask: Conditioning mask [B, 1, T, H, W]
            conditioning_scale: Scale factor(s) for control outputs
            padding_mask: Padding mask [B, 1, H, W] or None
            attention_mask: Optional attention mask or None
            fps: Frames per second for RoPE or None
            return_dict: Whether to return a CosmosControlNetOutput or a tuple

        Returns:
            CosmosControlNetOutput or tuple of control tensors
        """
        B, C, T, H, W = controls_latents.shape

        # 1. Prepare control latents
        control_hidden_states = controls_latents
        vace_in_channels = self.config.in_channels - 1
        if control_hidden_states.shape[1] < vace_in_channels - 1:
            pad_C = vace_in_channels - 1 - control_hidden_states.shape[1]
            control_hidden_states = torch.cat(
                [
                    control_hidden_states,
                    torch.zeros(
                        (B, pad_C, T, H, W), dtype=control_hidden_states.dtype, device=control_hidden_states.device
                    ),
                ],
                dim=1,
            )

        control_hidden_states = torch.cat([control_hidden_states, torch.zeros_like(controls_latents[:, :1])], dim=1)

        padding_mask_resized = transforms.functional.resize(
            padding_mask, list(control_hidden_states.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
        )
        control_hidden_states = torch.cat(
            [control_hidden_states, padding_mask_resized.unsqueeze(2).repeat(B, 1, T, 1, 1)], dim=1
        )

        # 2. Prepare base latents (same processing as transformer.forward)
        base_hidden_states = latents
        if condition_mask is not None:
            base_hidden_states = torch.cat([base_hidden_states, condition_mask], dim=1)

        base_padding_mask = transforms.functional.resize(
            padding_mask, list(base_hidden_states.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
        )
        base_hidden_states = torch.cat(
            [base_hidden_states, base_padding_mask.unsqueeze(2).repeat(B, 1, T, 1, 1)], dim=1
        )

        # 3. Generate positional embeddings (shared for both)
        image_rotary_emb = self.rope(control_hidden_states, fps=fps)
        extra_pos_emb = self.learnable_pos_embed(control_hidden_states) if self.learnable_pos_embed else None

        # 4. Patchify control latents
        control_hidden_states = self.patch_embed(control_hidden_states)
        control_hidden_states = control_hidden_states.flatten(1, 3)

        # 5. Patchify base latents
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = T // p_t
        post_patch_height = H // p_h
        post_patch_width = W // p_w

        base_hidden_states = self.patch_embed_base(base_hidden_states)
        base_hidden_states = base_hidden_states.flatten(1, 3)

        # 6. Time embeddings
        if timestep.ndim == 1:
            temb, embedded_timestep = self.time_embed(base_hidden_states, timestep)
        elif timestep.ndim == 5:
            batch_size, _, num_frames, _, _ = latents.shape
            assert timestep.shape == (batch_size, 1, num_frames, 1, 1), (
                f"Expected timestep to have shape [B, 1, T, 1, 1], but got {timestep.shape}"
            )
            timestep_flat = timestep.flatten()
            temb, embedded_timestep = self.time_embed(base_hidden_states, timestep_flat)
            temb, embedded_timestep = (
                x.view(batch_size, post_patch_num_frames, 1, 1, -1)
                .expand(-1, -1, post_patch_height, post_patch_width, -1)
                .flatten(1, 3)
                for x in (temb, embedded_timestep)
            )
        else:
            raise ValueError(f"Expected timestep to have shape [B, 1, T, 1, 1] or [T], but got {timestep.shape}")

        # 7. Process encoder hidden states
        if isinstance(encoder_hidden_states, tuple):
            text_context, img_context = encoder_hidden_states
        else:
            text_context = encoder_hidden_states
            img_context = None

        # Apply cross-attention projection to text context
        if self.crossattn_proj is not None:
            text_context = self.crossattn_proj(text_context)

        # Apply cross-attention projection to image context (if provided)
        if img_context is not None and self.img_context_proj is not None:
            img_context = self.img_context_proj(img_context)

        # Combine text and image context into a single tuple
        if self.config.img_context_dim_in is not None and self.config.img_context_dim_in > 0:
            processed_encoder_hidden_states = (text_context, img_context)
        else:
            processed_encoder_hidden_states = text_context

        # 8. Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]

        # 9. Run control blocks
        scales = self._expand_conditioning_scale(conditioning_scale)
        result = []
        for block_idx, (block, scale) in enumerate(zip(self.control_blocks, scales)):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                control_hidden_states, control_proj = self._gradient_checkpointing_func(
                    block,
                    control_hidden_states,
                    processed_encoder_hidden_states,
                    embedded_timestep,
                    temb,
                    image_rotary_emb,
                    extra_pos_emb,
                    attention_mask,
                    None,  # controlnet_residual
                    base_hidden_states,
                    block_idx,
                )
            else:
                control_hidden_states, control_proj = block(
                    hidden_states=control_hidden_states,
                    encoder_hidden_states=processed_encoder_hidden_states,
                    embedded_timestep=embedded_timestep,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    extra_pos_emb=extra_pos_emb,
                    attention_mask=attention_mask,
                    controlnet_residual=None,
                    latents=base_hidden_states,
                    block_idx=block_idx,
                )
            result.append(control_proj * scale)

        if not return_dict:
            return (result,)

        return CosmosControlNetOutput(control_block_samples=result)
