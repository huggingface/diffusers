# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LTX-2 sink-bidirectional Euler refiner used as SANA-WM stage 2.

Wraps diffusers' own ``LTX2VideoTransformer3DModel`` + ``LTX2TextConnectors``
plus a Gemma-3 text encoder. The transformer's public forward always runs the
audio stream and does not expose the streaming sink/current self-attention
mask this refiner was trained with, so we run a video-only forward in-place
with a sink/current attention split.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_utils import ModelMixin


# Sigma schedule for the 3-step distilled refiner (matches the public release).
STAGE_2_DISTILLED_SIGMA_VALUES: tuple[float, ...] = (0.909375, 0.725, 0.421875, 0.0)


class SanaWMLTX2Refiner(ModelMixin, ConfigMixin):
    r"""
    LTX-2 sink-bidirectional Euler refiner used as SANA-WM stage 2.

    Wraps the diffusers LTX-2 components (transformer + text connectors + Gemma-3
    text encoder + tokenizer). Saved on disk as a directory:

        refiner/
        ├── config.json
        ├── transformer/        # LTX2VideoTransformer3DModel
        ├── connectors/         # LTX2TextConnectors
        └── text_encoder/       # Gemma-3 (+ co-located tokenizer files)

    Args:
        text_max_sequence_length (`int`, defaults to 1024):
            Maximum tokens passed to the Gemma-3 tokenizer.
    """

    config_name = "config.json"
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(self, text_max_sequence_length: int = 1024) -> None:
        super().__init__()
        self.text_max_sequence_length = int(text_max_sequence_length)
        # Sub-modules populated by from_pretrained (or set explicitly).
        self.transformer = None
        self.connectors = None
        self.tokenizer = None
        self.text_encoder = None

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs: Any,
    ) -> SanaWMLTX2Refiner:
        # Drop standard diffusers loader kwargs we don't honor — this refiner is
        # composed of sub-models that need their own load calls.
        for k in ("device_map", "max_memory", "offload_folder", "offload_state_dict",
                  "variant", "use_safetensors", "use_flashpack", "low_cpu_mem_usage"):
            kwargs.pop(k, None)
        from ...models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel  # noqa: PLC0415
        from ..ltx2 import LTX2TextConnectors  # noqa: PLC0415
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration  # noqa: PLC0415

        root = Path(pretrained_model_name_or_path)
        cfg_path = root / cls.config_name
        cfg: dict[str, Any] = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}

        self = cls(text_max_sequence_length=int(cfg.get("text_max_sequence_length", 1024)))
        self.transformer = LTX2VideoTransformer3DModel.from_pretrained(
            root / "transformer", torch_dtype=torch_dtype
        ).eval()
        self.connectors = LTX2TextConnectors.from_pretrained(
            root / "connectors", torch_dtype=torch_dtype
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(root / "text_encoder")
        self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            root / "text_encoder", torch_dtype=torch_dtype, low_cpu_mem_usage=True
        ).eval()
        return self

    def save_pretrained(self, save_directory: str | Path) -> None:
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / self.config_name).write_text(
            json.dumps({"text_max_sequence_length": self.text_max_sequence_length}, indent=2)
        )
        if self.transformer is not None:
            self.transformer.save_pretrained(root / "transformer")
        if self.connectors is not None:
            self.connectors.save_pretrained(root / "connectors")
        if self.text_encoder is not None:
            self.text_encoder.save_pretrained(root / "text_encoder")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(root / "text_encoder")

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def refine_latents(
        self,
        sana_latent: torch.Tensor,
        prompt: str,
        *,
        fps: float,
        sink_size: int = 1,
        seed: int = 42,
        progress: bool = True,
    ) -> torch.Tensor:
        """Run the 3-step LTX-2 refiner.

        Args:
            sana_latent: ``(B, C, T, H, W)`` stage-1 latent in LTX-2 VAE space.
            prompt: Text prompt.
            fps: Frame rate (scales temporal positions).
            sink_size: Number of leading frames left unrefined (default 1).
            seed: Refiner sampling seed.
            progress: Show a tqdm progress bar.

        Returns:
            Refined latent ``(B, C, T, H, W)``. The first ``sink_size`` frames
            are the unmodified sink; the rest are refined.
        """
        if sana_latent.shape[2] <= sink_size:
            raise ValueError(f"Stage-1 latent has {sana_latent.shape[2]} frames but sink_size={sink_size}.")

        dtype = next(self.transformer.parameters()).dtype
        device = next(self.transformer.parameters()).device

        # Free transformer GPU memory while we run the text encoder.
        self.transformer.to("cpu")
        _empty_cuda_cache()
        prompt_embeds, prompt_attention_mask = self._encode_prompt(prompt, device=device, dtype=dtype)

        self.transformer.to(device)
        z = sana_latent.to(device=device, dtype=dtype)
        sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device)
        start_sigma = float(sigmas[0])

        sink = z[:, :, :sink_size].contiguous()
        current = z[:, :, sink_size:].contiguous()
        generator = torch.Generator(device=device).manual_seed(int(seed))
        eps = torch.randn(current.shape, generator=generator, device=device, dtype=dtype)
        noisy = (1.0 - start_sigma) * current + start_sigma * eps

        iterator = range(len(sigmas) - 1)
        if progress:
            iterator = tqdm(iterator, desc="refiner", unit="step")

        patch_size = self.transformer.config.patch_size
        patch_size_t = self.transformer.config.patch_size_t

        for step_index in iterator:
            sigma = sigmas[step_index]
            denoised = self._predict_current_x0(
                sink=sink,
                noisy_current=noisy,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                sigma=sigma,
                fps=fps,
                dtype=dtype,
                device=device,
            )
            noisy_tokens = _pack_latents(noisy, patch_size=patch_size, patch_size_t=patch_size_t)
            velocity = (noisy_tokens.float() - denoised.float()) / sigma.float()
            next_tokens = noisy_tokens.float() + velocity * (sigmas[step_index + 1] - sigma).float()
            noisy = _unpack_latents(
                next_tokens.to(dtype),
                num_frames=noisy.shape[2],
                height=noisy.shape[3],
                width=noisy.shape[4],
                patch_size=patch_size,
                patch_size_t=patch_size_t,
            )

        return torch.cat([sink, noisy], dim=2)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _encode_prompt(
        self, prompt: str, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        text_inputs = tokenizer(
            [prompt.strip()],
            padding="max_length",
            max_length=self.text_max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        self.text_encoder.to(device)
        text_backbone = getattr(self.text_encoder, "model", self.text_encoder)
        outputs = text_backbone(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        hidden_states = torch.stack(outputs.hidden_states, dim=-1)
        sequence_lengths = attention_mask.sum(dim=-1)
        prompt_embeds = _pack_text_embeds(
            hidden_states,
            sequence_lengths,
            device=device,
            padding_side=tokenizer.padding_side,
        ).to(dtype=dtype)

        del outputs, hidden_states
        _empty_cuda_cache()

        self.connectors.to(device)
        connector_prompt_embeds, _, connector_attention_mask = self.connectors(prompt_embeds, attention_mask)
        self.connectors.to("cpu")
        del prompt_embeds, attention_mask
        _empty_cuda_cache()

        return (
            connector_prompt_embeds.to(device=device, dtype=dtype),
            connector_attention_mask.to(device=device),
        )

    def _predict_current_x0(
        self,
        *,
        sink: torch.Tensor,
        noisy_current: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        sigma: torch.Tensor,
        fps: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        full_latent = torch.cat([sink, noisy_current], dim=2)
        batch_size, _, num_frames, height, width = full_latent.shape
        patch_size = self.transformer.config.patch_size
        patch_size_t = self.transformer.config.patch_size_t

        latent_tokens = _pack_latents(full_latent, patch_size=patch_size, patch_size_t=patch_size_t)
        n_context_tokens = _pack_latents(sink, patch_size=patch_size, patch_size_t=patch_size_t).shape[1]

        raw_timestep = torch.zeros(
            batch_size, latent_tokens.shape[1], 1, dtype=torch.float32, device=device
        )
        raw_timestep[:, n_context_tokens:, 0] = sigma.float()
        model_timestep = raw_timestep.squeeze(-1) * float(
            self.transformer.config.timestep_scale_multiplier
        )

        velocity = self._forward_video_only(
            hidden_states=latent_tokens,
            encoder_hidden_states=prompt_embeds,
            timestep=model_timestep,
            encoder_attention_mask=prompt_attention_mask,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            n_context_tokens=n_context_tokens,
        )
        denoised = latent_tokens.float() - velocity.float() * raw_timestep
        return denoised[:, n_context_tokens:, :].to(dtype)

    def _forward_video_only(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        num_frames: int,
        height: int,
        width: int,
        fps: float,
        n_context_tokens: int,
    ) -> torch.Tensor:
        transformer = self.transformer
        batch_size = hidden_states.size(0)

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        video_coords = transformer.rope.prepare_video_coords(
            batch_size, num_frames, height, width, hidden_states.device, fps=fps
        )
        video_rotary_emb = transformer.rope(video_coords, device=hidden_states.device)

        hidden_states = transformer.proj_in(hidden_states)
        temb, embedded_timestep = transformer.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        encoder_hidden_states = transformer.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

        for block in transformer.transformer_blocks:
            hidden_states = _forward_video_block(
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                video_rotary_emb=video_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                n_context_tokens=n_context_tokens,
            )

        scale_shift_values = transformer.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = transformer.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        return transformer.proj_out(hidden_states)


# -------------------------------------------------------------------------
# private helpers (block + attention + packing)
# -------------------------------------------------------------------------


def _forward_video_block(
    *,
    block: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    video_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    encoder_attention_mask: torch.Tensor | None,
    n_context_tokens: int,
) -> torch.Tensor:
    batch_size = hidden_states.size(0)

    norm_hidden_states = block.norm1(hidden_states)
    num_ada_params = block.scale_shift_table.shape[0]
    ada_values = block.scale_shift_table[None, None].to(temb.device) + temb.reshape(
        batch_size, temb.size(1), num_ada_params, -1
    )
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
    norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

    attn_hidden_states = _streaming_self_attention(
        attn=block.attn1,
        hidden_states=norm_hidden_states,
        query_rotary_emb=video_rotary_emb,
        n_context_tokens=n_context_tokens,
    )
    hidden_states = hidden_states + attn_hidden_states * gate_msa

    norm_hidden_states = block.norm2(hidden_states)
    attn_hidden_states = block.attn2(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        query_rotary_emb=None,
        attention_mask=encoder_attention_mask,
    )
    hidden_states = hidden_states + attn_hidden_states

    norm_hidden_states = block.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
    hidden_states = hidden_states + block.ff(norm_hidden_states) * gate_mlp
    return hidden_states


def _streaming_self_attention(
    *,
    attn: nn.Module,
    hidden_states: torch.Tensor,
    query_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    n_context_tokens: int,
) -> torch.Tensor:
    """LTX-2 self-attention with the SANA-WM sink/current streaming mask.

    The mask allows sink tokens to attend only sink tokens, and current tokens
    to attend everything. Splitting the query range gives the same result as
    the dense additive mask while keeping diffusers' attention kernels on the
    memory-efficient path.
    """
    sequence_length = hidden_states.shape[1]
    if n_context_tokens <= 0 or n_context_tokens >= sequence_length:
        return attn(hidden_states=hidden_states, encoder_hidden_states=None, query_rotary_emb=query_rotary_emb)

    from ...models.attention_dispatch import dispatch_attention_fn  # noqa: PLC0415
    from ...models.transformers.transformer_ltx2 import (  # noqa: PLC0415
        apply_interleaved_rotary_emb,
        apply_split_rotary_emb,
    )

    gate_logits = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None

    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    query = attn.norm_q(query)
    key = attn.norm_k(key)

    if attn.rope_type == "interleaved":
        query = apply_interleaved_rotary_emb(query, query_rotary_emb)
        key = apply_interleaved_rotary_emb(key, query_rotary_emb)
    elif attn.rope_type == "split":
        query = apply_split_rotary_emb(query, query_rotary_emb)
        key = apply_split_rotary_emb(key, query_rotary_emb)
    else:
        raise ValueError(f"Unsupported LTX-2 RoPE type: {attn.rope_type}")

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    processor = attn.processor
    backend = getattr(processor, "_attention_backend", None)
    parallel_config = getattr(processor, "_parallel_config", None)
    context_hidden_states = dispatch_attention_fn(
        query[:, :n_context_tokens],
        key[:, :n_context_tokens],
        value[:, :n_context_tokens],
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        backend=backend,
        parallel_config=parallel_config,
    )
    current_hidden_states = dispatch_attention_fn(
        query[:, n_context_tokens:],
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        backend=backend,
        parallel_config=parallel_config,
    )

    hidden_states = torch.cat([context_hidden_states, current_hidden_states], dim=1)
    hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

    if gate_logits is not None:
        hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
        gates = 2.0 * torch.sigmoid(gate_logits)
        hidden_states = hidden_states * gates.unsqueeze(-1)
        hidden_states = hidden_states.flatten(2, 3)

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


def _pack_text_embeds(
    text_hidden_states: torch.Tensor,
    sequence_lengths: torch.Tensor,
    device: str | torch.device,
    padding_side: str = "left",
    scale_factor: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    batch_size, seq_len, hidden_dim, _ = text_hidden_states.shape
    original_dtype = text_hidden_states.dtype

    token_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    elif padding_side == "left":
        start_indices = seq_len - sequence_lengths[:, None]
        mask = token_indices >= start_indices
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
    mask = mask[:, :, None, None]

    masked_text_hidden_states = text_hidden_states.masked_fill(~mask, 0.0)
    num_valid_positions = (sequence_lengths * hidden_dim).view(batch_size, 1, 1, 1)
    masked_mean = masked_text_hidden_states.sum(dim=(1, 2), keepdim=True) / (num_valid_positions + eps)

    x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)

    normalized_hidden_states = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
    normalized_hidden_states = normalized_hidden_states * scale_factor
    normalized_hidden_states = normalized_hidden_states.flatten(2)
    mask_flat = mask.squeeze(-1).expand(-1, -1, normalized_hidden_states.shape[-1])
    normalized_hidden_states = normalized_hidden_states.masked_fill(~mask_flat, 0.0)
    return normalized_hidden_states.to(dtype=original_dtype)


def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    batch_size, _, num_frames, height, width = latents.shape
    latents = latents.reshape(
        batch_size, -1,
        num_frames // patch_size_t, patch_size_t,
        height // patch_size, patch_size,
        width // patch_size, patch_size,
    )
    return latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)


def _unpack_latents(
    latents: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    return latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
