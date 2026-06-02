# Copyright 2025 The HuggingFace Team and SANA-WM Authors. All rights reserved.
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

"""LTX-2 chunk-causal AR refiner used as SANA-WM stage 2.

Wraps diffusers' own ``LTX2VideoTransformer3DModel`` + ``LTX2TextConnectors``
plus a Gemma-3 text encoder. The transformer's public forward always runs the
audio stream and does not expose the streaming sink/current self-attention
mask this refiner was trained with, so we run a video-only forward in-place
with a sink/current attention split.

Two refinement modes are supported:

* **AR / chunk-causal** (``block_size=3``, ``kv_max_frames=11`` — canonical):
  processes ``block_size`` latent frames at a time over a sliding window of
  ``[source_sink + recent_history + active_block]`` K/V. The model was trained
  with this contract; per-block compute is bounded by the window size so total
  refinement cost scales linearly with video length.
* **Single-shot** (``block_size=None``): denoises all current frames jointly
  in one O(T^2) attention pass. Out-of-distribution for the model and only
  kept around as a debugging fallback.
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
        for k in (
            "device_map",
            "max_memory",
            "offload_folder",
            "offload_state_dict",
            "variant",
            "use_safetensors",
            "use_flashpack",
            "low_cpu_mem_usage",
        ):
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
        self.connectors = LTX2TextConnectors.from_pretrained(root / "connectors", torch_dtype=torch_dtype).eval()
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
        block_size: int | None = 3,
        kv_max_frames: int = 11,
        sigmas: tuple[float, ...] = STAGE_2_DISTILLED_SIGMA_VALUES,
    ) -> torch.Tensor:
        """Run the LTX-2 refiner and return refined VAE latents.

        Defaults to the canonical chunk-causal AR recipe (``block_size=3``,
        ``kv_max_frames=11``): a sliding window of
        ``[source_sink + recent_history + active_block]`` K/V is fed to the
        transformer one block at a time. The model was trained on this contract
        and the per-block compute is bounded, so total refinement cost scales
        linearly with video length. Pass ``block_size=None`` to fall back to
        the legacy single-shot path (``O(T^2)``, OOD for the model — only kept
        for debugging).

        Args:
            sana_latent: ``(B, C, F, H, W)`` stage-1 latent.
            prompt: text prompt.
            fps: video frame rate (drives LTX-2 RoPE temporal scaling).
            sink_size: how many leading raw ``z_sana`` frames to anchor as the
                attention sink (canonical: 1).
            seed: noise seed for the FM endpoint.
            progress: show a tqdm bar.
            block_size: latent frames per AR block (canonical: 3). Set to
                ``None`` to disable AR mode.
            kv_max_frames: maximum context+active frames retained in the
                sliding window when AR mode is active (canonical: 11 =
                1 sink + 10 recent).
            sigmas: descending Euler schedule terminating at 0.0 (canonical
                3-step distilled: ``(0.909375, 0.725, 0.421875, 0.0)``).
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
        sigmas_t = torch.tensor(sigmas, dtype=torch.float32, device=device)
        start_sigma = float(sigmas_t[0])

        if block_size is not None:
            return self._refine_latents_ar(
                z=z,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                fps=fps,
                sigmas=sigmas_t,
                source_sink_frames=int(sink_size),
                block_size=int(block_size),
                kv_max_frames=int(kv_max_frames),
                seed=int(seed),
                progress=bool(progress),
                dtype=dtype,
                device=device,
            )

        sink = z[:, :, :sink_size].contiguous()
        current = z[:, :, sink_size:].contiguous()
        generator = torch.Generator(device=device).manual_seed(int(seed))
        eps = torch.randn(current.shape, generator=generator, device=device, dtype=dtype)
        noisy = (1.0 - start_sigma) * current + start_sigma * eps

        iterator = range(len(sigmas_t) - 1)
        if progress:
            iterator = tqdm(iterator, desc="refiner", unit="step")

        patch_size = self.transformer.config.patch_size
        patch_size_t = self.transformer.config.patch_size_t

        for step_index in iterator:
            sigma = sigmas_t[step_index]
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
            next_tokens = noisy_tokens.float() + velocity * (sigmas_t[step_index + 1] - sigma).float()
            noisy = _unpack_latents(
                next_tokens.to(dtype),
                num_frames=noisy.shape[2],
                height=noisy.shape[3],
                width=noisy.shape[4],
                patch_size=patch_size,
                patch_size_t=patch_size_t,
            )

        return torch.cat([sink, noisy], dim=2)

    @torch.inference_mode()
    def _refine_latents_ar(
        self,
        *,
        z: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        fps: float,
        sigmas: torch.Tensor,
        source_sink_frames: int,
        block_size: int,
        kv_max_frames: int,
        seed: int,
        progress: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Chunk-causal AR refinement — thin wrapper around ``_RefinerChunkRunner``.

        Implements the canonical ``rf_shifted_sink`` KV-cache contract end-to-end:

        1. Pre-capture **pre-RoPE** sink K/V from raw ``z_sana[:source_sink_frames]``
           at σ=0. The sink frames themselves are **never refined** — they sit
           unchanged in the output volume.
        2. AR blocks cover frames ``[source_sink_frames, T_full)`` in
           ``block_size``-frame chunks. For each block:
           - Initialize ``x_t = (1-σ₀)·z_sana_block + σ₀·ε`` (single eps per block).
           - 3-step deterministic Euler. Each step injects the per-layer prefix
             ``{sink_k_pre, sink_v, sink_pe, history_k, history_v}`` where
             ``sink_pe`` is rebuilt at ``sink_rope_offset = active_start -
             history_frames - source_sink_frames`` so the sink slides to sit
             immediately before the bounded working cache.
           - Capture **post-RoPE** K/V from the refined block under the same
             prefix; append to ``history_kv_post`` and trim to
             ``kv_max_frames - source_sink_frames``.

        The returned tensor has the same shape ``(B, C, T_full, H, W)`` as
        ``z``; the first ``source_sink_frames`` slots carry the raw sink
        latents unchanged, the rest carry the refined output.
        """
        runner = _RefinerChunkRunner(
            self,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            fps=fps,
            sigmas=sigmas,
            source_sink_frames=int(source_sink_frames),
            block_size=int(block_size),
            kv_max_frames=int(kv_max_frames),
            seed=int(seed),
            spatial_shape=(int(z.shape[3]), int(z.shape[4])),
            dtype=dtype,
            device=device,
        )

        T_full = z.shape[2]
        sink_size = int(source_sink_frames)
        # Output keeps the raw sink prefix verbatim; AR blocks fill frames
        # [sink_size, T_full).
        output = z.clone()
        n_active = max(T_full - sink_size, 0)
        n_blocks = (n_active + block_size - 1) // block_size if n_active > 0 else 0
        iterator = range(n_blocks)
        if progress:
            iterator = tqdm(iterator, desc="refiner-ar", unit="block")

        for block_idx in iterator:
            block_start = sink_size + block_idx * block_size
            block_end = min(block_start + block_size, T_full)
            clean_block = z[:, :, block_start:block_end]
            refined = runner.refine_block(
                block_idx=block_idx,
                clean_block=clean_block,
                block_start=block_start,
                block_end=block_end,
                sink_seed_frames=(z[:, :, :sink_size] if block_idx == 0 else None),
            )
            output[:, :, block_start:block_end] = refined

        return output

    def _predict_x0_active_block(
        self,
        *,
        active: torch.Tensor,
        active_positions: list[int],
        sigma_cur: float,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        fps: float,
        kv_prefix_per_layer: list[dict[str, object]] | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Forward through the transformer on the active block only and return x0.

        The active block's Q attends to ``[prefix, current]`` K/V via the
        ``_tf_kv_prefix`` hook on every self-attention block. All active tokens
        carry the same ``sigma_cur``.
        """
        latent_tokens = _pack_latents(
            active,
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        )
        batch_size, seq_len, _ = latent_tokens.shape
        timestep_scalar = float(sigma_cur) * float(self.transformer.config.timestep_scale_multiplier)
        model_timestep = torch.full((batch_size, seq_len), timestep_scalar, dtype=torch.float32, device=device)

        video_rotary_emb = _build_rotary_emb_for_absolute_positions(
            transformer=self.transformer,
            batch_size=batch_size,
            frame_positions=active_positions,
            height=int(active.shape[3]),
            width=int(active.shape[4]),
            device=device,
            fps=float(fps),
        )

        _set_kv_prefix_on_blocks(self.transformer, kv_prefix_per_layer)
        try:
            velocity = self._forward_video_only_with_rope(
                hidden_states=latent_tokens,
                encoder_hidden_states=prompt_embeds,
                timestep=model_timestep,
                encoder_attention_mask=prompt_attention_mask,
                video_rotary_emb=video_rotary_emb,
                n_context_tokens=0,
            )
        finally:
            _clear_kv_prefix_on_blocks(self.transformer)

        # FM x0 prediction: x_t - σ_cur · v.
        raw_sigma = torch.full((batch_size, seq_len, 1), float(sigma_cur), dtype=torch.float32, device=device)
        denoised_tokens = latent_tokens.float() - velocity.float() * raw_sigma
        return _unpack_latents(
            denoised_tokens.to(dtype),
            num_frames=int(active.shape[2]),
            height=int(active.shape[3]),
            width=int(active.shape[4]),
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        )

    @torch.inference_mode()
    def _capture_block_kv(
        self,
        *,
        clean_block: torch.Tensor,
        frame_positions: list[int],
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        fps: float,
        capture_mode: str,
        kv_prefix_per_layer: list[dict[str, object]] | None,
        device: torch.device,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Run one forward at σ=0 with capture hooks; return per-layer (K, V).

        ``capture_mode='pre_rope'`` saves PRE-RoPE K/V (so a future window can
        re-RoPE the sink to its shifted offset). ``capture_mode='post_rope'``
        saves POST-RoPE K/V (ready to concatenate directly into the next
        window's prefix).
        """
        latent_tokens = _pack_latents(
            clean_block,
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        )
        batch_size, seq_len, _ = latent_tokens.shape
        model_timestep = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)

        video_rotary_emb = _build_rotary_emb_for_absolute_positions(
            transformer=self.transformer,
            batch_size=batch_size,
            frame_positions=frame_positions,
            height=int(clean_block.shape[3]),
            width=int(clean_block.shape[4]),
            device=device,
            fps=float(fps),
        )

        _set_kv_prefix_on_blocks(self.transformer, kv_prefix_per_layer)
        _set_capture_flag_on_blocks(self.transformer, capture_mode, enable=True)
        try:
            _ = self._forward_video_only_with_rope(
                hidden_states=latent_tokens,
                encoder_hidden_states=prompt_embeds,
                timestep=model_timestep,
                encoder_attention_mask=prompt_attention_mask,
                video_rotary_emb=video_rotary_emb,
                n_context_tokens=0,
            )
        finally:
            _set_capture_flag_on_blocks(self.transformer, capture_mode, enable=False)
            _clear_kv_prefix_on_blocks(self.transformer)

        return _collect_captured_kv_from_blocks(self.transformer, capture_mode)

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
        outputs = text_backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states, dim=-1)
        sequence_lengths = attention_mask.sum(dim=-1)
        prompt_embeds = _pack_text_embeds(
            hidden_states,
            sequence_lengths,
            device=device,
            padding_side=tokenizer.padding_side,
        ).to(dtype=dtype)

        # Release the text encoder once we have the prompt embeds — otherwise it
        # stays resident on GPU through the entire (much longer) AR refinement.
        self.text_encoder.to("cpu")
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

        raw_timestep = torch.zeros(batch_size, latent_tokens.shape[1], 1, dtype=torch.float32, device=device)
        raw_timestep[:, n_context_tokens:, 0] = sigma.float()
        model_timestep = raw_timestep.squeeze(-1) * float(self.transformer.config.timestep_scale_multiplier)

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

    def _forward_video_only_with_rope(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        video_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        n_context_tokens: int,
    ) -> torch.Tensor:
        """Shared body of ``_forward_video_only`` that takes a pre-built RoPE.

        Used by the AR refinement path where each block forward needs custom
        per-frame absolute positions in the source video.
        """
        transformer = self.transformer
        batch_size = hidden_states.size(0)

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

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


class _RefinerChunkRunner:
    """Stateful per-AR-block driver for :class:`SanaWMLTX2Refiner`.

    Owns the rolling KV state that the chunk-causal AR recipe accumulates as
    refiner blocks complete:

    * ``_sink_kv_pre``: per-layer pre-RoPE K/V captured from the first
      ``source_sink_frames`` raw stage-1 latents at σ=0. Lazily filled on the
      first call to :meth:`refine_block`.
    * ``_history_kv_post``: per-layer post-RoPE K/V of every refined block
      already produced, trimmed to ``kv_max_frames - source_sink_frames``
      frames so the sliding window stays bounded.
    * ``_history_frames``: number of frames currently in ``_history_kv_post``.
    """

    def __init__(
        self,
        refiner: SanaWMLTX2Refiner,
        *,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        fps: float,
        sigmas: torch.Tensor,
        source_sink_frames: int,
        block_size: int,
        kv_max_frames: int,
        seed: int,
        spatial_shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._refiner = refiner
        self._prompt_embeds = prompt_embeds
        self._prompt_attention_mask = prompt_attention_mask
        self._fps = float(fps)
        self._sigmas = sigmas
        self._sigma_max = float(sigmas[0])
        self._n_steps = int(sigmas.numel() - 1)
        self._source_sink_frames = int(source_sink_frames)
        self._block_size = int(block_size)
        self._kv_max_frames = int(kv_max_frames)
        self._max_history_frames = int(kv_max_frames) - int(source_sink_frames)
        self._device = device
        self._dtype = dtype
        self._generator = torch.Generator(device=self._device).manual_seed(int(seed))

        transformer = refiner.transformer
        self._n_layers = len(transformer.transformer_blocks)
        H, W = spatial_shape
        self._H, self._W = int(H), int(W)
        self._tokens_per_frame = (
            int(H // transformer.config.patch_size)
            * int(W // transformer.config.patch_size)
            * int(transformer.config.patch_size_t)
        )

        self._sink_kv_pre: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._history_kv_post: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self._n_layers
        self._history_frames: int = 0

    @torch.inference_mode()
    def refine_block(
        self,
        *,
        block_idx: int,
        clean_block: torch.Tensor,
        block_start: int,
        block_end: int,
        sink_seed_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Refine one AR block; advance internal KV state.

        Args:
            block_idx: 0-based block index in the AR schedule.
            clean_block: ``(B, C, active_len, H, W)`` clean stage-1 latents
                covering frames ``[block_start, block_end)``.
            block_start: absolute latent-frame index of the active block's
                first frame (drives the ``rf_shifted_sink`` RoPE offset).
                Must be >= ``source_sink_frames``.
            block_end: absolute latent-frame index just past the active block.
            sink_seed_frames: ``(B, C, source_sink_frames, H, W)`` raw sink
                latents used once on the first call to pre-capture the
                pre-RoPE sink K/V at ``sigma=0`` with frame positions
                ``[0, source_sink_frames)``.
        """
        refiner = self._refiner
        device = self._device
        B = int(clean_block.shape[0])
        active_len = block_end - block_start
        if block_start < self._source_sink_frames:
            raise ValueError(
                f"block_start={block_start} overlaps the source sink "
                f"(source_sink_frames={self._source_sink_frames})."
            )

        # 1) On the first call: pre-capture PRE-RoPE sink K/V from the supplied
        # raw sink latents at sigma=0 with absolute positions [0, sink_size).
        if self._sink_kv_pre is None:
            if sink_seed_frames is None:
                raise ValueError("First refine_block call requires sink_seed_frames (raw stage-1 sink latents).")
            if sink_seed_frames.shape[2] != self._source_sink_frames:
                raise ValueError(
                    f"sink_seed_frames has {sink_seed_frames.shape[2]} frames "
                    f"but source_sink_frames={self._source_sink_frames}."
                )
            source_sink = sink_seed_frames.contiguous()
            self._sink_kv_pre = refiner._capture_block_kv(
                clean_block=source_sink,
                frame_positions=list(range(self._source_sink_frames)),
                prompt_embeds=self._prompt_embeds,
                prompt_attention_mask=self._prompt_attention_mask,
                fps=self._fps,
                capture_mode="pre_rope",
                kv_prefix_per_layer=None,
                device=device,
            )

        # 2) Build per-window kv_prefix dict per layer.
        sink_rope_offset = block_start - self._history_frames - self._source_sink_frames
        sink_pe = _build_rotary_emb_for_absolute_positions(
            transformer=refiner.transformer,
            batch_size=B,
            frame_positions=list(range(sink_rope_offset, sink_rope_offset + self._source_sink_frames)),
            height=self._H,
            width=self._W,
            device=device,
            fps=self._fps,
        )
        kv_prefix_per_layer: list[dict[str, object]] = []
        for layer_idx in range(self._n_layers):
            hk = self._history_kv_post[layer_idx]
            kv_prefix_per_layer.append(
                {
                    "mode": "rf_shifted_sink",
                    "sink_k_pre": self._sink_kv_pre[layer_idx][0],
                    "sink_v": self._sink_kv_pre[layer_idx][1],
                    "sink_pe": sink_pe,
                    "history_k": (hk[0] if hk is not None else None),
                    "history_v": (hk[1] if hk is not None else None),
                }
            )

        # 3) FM endpoint at sigma=sigma0: single epsilon per block.
        eps = torch.randn(clean_block.shape, generator=self._generator, device=device, dtype=self._dtype)
        x_t = ((1.0 - self._sigma_max) * clean_block.float() + self._sigma_max * eps.float()).to(self._dtype)

        active_positions = list(range(int(block_start), int(block_end)))
        for level in range(self._n_steps):
            sigma_cur = float(self._sigmas[level].item())
            sigma_next = float(self._sigmas[level + 1].item())
            pred_x0 = refiner._predict_x0_active_block(
                active=x_t,
                active_positions=active_positions,
                sigma_cur=sigma_cur,
                prompt_embeds=self._prompt_embeds,
                prompt_attention_mask=self._prompt_attention_mask,
                fps=self._fps,
                kv_prefix_per_layer=kv_prefix_per_layer,
                dtype=self._dtype,
                device=device,
            )
            if sigma_cur <= 1.0e-6:
                x_t = pred_x0.to(self._dtype)
            else:
                ratio = sigma_next / sigma_cur
                x_t = (ratio * x_t.float() + (1.0 - ratio) * pred_x0.float()).to(self._dtype)

        # 4) Capture POST-RoPE K/V for this refined block under the same prefix.
        block_kv_post = refiner._capture_block_kv(
            clean_block=x_t,
            frame_positions=active_positions,
            prompt_embeds=self._prompt_embeds,
            prompt_attention_mask=self._prompt_attention_mask,
            fps=self._fps,
            capture_mode="post_rope",
            kv_prefix_per_layer=kv_prefix_per_layer,
            device=device,
        )
        for layer_idx in range(self._n_layers):
            new_k, new_v = block_kv_post[layer_idx]
            old = self._history_kv_post[layer_idx]
            if old is None:
                self._history_kv_post[layer_idx] = (new_k, new_v)
            else:
                self._history_kv_post[layer_idx] = (
                    torch.cat([old[0], new_k], dim=1),
                    torch.cat([old[1], new_v], dim=1),
                )
        self._history_frames += active_len

        if self._max_history_frames > 0 and self._history_frames > self._max_history_frames:
            keep_tokens = self._max_history_frames * self._tokens_per_frame
            for layer_idx in range(self._n_layers):
                hk = self._history_kv_post[layer_idx]
                if hk is not None:
                    self._history_kv_post[layer_idx] = (hk[0][:, -keep_tokens:], hk[1][:, -keep_tokens:])
            self._history_frames = self._max_history_frames

        return x_t


# -------------------------------------------------------------------------
# private helpers (block + attention + packing)
# -------------------------------------------------------------------------


def _build_rotary_emb_for_absolute_positions(
    *,
    transformer: nn.Module,
    batch_size: int,
    frame_positions: list[int],
    height: int,
    width: int,
    device: torch.device,
    fps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reimplement ``LTX2VideoRotaryPosEmbed.prepare_video_coords`` with explicit per-frame positions.

    The default helper assumes contiguous ``torch.arange(num_frames)`` which is
    fine for bidirectional inference; the sliding-window AR refiner needs to
    keep each frame's absolute index in the source video so RoPE captures the
    correct temporal phase across the sink + recent + active window.
    """
    rope = transformer.rope
    patch_size_t = int(rope.patch_size_t)
    patch_size = int(rope.patch_size)
    f_positions = torch.tensor(frame_positions, dtype=torch.float32, device=device)
    if patch_size_t > 1:
        # Each patch covers ``patch_size_t`` latent frames; pick the start of each patch.
        f_positions = f_positions[::patch_size_t]
    grid_h = torch.arange(start=0, end=height, step=patch_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(start=0, end=width, step=patch_size, dtype=torch.float32, device=device)
    grid = torch.meshgrid(f_positions, grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0)

    patch_size_delta = torch.tensor((patch_size_t, patch_size, patch_size), dtype=grid.dtype, device=device)
    patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)
    latent_coords = torch.stack([grid, patch_ends], dim=-1)
    latent_coords = latent_coords.flatten(1, 3).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    scale_tensor = torch.tensor(rope.scale_factors, device=device)
    broadcast_shape = [1] * latent_coords.ndim
    broadcast_shape[1] = -1
    pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)
    pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + rope.causal_offset - rope.scale_factors[0]).clamp(min=0)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / float(fps)
    return rope(pixel_coords, device=device)


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
    """LTX-2 self-attention with sink/current streaming mask + AR KV-cache hooks.

    Two modes layered on top of vanilla diffusers self-attention, selected by
    ``n_context_tokens`` and per-block hook attributes (set by the AR refiner):

    * ``n_context_tokens > 0`` (legacy single-shot path): sink queries attend
      sink only, current queries attend ``[sink + current]`` via two SDPA calls.

    * ``n_context_tokens == 0`` (AR mode): Q comes from the active block only;
      the per-block ``_tf_kv_prefix`` dict (``rf_shifted_sink``) supplies the
      pre-RoPE sink K/V (re-RoPE'd here with its sliding offset PE) and the
      post-RoPE recent-history K/V, concatenated before SDPA. The
      ``_kv_cache_capture`` and ``_tf_capture_kv`` hooks record K/V into the
      module for the AR orchestrator to read back.
    """
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

    # KV-cache capture / inject hooks for ``rf_shifted_sink`` AR refinement:
    # - ``_kv_cache_capture`` saves PRE-RoPE (post-norm) K/V so a future window
    #   can re-apply RoPE at its shifted sink offset.
    # - ``_tf_capture_kv`` saves POST-RoPE K/V so the next window can directly
    #   concatenate the recent history.
    # - ``_tf_kv_prefix`` (a dict with ``mode='rf_shifted_sink'``) prepends a
    #   re-RoPE'd sink + already-post-RoPE recent history before SDPA.
    if getattr(attn, "_kv_cache_capture", False):
        attn._cached_kv_pre = (key.detach().clone(), value.detach().clone())

    if attn.rope_type == "interleaved":
        query = apply_interleaved_rotary_emb(query, query_rotary_emb)
        key = apply_interleaved_rotary_emb(key, query_rotary_emb)
    elif attn.rope_type == "split":
        query = apply_split_rotary_emb(query, query_rotary_emb)
        key = apply_split_rotary_emb(key, query_rotary_emb)
    else:
        raise ValueError(f"Unsupported LTX-2 RoPE type: {attn.rope_type}")

    if getattr(attn, "_tf_capture_kv", False):
        attn._cached_kv_post = (key.detach().clone(), value.detach().clone())

    tf_prefix = getattr(attn, "_tf_kv_prefix", None)
    if isinstance(tf_prefix, dict) and tf_prefix.get("mode") == "rf_shifted_sink":
        prefix_k_parts: list[torch.Tensor] = []
        prefix_v_parts: list[torch.Tensor] = []
        sink_k_pre = tf_prefix.get("sink_k_pre")
        sink_v = tf_prefix.get("sink_v")
        if sink_k_pre is not None and sink_v is not None and sink_k_pre.shape[1] > 0:
            sink_pe = tf_prefix.get("sink_pe")
            if sink_pe is None:
                raise RuntimeError("rf_shifted_sink prefix requires a sink_pe RoPE tuple.")
            sink_k_pre_dt = sink_k_pre.to(key.dtype)
            if attn.rope_type == "interleaved":
                sink_k = apply_interleaved_rotary_emb(sink_k_pre_dt, sink_pe)
            else:
                sink_k = apply_split_rotary_emb(sink_k_pre_dt, sink_pe)
            prefix_k_parts.append(sink_k)
            prefix_v_parts.append(sink_v.to(value.dtype))
        history_k = tf_prefix.get("history_k")
        history_v = tf_prefix.get("history_v")
        if history_k is not None and history_v is not None and history_k.shape[1] > 0:
            prefix_k_parts.append(history_k.to(key.dtype))
            prefix_v_parts.append(history_v.to(value.dtype))
        if prefix_k_parts:
            key = torch.cat([*prefix_k_parts, key], dim=1)
            value = torch.cat([*prefix_v_parts, value], dim=1)

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    processor = attn.processor
    backend = getattr(processor, "_attention_backend", None)
    parallel_config = getattr(processor, "_parallel_config", None)

    # AR mode (n_context_tokens == 0): Q from active block attends to the
    # injected prefix + current K/V in one SDPA call. Legacy single-shot
    # mode keeps the sink-self / current-cross split.
    if n_context_tokens <= 0 or n_context_tokens >= query.shape[1]:
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=backend,
            parallel_config=parallel_config,
        )
    else:
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


def _set_kv_prefix_on_blocks(
    transformer: nn.Module,
    kv_prefix_per_layer: list[dict[str, object]] | None,
) -> None:
    """Attach a per-layer KV-prefix dict to each ``attn1`` for the AR refiner."""
    blocks = transformer.transformer_blocks
    if kv_prefix_per_layer is None:
        _clear_kv_prefix_on_blocks(transformer)
        return
    if len(kv_prefix_per_layer) != len(blocks):
        raise RuntimeError(
            f"kv_prefix_per_layer has {len(kv_prefix_per_layer)} entries but transformer has {len(blocks)} blocks."
        )
    for block, prefix in zip(blocks, kv_prefix_per_layer):
        block.attn1._tf_kv_prefix = prefix


def _clear_kv_prefix_on_blocks(transformer: nn.Module) -> None:
    for block in transformer.transformer_blocks:
        block.attn1._tf_kv_prefix = None


def _set_capture_flag_on_blocks(transformer: nn.Module, mode: str, *, enable: bool) -> None:
    """Toggle ``_kv_cache_capture`` (pre-RoPE) or ``_tf_capture_kv`` (post-RoPE) per block."""
    if mode == "pre_rope":
        attr = "_kv_cache_capture"
        clear_attr = "_cached_kv_pre"
    elif mode == "post_rope":
        attr = "_tf_capture_kv"
        clear_attr = "_cached_kv_post"
    else:
        raise ValueError(f"capture_mode must be 'pre_rope' or 'post_rope', got {mode!r}")
    for block in transformer.transformer_blocks:
        setattr(block.attn1, attr, bool(enable))
        if enable and hasattr(block.attn1, clear_attr):
            setattr(block.attn1, clear_attr, None)


def _collect_captured_kv_from_blocks(
    transformer: nn.Module,
    mode: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    attr = "_cached_kv_pre" if mode == "pre_rope" else "_cached_kv_post"
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for block in transformer.transformer_blocks:
        cached = getattr(block.attn1, attr, None)
        if cached is None:
            raise RuntimeError(f"Expected {attr!r} on attn1 after capture forward, but found None.")
        out.append(cached)
        # Release the reference so the orchestrator owns the only handle.
        setattr(block.attn1, attr, None)
    return out


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
        batch_size,
        -1,
        num_frames // patch_size_t,
        patch_size_t,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
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
