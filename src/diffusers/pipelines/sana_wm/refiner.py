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

Wraps [`SanaWMLTX2RefinerTransformer3DModel`] (an LTX-2 DiT with a sliding-window KV cache and a video-only forward)
plus ``LTX2TextConnectors`` and a Gemma-3 text encoder.

Refinement is chunk-causal / autoregressive (``block_size=3``, ``kv_max_frames=11``): ``block_size`` latent frames are
processed at a time over a sliding window of ``[source_sink + recent_history + active_block]`` K/V. The model was
trained with this contract; per-block compute is bounded by the window size, so total cost scales linearly with video
length.
"""

from __future__ import annotations

import torch
from tqdm.auto import tqdm
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizer, GemmaTokenizerFast

from ...models.transformers.transformer_sana_wm_refiner import (
    KV_CACHE_MODE_CAPTURE_PRE_ROPE,
    KV_CACHE_MODE_INJECT,
    KV_CACHE_MODE_INJECT_AND_CAPTURE_POST_ROPE,
    SanaWMLTX2RefinerTransformer3DModel,
    SanaWMRefinerKVCache,
)
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils.torch_utils import empty_device_cache, randn_tensor
from ..ltx2.connectors import LTX2TextConnectors
from ..pipeline_utils import DiffusionPipeline


# Sigma schedule for the 3-step distilled refiner (matches the public release).
STAGE_2_DISTILLED_SIGMA_VALUES: tuple[float, ...] = (0.909375, 0.725, 0.421875, 0.0)


class SanaWMLTX2Refiner(DiffusionPipeline):
    r"""
    LTX-2 sink-bidirectional Euler refiner — SANA-WM stage 2, as a standalone pipeline.

    Wraps the LTX-2 components (refiner transformer + text connectors + Gemma-3 text encoder + tokenizer) plus a
    [`FlowMatchEulerDiscreteScheduler`] that carries the distilled sigma schedule and performs the Euler steps. It is
    registered as an optional component of [`SanaWMPipeline`] and can also be used on its own to refine stage-1
    latents.

    Args:
        transformer ([`SanaWMLTX2RefinerTransformer3DModel`]):
            The LTX-2 video DiT with the chunk-causal sliding-window KV cache.
        connectors ([`LTX2TextConnectors`]):
            LTX-2 text connectors.
        tokenizer:
            Gemma-3 tokenizer.
        text_encoder:
            Gemma-3 text encoder.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Flow-matching Euler scheduler. Constructed with ``shift=1.0`` so the distilled sigmas pass through
            unmodified.
        text_max_sequence_length (`int`, defaults to 1024):
            Maximum tokens passed to the Gemma-3 tokenizer.
    """

    model_cpu_offload_seq = "text_encoder->connectors->transformer"

    def __init__(
        self,
        transformer: SanaWMLTX2RefinerTransformer3DModel,
        connectors: LTX2TextConnectors,
        tokenizer: GemmaTokenizer | GemmaTokenizerFast,
        text_encoder: Gemma3ForConditionalGeneration,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_max_sequence_length: int = 1024,
    ) -> None:
        super().__init__()
        self.register_modules(
            transformer=transformer,
            connectors=connectors,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
        )
        self.register_to_config(text_max_sequence_length=int(text_max_sequence_length))
        self.text_max_sequence_length = int(text_max_sequence_length)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        sana_latent: torch.Tensor,
        prompt: str,
        *,
        fps: float,
        sink_size: int = 1,
        generator: torch.Generator | None = None,
        progress: bool = True,
        block_size: int = 3,
        kv_max_frames: int = 11,
        sigmas: tuple[float, ...] = STAGE_2_DISTILLED_SIGMA_VALUES,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Run the LTX-2 refiner and return refined VAE latents.

        Uses the chunk-causal AR recipe the model was trained on (``block_size=3``, ``kv_max_frames=11``): a sliding
        window of ``[source_sink + recent_history + active_block]`` K/V is fed to the transformer one block at a time,
        so per-block compute is bounded and total refinement cost scales linearly with video length.

        Args:
            sana_latent: ``(B, C, F, H, W)`` stage-1 latent.
            prompt: text prompt.
            fps: video frame rate (drives LTX-2 RoPE temporal scaling).
            sink_size: how many leading raw ``z_sana`` frames to anchor as the
                attention sink (canonical: 1).
            generator: torch.Generator for the FM endpoint noise. Defaults to a generator seeded with 42
                so results are reproducible out of the box.
            progress: show a tqdm bar.
            block_size: latent frames per AR block (canonical: 3).
            kv_max_frames: maximum context+active frames retained in the
                sliding window (canonical: 11 = 1 sink + 10 recent).
            sigmas: descending Euler schedule terminating at 0.0 (canonical
                3-step distilled: ``(0.909375, 0.725, 0.421875, 0.0)``). Fed to ``self.scheduler`` (minus the trailing
                0.0, which the scheduler appends itself).
            device: execution device for the refiner's sub-modules. If ``None``, falls back to where the transformer
                currently lives. The refiner moves each sub-module on/off this device as it runs.

        Returns:
            `torch.Tensor`: Refined VAE latents of shape ``(B, C, F, H, W)`` — the first ``sink_size`` frames carry the
            raw stage-1 sink latents unchanged, the rest carry the refined output.
        """
        if sana_latent.shape[2] <= sink_size:
            raise ValueError(f"Stage-1 latent has {sana_latent.shape[2]} frames but sink_size={sink_size}.")

        dtype = next(self.transformer.parameters()).dtype
        # The refiner moves its own sub-modules on/off ``device`` as it runs (so
        # peak VRAM ~= the largest single sub-model, not the sum). Callers pass
        # the execution device explicitly; otherwise fall back to where the
        # transformer currently lives.
        if device is None:
            device = next(self.transformer.parameters()).device
        device = torch.device(device)

        # Load the distilled sigma schedule into the scheduler. Drop the trailing
        # 0.0 — ``FlowMatchEulerDiscreteScheduler.set_timesteps`` appends the
        # terminal 0.0 itself, so ``self.scheduler.sigmas`` reproduces ``sigmas``.
        self.scheduler.set_timesteps(sigmas=list(sigmas[:-1]), device=device)
        sigmas_t = self.scheduler.sigmas.to(device=device, dtype=torch.float32)

        # Free transformer GPU memory while we run the text encoder.
        self.transformer.to("cpu")
        empty_device_cache(device.type)
        prompt_embeds, prompt_attention_mask = self._encode_prompt(prompt, device=device, dtype=dtype)

        self.transformer.to(device)
        z = sana_latent.to(device=device, dtype=dtype)

        # Chunk-causal AR refinement implementing the canonical `rf_shifted_sink` KV-cache contract:
        #
        # 1. Pre-capture **pre-RoPE** sink K/V from raw `z_sana[:sink_size]` at sigma=0. The sink frames themselves
        #    are never refined — they sit unchanged in the output volume.
        # 2. AR blocks cover frames `[sink_size, T_full)` in `block_size`-frame chunks. For each block:
        #    - Initialize `x_t = (1-sigma_0) * z_sana_block + sigma_0 * eps` (single eps per block).
        #    - 3-step deterministic Euler. Each step injects the per-layer prefix
        #      `{sink_k_pre, sink_v, sink_pe, history_k, history_v}`, where `sink_pe` is rebuilt at
        #      `sink_rope_offset = active_start - history_frames - sink_size` so the sink slides to sit immediately
        #      before the bounded working cache.
        #    - Capture **post-RoPE** K/V from the refined block under the same prefix, append to `history_kv_post`,
        #      and trim to `kv_max_frames - sink_size`.
        sink_size = int(sink_size)
        block_size = int(block_size)
        runner = _RefinerChunkRunner(
            self,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            fps=fps,
            sigmas=sigmas_t,
            source_sink_frames=sink_size,
            block_size=block_size,
            kv_max_frames=int(kv_max_frames),
            generator=generator,
            spatial_shape=(int(z.shape[3]), int(z.shape[4])),
            dtype=dtype,
            device=device,
        )

        # Output keeps the raw sink prefix verbatim; AR blocks fill frames [sink_size, T_full).
        T_full = z.shape[2]
        output = z.clone()
        n_active = max(T_full - sink_size, 0)
        n_blocks = (n_active + block_size - 1) // block_size if n_active > 0 else 0

        iterator = range(n_blocks)
        if progress:
            iterator = tqdm(iterator, desc="refiner-ar", unit="block", total=n_blocks)

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
        kv_cache: SanaWMRefinerKVCache,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Forward through the transformer on the active block only and return x0.

        The active block's Q attends to ``[sink, history, current]`` K/V supplied by ``kv_cache``. All active tokens
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

        video_rotary_emb = self.transformer.build_rotary_emb_for_absolute_positions(
            batch_size=batch_size,
            frame_positions=active_positions,
            height=int(active.shape[3]),
            width=int(active.shape[4]),
            device=device,
            fps=float(fps),
        )

        velocity = self.transformer(
            hidden_states=latent_tokens,
            encoder_hidden_states=prompt_embeds,
            timestep=model_timestep,
            video_rotary_emb=video_rotary_emb,
            encoder_attention_mask=prompt_attention_mask,
            kv_cache=kv_cache,
            kv_cache_mode=KV_CACHE_MODE_INJECT,
            return_dict=False,
        )[0]

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

    def _capture_block_kv(
        self,
        *,
        clean_block: torch.Tensor,
        frame_positions: list[int],
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        fps: float,
        kv_cache: SanaWMRefinerKVCache,
        kv_cache_mode: str,
        device: torch.device,
    ) -> None:
        """Run one forward at σ=0 in a capturing ``kv_cache_mode``; the K/V lands in ``kv_cache``.

        ``'capture_pre_rope'`` saves PRE-RoPE K/V (so a future window can re-RoPE the sink to its shifted offset) and
        injects no prefix. ``'inject_and_capture_post_rope'`` attends to the current window's prefix and saves the
        block's POST-RoPE K/V, ready to be appended to the recent history.
        """
        latent_tokens = _pack_latents(
            clean_block,
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        )
        batch_size, seq_len, _ = latent_tokens.shape
        model_timestep = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)

        video_rotary_emb = self.transformer.build_rotary_emb_for_absolute_positions(
            batch_size=batch_size,
            frame_positions=frame_positions,
            height=int(clean_block.shape[3]),
            width=int(clean_block.shape[4]),
            device=device,
            fps=float(fps),
        )

        self.transformer(
            hidden_states=latent_tokens,
            encoder_hidden_states=prompt_embeds,
            timestep=model_timestep,
            video_rotary_emb=video_rotary_emb,
            encoder_attention_mask=prompt_attention_mask,
            kv_cache=kv_cache,
            kv_cache_mode=kv_cache_mode,
            return_dict=False,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _encode_prompt(
        self, prompt: str, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self.tokenizer
        text_inputs = tokenizer(
            [prompt.strip()],
            padding="max_length",
            padding_side="left",
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
            padding_side="left",
        ).to(dtype=dtype)

        # Release the text encoder once we have the prompt embeds — otherwise it
        # stays resident on GPU through the entire (much longer) AR refinement.
        self.text_encoder.to("cpu")
        del outputs, hidden_states
        empty_device_cache(device.type)

        self.connectors.to(device)
        connector_prompt_embeds, _, connector_attention_mask = self.connectors(prompt_embeds, attention_mask)
        self.connectors.to("cpu")
        del prompt_embeds, attention_mask
        empty_device_cache(device.type)

        return (
            connector_prompt_embeds.to(device=device, dtype=dtype),
            connector_attention_mask.to(device=device),
        )


class _RefinerChunkRunner:
    """Stateful per-AR-block driver for :class:`SanaWMLTX2Refiner`.

    Owns the [`SanaWMRefinerKVCache`] that the chunk-causal AR recipe accumulates as refiner blocks complete:

    * each layer cache's **sink** entry holds the pre-RoPE K/V captured from the first ``source_sink_frames`` raw
      stage-1 latents at σ=0. Lazily filled on the first call to :meth:`refine_block`.
    * each layer cache's **history** entry holds the post-RoPE K/V of every refined block already produced, trimmed to
      ``kv_max_frames - source_sink_frames`` frames so the sliding window stays bounded.
    * ``_history_frames``: number of frames currently held in the history.
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
        generator: torch.Generator | None,
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
        self._max_history_frames = int(kv_max_frames) - int(source_sink_frames)
        self._device = device
        self._dtype = dtype
        self._generator = generator if generator is not None else torch.Generator(device=self._device).manual_seed(42)

        transformer = refiner.transformer
        self._n_layers = len(transformer.transformer_blocks)
        H, W = spatial_shape
        self._H, self._W = int(H), int(W)
        # ``_pack_latents`` emits ``(T // patch_size_t) * (H // p) * (W // p)`` tokens,
        # so a single latent frame contributes ``(H // p) * (W // p) / patch_size_t``
        # tokens. (No-op for LTX-2, which uses ``patch_size_t=1``.)
        self._tokens_per_frame = (
            int(H // transformer.config.patch_size)
            * int(W // transformer.config.patch_size)
            // int(transformer.config.patch_size_t)
        )

        self._kv_cache = SanaWMRefinerKVCache(self._n_layers)
        self._sink_captured = False
        self._history_frames: int = 0

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
                first frame (drives the ``rf_shifted_sink`` RoPE offset). Must be >= ``source_sink_frames``.
            block_end: absolute latent-frame index just past the active block.
            sink_seed_frames: ``(B, C, source_sink_frames, H, W)`` raw sink
                latents used once on the first call to pre-capture the pre-RoPE sink K/V at ``sigma=0`` with frame
                positions ``[0, source_sink_frames)``.
        """
        refiner = self._refiner
        device = self._device
        B = int(clean_block.shape[0])
        active_len = block_end - block_start
        if block_start < self._source_sink_frames:
            raise ValueError(
                f"block_start={block_start} overlaps the source sink (source_sink_frames={self._source_sink_frames})."
            )

        # 1) On the first call: pre-capture PRE-RoPE sink K/V from the supplied
        # raw sink latents at sigma=0 with absolute positions [0, sink_size).
        if not self._sink_captured:
            if sink_seed_frames is None:
                raise ValueError("First refine_block call requires sink_seed_frames (raw stage-1 sink latents).")
            if sink_seed_frames.shape[2] != self._source_sink_frames:
                raise ValueError(
                    f"sink_seed_frames has {sink_seed_frames.shape[2]} frames "
                    f"but source_sink_frames={self._source_sink_frames}."
                )
            source_sink = sink_seed_frames.contiguous()
            refiner._capture_block_kv(
                clean_block=source_sink,
                frame_positions=list(range(self._source_sink_frames)),
                prompt_embeds=self._prompt_embeds,
                prompt_attention_mask=self._prompt_attention_mask,
                fps=self._fps,
                kv_cache=self._kv_cache,
                kv_cache_mode=KV_CACHE_MODE_CAPTURE_PRE_ROPE,
                device=device,
            )
            for layer_idx in range(self._n_layers):
                layer_cache = self._kv_cache.get(layer_idx)
                layer_cache.store_sink(*layer_cache.get_captured_pre_rope())
            self._sink_captured = True

        # 2) Slide the sink's RoPE so it sits immediately before the bounded working cache.
        sink_rope_offset = block_start - self._history_frames - self._source_sink_frames
        self._kv_cache.sink_pe = refiner.transformer.build_rotary_emb_for_absolute_positions(
            batch_size=B,
            frame_positions=list(range(sink_rope_offset, sink_rope_offset + self._source_sink_frames)),
            height=self._H,
            width=self._W,
            device=device,
            fps=self._fps,
        )

        # 3) FM endpoint at sigma=sigma0: single epsilon per block.
        eps = randn_tensor(clean_block.shape, generator=self._generator, device=device, dtype=self._dtype)
        x_t = ((1.0 - self._sigma_max) * clean_block.float() + self._sigma_max * eps.float()).to(self._dtype)

        # Reset the shared scheduler to step 0 for this block's Euler run (blocks
        # are processed sequentially, so re-seeding the schedule per block is safe).
        scheduler = refiner.scheduler
        scheduler.set_timesteps(sigmas=[float(s) for s in self._sigmas[:-1]], device=device)
        timesteps = scheduler.timesteps

        active_positions = list(range(int(block_start), int(block_end)))
        for level, t in enumerate(timesteps):
            sigma_cur = float(self._sigmas[level].item())
            pred_x0 = refiner._predict_x0_active_block(
                active=x_t,
                active_positions=active_positions,
                sigma_cur=sigma_cur,
                prompt_embeds=self._prompt_embeds,
                prompt_attention_mask=self._prompt_attention_mask,
                fps=self._fps,
                kv_cache=self._kv_cache,
                dtype=self._dtype,
                device=device,
            )
            if sigma_cur <= 1.0e-6:
                x_t = pred_x0.to(self._dtype)
            else:
                # FM velocity from x0; the scheduler applies the Euler update.
                velocity = (x_t.float() - pred_x0.float()) / sigma_cur
                x_t = scheduler.step(velocity, t, x_t.float(), return_dict=False)[0].to(self._dtype)

        # 4) Capture POST-RoPE K/V for this refined block under the same prefix.
        refiner._capture_block_kv(
            clean_block=x_t,
            frame_positions=active_positions,
            prompt_embeds=self._prompt_embeds,
            prompt_attention_mask=self._prompt_attention_mask,
            fps=self._fps,
            kv_cache=self._kv_cache,
            kv_cache_mode=KV_CACHE_MODE_INJECT_AND_CAPTURE_POST_ROPE,
            device=device,
        )
        for layer_idx in range(self._n_layers):
            layer_cache = self._kv_cache.get(layer_idx)
            new_k, new_v = layer_cache.get_captured_post_rope()
            old = layer_cache.get_history()
            if old is None:
                layer_cache.store_history(new_k, new_v)
            else:
                layer_cache.store_history(
                    torch.cat([old[0], new_k], dim=1),
                    torch.cat([old[1], new_v], dim=1),
                )
        self._history_frames += active_len

        if self._max_history_frames > 0 and self._history_frames > self._max_history_frames:
            keep_tokens = self._max_history_frames * self._tokens_per_frame
            for layer_idx in range(self._n_layers):
                layer_cache = self._kv_cache.get(layer_idx)
                hk = layer_cache.get_history()
                if hk is not None:
                    layer_cache.store_history(hk[0][:, -keep_tokens:], hk[1][:, -keep_tokens:])
            self._history_frames = self._max_history_frames

        return x_t


# -------------------------------------------------------------------------
# private helpers (text embedding + latent packing)
# -------------------------------------------------------------------------


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
