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
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizer, GemmaTokenizerFast

from ...models.autoencoders import AutoencoderKLLTX2Video
from ...models.transformers.transformer_sana_wm_refiner import (
    KV_CACHE_MODE_CAPTURE_PRE_ROPE,
    KV_CACHE_MODE_INJECT,
    KV_CACHE_MODE_INJECT_AND_CAPTURE_POST_ROPE,
    SanaWMLTX2RefinerTransformer3DModel,
    SanaWMRefinerKVCache,
)
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor

# TODO: `LTX2TextConnectors` lives in the LTX-2 pipeline folder, so stage 2 has to reach across
# pipelines for it. Once https://github.com/huggingface/diffusers/issues/14749 moves the connector
# to a shared home (e.g. `models/`), import it from there and drop this cross-pipeline import.
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
        vae ([`AutoencoderKLLTX2Video`], *optional*):
            The same VAE used by [`SanaWMPipeline`]; pass `vae=pipe.vae` to share the weights. When given, the refiner
            decodes to video, otherwise it returns refined latents.
        text_max_sequence_length (`int`, defaults to 1024):
            Maximum tokens passed to the Gemma-3 tokenizer.
    """

    model_cpu_offload_seq = "text_encoder->connectors->transformer->vae"
    _optional_components = ["vae"]

    def __init__(
        self,
        transformer: SanaWMLTX2RefinerTransformer3DModel,
        connectors: LTX2TextConnectors,
        tokenizer: GemmaTokenizer | GemmaTokenizerFast,
        text_encoder: Gemma3ForConditionalGeneration,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLLTX2Video | None = None,
        text_max_sequence_length: int = 1024,
    ) -> None:
        super().__init__()
        self.register_modules(
            transformer=transformer,
            connectors=connectors,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            vae=vae,
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
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
        block_size: int = 3,
        kv_max_frames: int = 11,
        sigmas: tuple[float, ...] = STAGE_2_DISTILLED_SIGMA_VALUES,
        output_type: str = "np",
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
            block_size: latent frames per AR block (canonical: 3).
            kv_max_frames: maximum context+active frames retained in the
                sliding window (canonical: 11 = 1 sink + 10 recent).
            sigmas: descending Euler schedule terminating at 0.0 (canonical
                3-step distilled: ``(0.909375, 0.725, 0.421875, 0.0)``). Fed to ``self.scheduler`` (minus the trailing
                0.0, which the scheduler appends itself).
            output_type: `"latent"` returns the refined latents. Anything else decodes through `self.vae` and
                post-processes to that type (`"np"`, `"pt"`, `"pil"`); without a `vae` the latents are returned
                regardless.

        Returns:
            `torch.Tensor`: Refined VAE latents of shape ``(B, C, F, H, W)`` — the first ``sink_size`` frames carry the
            raw stage-1 sink latents unchanged, the rest carry the refined output.
        """
        if sana_latent.shape[2] <= sink_size:
            raise ValueError(f"Stage-1 latent has {sana_latent.shape[2]} frames but sink_size={sink_size}.")

        # Stage 2 is memory hungry (a Gemma-3 text encoder plus a 48-layer DiT), so it is meant to be
        # run under `enable_model_cpu_offload()`: `model_cpu_offload_seq` walks
        # `text_encoder -> connectors -> transformer -> vae`, which is exactly the order below, so each
        # sub-model is on the accelerator only while it runs.
        device = self._execution_device
        dtype = next(self.transformer.parameters()).dtype
        transformer_config = self.transformer.config
        sink_size = int(sink_size)
        block_size = int(block_size)
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(42)

        # 1. Load the distilled sigma schedule into the scheduler. Drop the trailing
        # 0.0 — ``FlowMatchEulerDiscreteScheduler.set_timesteps`` appends the
        # terminal 0.0 itself, so ``self.scheduler.sigmas`` reproduces ``sigmas``.
        self.scheduler.set_timesteps(sigmas=list(sigmas[:-1]), device=device)
        sigmas_t = self.scheduler.sigmas.to(device=device, dtype=torch.float32)
        sigma_max = float(sigmas_t[0])

        # 2. Encode the prompt.
        prompt_embeds, prompt_attention_mask = self._encode_prompt(prompt, device=device, dtype=dtype)

        # 3. Prepare the latents. The output keeps the raw sink prefix verbatim; the AR blocks fill
        # frames [sink_size, num_frames).
        z = sana_latent.to(device=device, dtype=dtype)
        latents = z.clone()
        batch_size, _, num_frames, height, width = z.shape
        num_blocks = (num_frames - sink_size + block_size - 1) // block_size

        # 4. Chunk-causal AR refinement implementing the canonical `rf_shifted_sink` KV-cache contract:
        #
        # a. Pre-capture **pre-RoPE** sink K/V from raw `z_sana[:sink_size]` at sigma=0. The sink frames themselves
        #    are never refined — they sit unchanged in the output volume.
        # b. AR blocks cover frames `[sink_size, num_frames)` in `block_size`-frame chunks. For each block:
        #    - Initialize `x_t = (1-sigma_0) * z_sana_block + sigma_0 * eps` (single eps per block).
        #    - 3-step deterministic Euler. Each step injects the per-layer prefix
        #      `{sink_k_pre, sink_v, sink_pe, history_k, history_v}`, where `sink_pe` is rebuilt at
        #      `sink_rope_offset = block_start - history_frames - sink_size` so the sink slides to sit immediately
        #      before the bounded working cache.
        #    - Capture **post-RoPE** K/V from the refined block under the same prefix, append it to the history,
        #      and trim the history to `kv_max_frames - sink_size` frames.
        num_layers = len(self.transformer.transformer_blocks)
        max_history_frames = int(kv_max_frames) - sink_size
        # ``_pack_latents`` emits ``(T // patch_size_t) * (H // p) * (W // p)`` tokens, so a single latent
        # frame contributes ``(H // p) * (W // p) / patch_size_t`` tokens. (No-op for LTX-2, which uses
        # ``patch_size_t=1``.)
        tokens_per_frame = (
            (height // transformer_config.patch_size)
            * (width // transformer_config.patch_size)
            // transformer_config.patch_size_t
        )
        history_frames = 0

        kv_cache = SanaWMRefinerKVCache(num_layers)
        self._capture_block_kv(
            clean_block=z[:, :, :sink_size].contiguous(),
            frame_positions=list(range(sink_size)),
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            fps=fps,
            kv_cache=kv_cache,
            kv_cache_mode=KV_CACHE_MODE_CAPTURE_PRE_ROPE,
            device=device,
        )
        for layer_idx in range(num_layers):
            layer_cache = kv_cache.get(layer_idx)
            layer_cache.store_sink(*layer_cache.get_captured_pre_rope())

        with self.progress_bar(total=num_blocks) as progress_bar:
            for block_idx in range(num_blocks):
                block_start = sink_size + block_idx * block_size
                block_end = min(block_start + block_size, num_frames)
                clean_block = z[:, :, block_start:block_end]
                frame_positions = list(range(block_start, block_end))

                # Slide the sink's RoPE so it sits immediately before the bounded working cache.
                sink_rope_offset = block_start - history_frames - sink_size
                kv_cache.sink_pe = self.transformer.build_rotary_emb_for_absolute_positions(
                    batch_size=batch_size,
                    frame_positions=list(range(sink_rope_offset, sink_rope_offset + sink_size)),
                    height=height,
                    width=width,
                    device=device,
                    fps=float(fps),
                )

                # FM endpoint at sigma_max: a single epsilon per block.
                noise = randn_tensor(clean_block.shape, generator=generator, device=device, dtype=dtype)
                latent_block = ((1.0 - sigma_max) * clean_block.float() + sigma_max * noise.float()).to(dtype)

                # Reset the shared scheduler to step 0 for this block's Euler run (blocks are processed
                # sequentially, so re-seeding the schedule per block is safe).
                self.scheduler.set_timesteps(sigmas=[float(s) for s in sigmas_t[:-1]], device=device)
                timesteps = self.scheduler.timesteps

                for i, t in enumerate(timesteps):
                    sigma = float(sigmas_t[i].item())

                    # Only the active block is forwarded; its queries attend to the `[sink, history, current]`
                    # K/V supplied by `kv_cache`. All active tokens carry the same sigma.
                    latent_tokens = _pack_latents(
                        latent_block,
                        patch_size=transformer_config.patch_size,
                        patch_size_t=transformer_config.patch_size_t,
                    )
                    seq_len = latent_tokens.shape[1]
                    timestep = torch.full(
                        (batch_size, seq_len),
                        sigma * float(transformer_config.timestep_scale_multiplier),
                        dtype=torch.float32,
                        device=device,
                    )
                    video_rotary_emb = self.transformer.build_rotary_emb_for_absolute_positions(
                        batch_size=batch_size,
                        frame_positions=frame_positions,
                        height=height,
                        width=width,
                        device=device,
                        fps=float(fps),
                    )
                    velocity_pred = self.transformer(
                        hidden_states=latent_tokens,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        video_rotary_emb=video_rotary_emb,
                        encoder_attention_mask=prompt_attention_mask,
                        kv_cache=kv_cache,
                        kv_cache_mode=KV_CACHE_MODE_INJECT,
                        return_dict=False,
                    )[0]

                    # FM x0 prediction: x_t - σ_cur · v.
                    raw_sigma = torch.full((batch_size, seq_len, 1), sigma, dtype=torch.float32, device=device)
                    denoised_tokens = latent_tokens.float() - velocity_pred.float() * raw_sigma
                    pred_x0 = _unpack_latents(
                        denoised_tokens.to(dtype),
                        num_frames=block_end - block_start,
                        height=height,
                        width=width,
                        patch_size=transformer_config.patch_size,
                        patch_size_t=transformer_config.patch_size_t,
                    )

                    if sigma <= 1.0e-6:
                        latent_block = pred_x0.to(dtype)
                    else:
                        # FM velocity from x0; the scheduler applies the Euler update.
                        velocity = (latent_block.float() - pred_x0.float()) / sigma
                        latent_block = self.scheduler.step(velocity, t, latent_block.float(), return_dict=False)[0].to(
                            dtype
                        )

                # Capture POST-RoPE K/V for this refined block under the same prefix and append it to the history.
                self._capture_block_kv(
                    clean_block=latent_block,
                    frame_positions=frame_positions,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    fps=fps,
                    kv_cache=kv_cache,
                    kv_cache_mode=KV_CACHE_MODE_INJECT_AND_CAPTURE_POST_ROPE,
                    device=device,
                )
                for layer_idx in range(num_layers):
                    layer_cache = kv_cache.get(layer_idx)
                    new_key, new_value = layer_cache.get_captured_post_rope()
                    history = layer_cache.get_history()
                    if history is None:
                        layer_cache.store_history(new_key, new_value)
                    else:
                        layer_cache.store_history(
                            torch.cat([history[0], new_key], dim=1),
                            torch.cat([history[1], new_value], dim=1),
                        )
                history_frames += block_end - block_start

                # Trim the history so the sliding window stays bounded.
                if max_history_frames > 0 and history_frames > max_history_frames:
                    keep_tokens = max_history_frames * tokens_per_frame
                    for layer_idx in range(num_layers):
                        layer_cache = kv_cache.get(layer_idx)
                        history = layer_cache.get_history()
                        if history is not None:
                            layer_cache.store_history(history[0][:, -keep_tokens:], history[1][:, -keep_tokens:])
                    history_frames = max_history_frames

                latents[:, :, block_start:block_end] = latent_block
                progress_bar.update()

        if self.vae is None or output_type == "latent":
            self.maybe_free_model_hooks()
            return latents

        # The sink frames are carried through unrefined, so drop the anchor before decoding.
        decoded = self._decode_latents(latents)[:, :, sink_size:]
        video = self.video_processor.postprocess_video(decoded, output_type=output_type)[0]

        self.maybe_free_model_hooks()
        return video

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to a `(B, C, F, H, W)` tensor in `[-1, 1]` (the VAE's native output range)."""
        latents = latents.to(self._execution_device, dtype=self.vae.dtype)
        latents_mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(latents)
        latents_std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(latents)
        latents = latents / self.vae.config.scaling_factor * latents_std + latents_mean
        return self.vae.decode(latents, return_dict=False)[0]

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

        # Call the top-level `text_encoder` (not its inner backbone) so that the model CPU offload
        # hook installed on it by `enable_model_cpu_offload()` fires and onloads it first.
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states, dim=-1)
        sequence_lengths = attention_mask.sum(dim=-1)
        prompt_embeds = _pack_text_embeds(
            hidden_states,
            sequence_lengths,
            device=device,
            padding_side="left",
        ).to(dtype=dtype)

        connector_prompt_embeds, _, connector_attention_mask = self.connectors(prompt_embeds, attention_mask)
        return (
            connector_prompt_embeds.to(device=device, dtype=dtype),
            connector_attention_mask.to(device=device),
        )


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
