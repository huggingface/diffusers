# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import math
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor, _get_qkv_projections
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# FLUX is trained at 1024x1024 -> a 64x64 packed grid (4096 image tokens) plus 512 text tokens.
_TRAIN_SEQ_LEN = 64**2 + 512

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained(
        ...     "black-forest-labs/FLUX.1-dev", dtype=torch.bfloat16, custom_pipeline="pipeline_flux_hrdit"
        ... ).to("cuda")
        >>> image = pipe(
        ...     "a photo of a mountain lake at dawn", height=4096, width=4096
        ... ).images[0]
        >>> image.save("hrdit_4096.png")
        ```
"""


# ---------------------------------------------------------------------------------------
# SPA: Spatial Position Alignment + NTK-aware RoPE
# ---------------------------------------------------------------------------------------


def _phi(x: torch.Tensor, n1: int, size: int) -> torch.Tensor:
    """Bundle mapping: 0 for x < n1, else ceil((x + 1 - n1) / size). Monotonic non-decreasing."""
    return torch.where(x < n1, torch.zeros_like(x), (x + 1 - n1 + size - 1) // size)


def build_bundle_id_variants(img_ids: torch.Tensor, group_num: int) -> List[torch.Tensor]:
    """
    Spatial Position Alignment (SPA) bundle-index variants of the packed-latent position ids ``img_ids``.

    Maps each token's grid coordinate into a small number of bundles via a monotonic (non-wrapping) coarsening
    ``_phi`` -- so many neighbouring tokens share a position id inside the trained window, with no periodic tiling.
    Residual bundle-boundary seams are averaged out over ``group_num`` sliding-origin variants. Adapted from HRDiT
    (https://arxiv.org/abs/2608.07003), ``hrdit/spa.py``.

    Args:
        img_ids (`torch.Tensor`): Packed-latent position ids `(T, 3)`; column 1 row index, column 2 column index.
        group_num (`int`): Bundle granularity; bundle size is `ceil(max_index / (group_num - 1))`. Must be >= 2.

    Returns:
        `List[torch.Tensor]` of shape `(T, 3)` each, one per sliding bundle-boundary variant.
    """
    if group_num < 2:
        raise ValueError(f"`group_num` must be >= 2 for SPA, got {group_num}.")

    rows = img_ids[:, 1].long()
    cols = img_ids[:, 2].long()
    s_row = max(1, math.ceil(rows.max().item() / (group_num - 1)))
    s_col = max(1, math.ceil(cols.max().item() / (group_num - 1)))

    def variant(n1_row: int, n1_col: int) -> torch.Tensor:
        ids = img_ids.clone()
        ids[:, 1] = _phi(rows, n1_row, s_row).to(img_ids.dtype)
        ids[:, 2] = _phi(cols, n1_col, s_col).to(img_ids.dtype)
        return ids

    variants = [variant(s_row, s_col)]
    variants += [variant(n, s_col) for n in range(1, s_row)]
    variants += [variant(s_row, m) for m in range(1, s_col)]
    return variants


def flux_rope(ids: torch.Tensor, axes_dim, theta: float, ntk_factor: float = 1.0):
    """
    Flux rotary embeddings for position ids ``ids`` [S, len(axes_dim)], with NTK-aware scaling.

    Identical to diffusers' `FluxPosEmbed` at ``ntk_factor == 1``; NTK scaling multiplies the RoPE base ``theta`` by
    ``ntk_factor`` (HRDiT ``hrdit/transformer.py::get_1d_rotary_pos_embed``), lowering every frequency and thereby
    compressing out-of-range high-resolution positions back into the trained band -- the primary training-free
    high-resolution mechanism. Returns ``(cos, sin)`` each `[S, sum(axes_dim)]`.
    """
    scaled_theta = theta * ntk_factor
    cos_out, sin_out = [], []
    for i, dim in enumerate(axes_dim):
        pos = ids[:, i].to(torch.float64)
        exps = torch.arange(0, dim, 2, dtype=torch.float64, device=ids.device)[: dim // 2] / dim
        freqs = torch.outer(pos, 1.0 / (scaled_theta**exps))  # [S, dim/2]
        cos_out.append(freqs.cos().repeat_interleave(2, dim=1).float())
        sin_out.append(freqs.sin().repeat_interleave(2, dim=1).float())
    return torch.cat(cos_out, dim=-1), torch.cat(sin_out, dim=-1)


def butterworth_low_pass_filter_2d(height: int, width: int, ratio: float, device, order: int = 4) -> torch.Tensor:
    """Centered 2D Butterworth low-pass mask `[1, 1, H, W]` for frequency-domain structure guidance."""
    if ratio <= 0:
        return torch.zeros(1, 1, height, width, device=device)
    yy = (2.0 * torch.arange(height, device=device) / height - 1.0).view(height, 1)
    xx = (2.0 * torch.arange(width, device=device) / width - 1.0).view(1, width)
    d_square = yy**2 + xx**2
    mask = 1.0 / (1.0 + (d_square / ratio**2) ** order)
    return mask.view(1, 1, height, width)


def split_low_freq(x: torch.Tensor, freq_filter: torch.Tensor) -> torch.Tensor:
    """Low-frequency component of ``x`` [B, C, H, W] under a centered ``freq_filter`` (real output)."""
    x_freq = torch.fft.fftshift(torch.fft.fft2(x.to(freq_filter.dtype)))
    x_low = x_freq * freq_filter
    return torch.fft.ifft2(torch.fft.ifftshift(x_low)).real


def sharpen(image: torch.Tensor, kernel_size: int = 3, sigma: float = 1.0, alpha: float = 1.0) -> torch.Tensor:
    """Unsharp-mask sharpening of the upscaled structural prior before it is re-encoded."""
    blurred = gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    return (alpha + 1.0) * image - alpha * blurred


class _SPAState:
    """
    Module-level carrier for the current stage's rotary embeddings.

    The transformer blocks share one processor instance and are not aware of SPA/NTK, so the pipeline precomputes the
    rotary embeddings once per stage and arms them here. ``base_rope`` is the single NTK-scaled RoPE used on every
    step; ``variant_ropes`` are the SPA bundle variants used only while ``spa_active`` is set (the leading steps).
    """

    def __init__(self):
        self.base_rope = None
        self.variant_ropes = None
        self.spa_active = False

    @property
    def enabled(self):
        return self.base_rope is not None

    def arm(self, base_rope, variant_ropes):
        self.base_rope = base_rope
        self.variant_ropes = variant_ropes
        self.spa_active = False

    def disarm(self):
        self.base_rope = None
        self.variant_ropes = None
        self.spa_active = False

    def current_ropes(self):
        return self.variant_ropes if self.spa_active else [self.base_rope]


_SPA_STATE = _SPAState()


class HRDiTFluxAttnProcessor(FluxAttnProcessor):
    """
    Flux attention processor implementing HRDiT's NTK RoPE + Spatial Position Alignment (SPA).

    When armed the processor ignores the transformer's own rotary embedding and uses the NTK-scaled RoPE from
    [`_SPAState`] on every step; on the leading SPA steps it instead runs attention once per bundle variant and
    averages the outputs (`mean_n(softmax(A_n)) @ V == mean_n(softmax(A_n) @ V)`, so O(T*D) memory), with a
    proportional attention scale for the longer sequence. When disarmed it is exactly the stock `FluxAttnProcessor`.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        if not _SPA_STATE.enabled:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        head_dim = query.shape[-1]
        seq_len = query.shape[1]
        # Proportional attention scale for the longer high-res sequence (equals the stock 1/sqrt(d) at train length).
        scale = math.sqrt(math.log(seq_len, _TRAIN_SEQ_LEN) / head_dim) if seq_len > 1 else head_dim**-0.5

        value_t = value.transpose(1, 2).contiguous()  # [B, H, S, D]
        ropes = _SPA_STATE.current_ropes()
        acc = None
        for rope_variant in ropes:
            query_v = apply_rotary_emb(query, rope_variant, sequence_dim=1).transpose(1, 2).contiguous()
            key_v = apply_rotary_emb(key, rope_variant, sequence_dim=1).transpose(1, 2).contiguous()
            out = F.scaled_dot_product_attention(query_v, key_v, value_t, dropout_p=0.0, is_causal=False, scale=scale)
            acc = out if acc is None else acc + out
        hidden_states = (acc / len(ropes)).transpose(1, 2)  # [B, S, H, D]
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            num_txt = encoder_hidden_states.shape[1]
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [num_txt, hidden_states.shape[1] - num_txt], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        return hidden_states


# ---------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------


class HRDiTFluxPipeline(FluxPipeline):
    r"""
    Training-free high-resolution (up to 4096x4096) text-to-image with off-the-shelf Flux models.

    Adapted from HRDiT, "Training-Free High-Resolution Image Generation with Off-the-Shelf Diffusion Transformer
    Models" (https://arxiv.org/abs/2608.07003); MIT-licensed reference implementation at
    https://github.com/zylwithxy/HRDiT.

    Training-free pieces on top of the stock `FluxPipeline` denoise loop:

    - **NTK-aware RoPE scaling** (`flux_rope`) -- the primary high-resolution mechanism. On every upscale-stage step
      the rotary base `theta` is multiplied by a per-stage `ntk_factor`, compressing out-of-range positions into the
      trained band.
    - **SPA (Spatial Position Alignment)** -- for the leading `spa_steps` steps of a stage, `build_bundle_id_variants`
      additionally coarsens position ids via a monotonic bundle map (no wrapping) across sliding variants, averaged
      inside attention with a proportional scale. A light early-step correction for "spatial disorder".
    - **Progressive generation with structure guidance** -- the ladder climbs 1024 -> 2048 -> 4096; each stage decodes
      the previous latent, bicubic-upscales + sharpens it, and re-encodes it as a structural prior, then re-noises and
      denoises. At every step the low-frequency (coarse-structure) band of the prediction is pulled toward the
      upsampled previous-stage `pred_x0` (`alphas`, FFT Butterworth split), with a velocity-momentum term (`betas`).
      This is what keeps the highest stage from drifting to a washed-out mean.

    Not ported from the reference (documented follow-ups): HAP head-scope attention pruning
    (`configs/scope_plan_flux.json`), the `swin_pachify` shifted-window option, and DWT (as opposed to FFT) guidance.

    Args:
        prompt (`str` or `List[str]`): The prompt to render.
        height / width (`int`): Final output resolution. Defaults to 1024.
        resolutions (`List[int]`, optional): Progressive ladder (square side lengths). Defaults to doubling from 1024.
        group_num (`int`, defaults to 80): SPA bundle granularity; bundle size `ceil(max_index / (group_num - 1))`.
        ntk_factor (`List[float]`, optional): Per-upscale-stage NTK RoPE-base multiplier. Defaults to `[4.0, 10.0]`.
        spa_steps (`List[int]`, optional): Per-upscale-stage count of leading SPA steps. Defaults to `[3, 0]`.
        num_inference_steps (`int`, defaults to 30): Base-stage steps (also the shared schedule length).
        num_inference_steps_highres (`List[int]`, optional): Steps per upscale stage. Defaults to `[17, 10]`.
        guidance_scale (`float`, defaults to 3.5): Base-stage guidance.
        guidance_scale_highres (`List[float]`, optional): Per-upscale-stage guidance. Defaults to `[4.5, 6.0]`.
        alphas / betas (`List[float]`, optional): Per-stage structure-guidance weights (low-freq injection / velocity
            momentum). Default `[1.0, 0.25]` and `[0.5, 0.5]`.
        filter_ratio (`float`, defaults to 0.2): Butterworth low-pass cutoff for the structure split.

    Example: see `EXAMPLE_DOC_STRING`.
    """

    def _resolution_ladder(self, height: int, width: int, resolutions: Optional[List[int]]) -> List[int]:
        if resolutions is None:
            target = max(height, width)
            ladder = []
            side = min(1024, target)
            while side < target:
                ladder.append(side)
                side = min(side * 2, target)
            ladder.append(target)
            return ladder

        ladder = [int(res) for res in resolutions]
        if not ladder or any(ladder[i] >= ladder[i + 1] for i in range(len(ladder) - 1)):
            raise ValueError(f"`resolutions` must be a non-empty, strictly increasing list, got {resolutions}.")
        return ladder

    @staticmethod
    def _stage_dimensions(height: int, width: int, target: int, side: int, quant: int) -> tuple:
        if side >= target:
            return height, width
        stage_height = max(quant, int(round(height * side / target)) // quant * quant)
        stage_width = max(quant, int(round(width * side / target)) // quant * quant)
        return stage_height, stage_width

    def _encode_image_to_latents(self, image, batch_size, num_channels_latents):
        """Encode a pixel image to packed Flux latents (structural prior for a stage)."""
        latents = self.vae.encode(image.to(self.vae.dtype).to(self.vae.device)).latent_dist.mode()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        latent_height, latent_width = latents.shape[-2], latents.shape[-1]
        latents = self._pack_latents(latents, batch_size, num_channels_latents, latent_height, latent_width)
        return latents.to(self.transformer.dtype)

    def _flowmatch_step(
        self,
        model_output,
        timestep,
        sample,
        *,
        structure_on=False,
        pred_x0_dict=None,
        height_dict=None,
        width_dict=None,
        batch_size=None,
        num_channels_latents=None,
        target_height=None,
        target_width=None,
        filter_ratio=0.0,
        alpha=0.0,
        beta=0.0,
    ):
        """
        One flow-match step (the Euler update is delegated to ``self.scheduler.step``), optionally with HRDiT
        structure guidance.

        Structure guidance (``structure_on``) pulls the low-frequency band of the current predicted clean latent
        toward the upsampled previous-stage ``pred_x0`` (weight ``alpha``), then applies a cross-step velocity
        momentum (weight ``beta``); ``self._mo_high`` / ``self._mo_ref`` are reset per stage by the caller. Returns
        ``(prev_sample, pred_x0)`` where ``pred_x0`` is the *pre-guidance* prediction (stored as the next stage's
        structural reference).
        """
        sample = sample.to(torch.float32)
        sigma = self.scheduler.sigmas[self.scheduler.index_for_timestep(timestep)]

        pred_x0 = sample - model_output.to(torch.float32) * sigma
        original_pred_x0 = pred_x0

        if structure_on:
            x0 = self._unpack_latents(pred_x0, target_height, target_width, self.vae_scale_factor).float()
            latent_h, latent_w = x0.shape[-2], x0.shape[-1]

            ref_packed = pred_x0_dict[timestep.item()]
            ref = self._unpack_latents(
                ref_packed, height_dict[timestep.item()], width_dict[timestep.item()], self.vae_scale_factor
            ).float()
            ref = F.interpolate(ref, (latent_h, latent_w), mode="bicubic", align_corners=False)

            freq_filter = butterworth_low_pass_filter_2d(latent_h, latent_w, filter_ratio, x0.device)
            x0 = x0 + alpha * (split_low_freq(ref, freq_filter) - split_low_freq(x0, freq_filter))

            x0 = self._pack_latents(x0, batch_size, num_channels_latents, latent_h, latent_w)
            ref = self._pack_latents(ref, batch_size, num_channels_latents, latent_h, latent_w)

            model_output = (sample - x0) / (sigma + 1e-6)
            model_output_ref = (sample - ref) / (sigma + 1e-6)
            if self._mo_high is not None:
                model_output = model_output + beta * (self._mo_high + model_output_ref - self._mo_ref - model_output)
            self._mo_high = model_output
            self._mo_ref = model_output_ref
        else:
            model_output = model_output.to(torch.float32)

        # Let the scheduler own the Euler update; structure guidance only adjusts the velocity above.
        prev_sample = self.scheduler.step(model_output, timestep, sample, return_dict=False)[0]
        return prev_sample.to(self.transformer.dtype), original_pred_x0

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        resolutions: Optional[List[int]] = None,
        group_num: int = 80,
        ntk_factor: Optional[List[float]] = None,
        spa_steps: Optional[List[int]] = None,
        num_inference_steps: int = 30,
        num_inference_steps_highres: Optional[List[int]] = None,
        guidance_scale: float = 3.5,
        guidance_scale_highres: Optional[List[float]] = None,
        alphas: Optional[List[float]] = None,
        betas: Optional[List[float]] = None,
        filter_ratio: float = 0.2,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 512,
    ):
        r"""Generate a high-resolution image, training-free, with HRDiT (NTK RoPE + SPA + structure-guided progression).

        Examples:
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        quant = self.vae_scale_factor * 2
        if int(height) % quant != 0 or int(width) % quant != 0:
            raise ValueError(f"`height` and `width` must be multiples of {quant}, got {height} and {width}.")

        ladder = self._resolution_ladder(height, width, resolutions)
        target = max(height, width)

        ntk_schedule = ntk_factor if ntk_factor is not None else [4.0, 10.0]
        spa_schedule = spa_steps if spa_steps is not None else [3, 0]
        guidance_hr = guidance_scale_highres if guidance_scale_highres is not None else [4.5, 6.0]
        steps_hr = num_inference_steps_highres if num_inference_steps_highres is not None else [17, 10]
        alpha_schedule = alphas if alphas is not None else [1.0, 0.25]
        beta_schedule = betas if betas is not None else [0.5, 0.5]

        def _stage_value(schedule, j, fill):
            if not schedule:
                return fill
            return schedule[j] if j < len(schedule) else schedule[-1]

        device = self._execution_device
        dtype = prompt_embeds.dtype if prompt_embeds is not None else self.transformer.dtype

        # 1. Encode prompt.
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=None,
        )
        batch_size = prompt_embeds.shape[0]
        guidance_embeds = self.transformer.config.guidance_embeds

        def _guidance(scale):
            if not guidance_embeds:
                return None
            return torch.full([1], scale, device=device, dtype=torch.float32).expand(batch_size)

        self._joint_attention_kwargs = {}
        num_channels_latents = self.transformer.config.in_channels // 4
        axes_dim = self.transformer.pos_embed.axes_dim
        rope_theta = self.transformer.pos_embed.theta

        # Shared flow-match schedule (same sigmas + mu across stages so per-timestep pred_x0 references align).
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        base_h, base_w = ladder[0], ladder[0]
        base_grid = base_h // quant
        mu = calculate_shift(
            base_grid * base_grid,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )

        original_attn_processors = dict(self.transformer.attn_processors)
        self.transformer.set_attn_processor(HRDiTFluxAttnProcessor())

        pred_x0_dict, height_dict, width_dict = {}, {}, {}

        try:
            # 2. Base stage (1024): stock RoPE, standard flow-match; record pred_x0 per timestep for guidance.
            _SPA_STATE.disarm()
            latent_h = 2 * (base_h // quant)
            latent_w = 2 * (base_w // quant)
            latents = randn_tensor(
                (batch_size, num_channels_latents, latent_h, latent_w), generator=generator, device=device, dtype=dtype
            )
            latents = self._pack_latents(latents, batch_size, num_channels_latents, latent_h, latent_w)
            image_ids = self._prepare_latent_image_ids(batch_size, latent_h // 2, latent_w // 2, device, dtype)
            base_guidance = _guidance(guidance_scale)

            timesteps, _ = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
            cur_h, cur_w = base_h, base_w
            self.set_progress_bar_config(desc=f"HRDiT {base_w}x{base_h}")
            with self.progress_bar(total=len(timesteps)) as progress_bar:
                for t in timesteps:
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)
                    with self.transformer.cache_context("cond"):
                        noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=base_guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                    latents, pred_x0 = self._flowmatch_step(noise_pred, t, latents)
                    pred_x0_dict[t.item()] = pred_x0
                    height_dict[t.item()] = cur_h
                    width_dict[t.item()] = cur_w
                    progress_bar.update()

            # 3. Upscale stages with structure guidance.
            for stage in range(1, len(ladder)):
                j = stage - 1
                side = ladder[stage]
                stage_h, stage_w = self._stage_dimensions(height, width, target, side, quant)
                grid_h, grid_w = stage_h // quant, stage_w // quant
                stage_ntk = float(_stage_value(ntk_schedule, j, 1.0))
                stage_spa_steps = int(_stage_value(spa_schedule, j, 0))
                stage_guidance = _guidance(float(_stage_value(guidance_hr, j, guidance_scale)))
                stage_steps = int(_stage_value(steps_hr, j, max(1, round(num_inference_steps * 0.5))))
                stage_alpha0 = float(_stage_value(alpha_schedule, j, 0.0))
                stage_beta0 = float(_stage_value(beta_schedule, j, 0.0))

                # Structural prior: decode -> bicubic upscale -> sharpen -> re-encode at the new resolution.
                dec = self._unpack_latents(latents, cur_h, cur_w, self.vae_scale_factor)
                dec = (dec / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(dec.to(self.vae.dtype), return_dict=False)[0]
                image = F.interpolate(image, (stage_h, stage_w), mode="bicubic", align_corners=False)
                image = sharpen(image)
                latents = self._encode_image_to_latents(image, batch_size, num_channels_latents)
                image_ids = self._prepare_latent_image_ids(batch_size, grid_h, grid_w, device, dtype)

                # NTK RoPE (every step) + SPA bundle variants (leading steps only).
                base_rope = flux_rope(
                    torch.cat([text_ids, image_ids], dim=0), axes_dim, rope_theta, ntk_factor=stage_ntk
                )
                if stage_spa_steps > 0:
                    variants = build_bundle_id_variants(image_ids, group_num)
                    variant_ropes = [
                        flux_rope(torch.cat([text_ids, v], dim=0), axes_dim, rope_theta, ntk_factor=stage_ntk)
                        for v in variants
                    ]
                else:
                    variant_ropes = [base_rope]
                _SPA_STATE.arm(base_rope, variant_ropes)

                # Re-noise the prior to the tail of the shared schedule, then denoise the last `stage_steps` steps.
                retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
                dlfg_timesteps = self.scheduler.timesteps[-stage_steps:]
                noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                latents = self.scheduler.scale_noise(latents, dlfg_timesteps[:1], noise).to(self.transformer.dtype)

                # Reset the structure-guidance velocity momentum at the start of each stage.
                self._mo_high = None
                self._mo_ref = None
                self.set_progress_bar_config(desc=f"HRDiT {stage_w}x{stage_h}")
                with self.progress_bar(total=len(dlfg_timesteps)) as progress_bar:
                    for i, t in enumerate(dlfg_timesteps):
                        _SPA_STATE.spa_active = i < stage_spa_steps
                        decay = (stage_steps - i) / stage_steps
                        with self.transformer.cache_context("cond"):
                            noise_pred = self.transformer(
                                hidden_states=latents,
                                timestep=t.expand(latents.shape[0]).to(latents.dtype) / 1000,
                                guidance=stage_guidance,
                                pooled_projections=pooled_prompt_embeds,
                                encoder_hidden_states=prompt_embeds,
                                txt_ids=text_ids,
                                img_ids=image_ids,
                                joint_attention_kwargs=self.joint_attention_kwargs,
                                return_dict=False,
                            )[0]
                        latents, pred_x0 = self._flowmatch_step(
                            noise_pred,
                            t,
                            latents,
                            structure_on=True,
                            pred_x0_dict=pred_x0_dict,
                            height_dict=height_dict,
                            width_dict=width_dict,
                            batch_size=batch_size,
                            num_channels_latents=num_channels_latents,
                            target_height=stage_h,
                            target_width=stage_w,
                            filter_ratio=filter_ratio,
                            alpha=stage_alpha0 * decay,
                            beta=stage_beta0 * decay,
                        )
                        pred_x0_dict[t.item()] = pred_x0
                        height_dict[t.item()] = stage_h
                        width_dict[t.item()] = stage_w
                        progress_bar.update()

                cur_h, cur_w = stage_h, stage_w

            if output_type == "latent":
                image = latents
            else:
                latents = self._unpack_latents(latents, cur_h, cur_w, self.vae_scale_factor)
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)

            self.maybe_free_model_hooks()
        finally:
            _SPA_STATE.disarm()
            self.transformer.set_attn_processor(original_attn_processors)

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
