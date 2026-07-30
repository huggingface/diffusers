# Copyright 2026 The HuggingFace Team. All rights reserved.
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

# Two-guider design for LTX-2.X: a single-modality guider (`LTX2Guidance`, instantiated once as the video `guider`
# and once as the `audio_guider`) plus the denoiser-side union-plan helper. Wired into `LTX2LoopDenoiser`.
#
# Execution mirrors the standard `LTX2Pipeline`: `plan_guidance_passes` batches the cond+uncond CFG pair into a
# single transformer forward (`torch.cat([latents] * 2)` + `.chunk(2)`) and runs STG / modality-isolation as
# separate single-batch conditional forwards, matching the reference op-for-op (batch sizes, coords, cache
# contexts). This gives bitwise parity with the standard pipeline in fp32 -- verified at full-checkpoint scale,
# including under CPU offload.
#
# bf16 note. An earlier prototype ran *every* pass single-batch (no CFG concatenation). It was mathematically
# equivalent (fp32 still matched) but, because GPU matmul is not batch-invariant, left `cond` computed alone
# differing from `cond` computed inside a batch-of-2 -- negligible in fp32 (~1e-6/op) but ~1e-2/op in bf16, which
# the CFG delta and sampler amplified to ~10% mean-relative latent divergence. Batching the CFG pair removes that
# specific gap. A smaller bf16 gap remains (~1% at tiny scale, ~5% at full), and it is NOT a logic difference: fp32
# is bitwise across scale and with/without offload, so only the dtype changes the result. It is a bf16-kernel
# effect -- bf16's 7-bit mantissa (~1e4x coarser than fp32) makes non-associative accumulation-order differences
# visible, and bf16 uses different kernels than fp32 (layout-sensitive tensor-core GEMM algorithm selection, fused
# attention backends), so graphs that are byte-identical in fp32 can still dispatch to slightly different bf16
# kernels. Parity is therefore gated on fp32 (bitwise); the bf16 comparison is a close-but-not-bitwise check.

import math

import torch

from ...configuration_utils import register_to_config
from ...guiders.guider_utils import BaseGuidance, GuiderOutput, rescale_noise_cfg
from ..modular_pipeline import BlockState


class LTX2Guidance(BaseGuidance):
    """
    Single-modality guider for LTX-2.X. Combines up to three terms via the delta formulation:
      - classifier-free guidance (CFG),
      - spatio-temporal guidance (STG), an extra pass with a set of transformer blocks perturbed,
      - modality-isolation guidance, an extra pass with A2V/V2A cross-attention disabled.

    Instantiated once per modality (`guider` for video, `audio_guider` for audio) with that modality's scales;
    that keeps the per-modality scales in independent, fully-defined component configs. The combine is done in
    whatever space the caller feeds it — the LTX-2 denoiser converts velocity->x0 before calling this and back
    afterwards, so `forward` operates on x0 predictions.
    """

    # `pred_cond` is the base; the others are the guidance passes.
    _input_predictions = ["pred_cond", "pred_uncond", "pred_cond_stg", "pred_cond_modality"]

    @register_to_config
    def __init__(
        self,
        guidance_scale: float = 1.0,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        # STG perturbs whole transformer blocks, and one forward feeds *both* modalities, so the block list is a
        # shared/transformer-level knob rather than a per-modality one; only the video guider carries it (the audio
        # guider reuses the same STG forward via its own `stg_scale`). See `plan_guidance_passes`.
        spatio_temporal_guidance_blocks: list[int] | None = None,
        start: float = 0.0,
        stop: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(start, stop, enabled)
        self.guidance_scale = guidance_scale
        self.stg_scale = stg_scale
        self.modality_scale = modality_scale
        self.guidance_rescale = guidance_rescale
        self.spatio_temporal_guidance_blocks = spatio_temporal_guidance_blocks

    # --- per-term enabled checks (public: the denoiser ORs these across the two guiders) --------------------
    def _within_window(self) -> bool:
        if self._num_inference_steps is None:
            return True
        start_step = int(self._start * self._num_inference_steps)
        stop_step = int(self._stop * self._num_inference_steps)
        return start_step <= self._step < stop_step

    def is_cfg_enabled(self) -> bool:
        return self._enabled and self._within_window() and not math.isclose(self.guidance_scale, 1.0)

    def is_stg_enabled(self) -> bool:
        return self._enabled and self._within_window() and not math.isclose(self.stg_scale, 0.0)

    def is_modality_enabled(self) -> bool:
        return self._enabled and self._within_window() and not math.isclose(self.modality_scale, 1.0)

    def active_predictions(self) -> list[str]:
        preds = ["pred_cond"]
        if self.is_cfg_enabled():
            preds.append("pred_uncond")
        if self.is_stg_enabled():
            preds.append("pred_cond_stg")
        if self.is_modality_enabled():
            preds.append("pred_cond_modality")
        return preds

    @property
    def num_conditions(self) -> int:
        return len(self.active_predictions())

    @property
    def is_conditional(self) -> bool:
        # Not load-bearing here: the denoiser owns pass execution and carries the STG/modality flags on the
        # batches, so we never drive `prepare_models`/hooks off `_count_prepared` the way SkipLayerGuidance does.
        return True

    def prepare_inputs_from_block_state(
        self, data: BlockState, input_fields: dict[str, str | tuple[str, str]]
    ) -> list[BlockState]:
        # Build one identifier-tagged batch per active pass. `tuple_index` selects conditional(0) vs
        # unconditional(1) inputs; STG/modality use the conditional inputs (index 0). `input_fields` may be empty
        # in the two-guider flow (the denoiser assembles the shared conditioning itself and only fills
        # `noise_pred` per batch); it is honored here so the guider can also carry conditioning if desired.
        index_by_pred = {"pred_cond": 0, "pred_uncond": 1, "pred_cond_stg": 0, "pred_cond_modality": 0}
        return [
            self._prepare_batch_from_block_state(input_fields, data, index_by_pred[pred], pred)
            for pred in self.active_predictions()
        ]

    def forward(
        self,
        pred_cond: torch.Tensor,
        pred_uncond: torch.Tensor | None = None,
        pred_cond_stg: torch.Tensor | None = None,
        pred_cond_modality: torch.Tensor | None = None,
    ) -> GuiderOutput:
        # Delta formulation (matches the reference LTX-2 combine, applied in the x0 space the denoiser feeds):
        #   x0_cond + (gs-1)*(cond-uncond) + stg*(cond-stg) + (mod-1)*(cond-modality), then optional rescale.
        pred = pred_cond
        if self.is_cfg_enabled():
            pred = pred + (self.guidance_scale - 1.0) * (pred_cond - pred_uncond)
        if self.is_stg_enabled():
            pred = pred + self.stg_scale * (pred_cond - pred_cond_stg)
        if self.is_modality_enabled():
            pred = pred + (self.modality_scale - 1.0) * (pred_cond - pred_cond_modality)
        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)
        return GuiderOutput(pred=pred, pred_cond=pred_cond, pred_uncond=pred_uncond)


def plan_guidance_passes(video_guider: LTX2Guidance, audio_guider: LTX2Guidance) -> list[dict]:
    """
    Denoiser-owned union plan: the transformer forwards needed this step, unioned across the two guiders and
    grouped to mirror the standard `LTX2Pipeline` execution.

    Both guiders share one transformer per forward (it emits both modalities), so the set of active passes is the
    union of each guider's `active_predictions()`. A guider that doesn't want a term simply omits it from its own
    combine, but the forward still runs if the *other* modality wants it. The passes are grouped into forwards
    exactly as the reference pipeline does:
      - one forward for the CFG pair, batching `[uncond, cond]` (`torch.cat([latents] * 2)` + `.chunk(2)`) when
        either modality wants CFG, else a single conditional forward for `pred_cond` alone;
      - one single-batch conditional forward each for STG and modality-isolation, carrying that pass's transformer
        flag (`spatio_temporal_guidance_blocks` / `isolate_modalities`).

    Each entry is `{"identifiers", "conditioning", "flags", "cache_context"}`:
      - `identifiers` / `conditioning`: aligned lists — the pass identifier and the text conditioning ("cond" or
        "uncond") for each batch element, in the order the forward's output is chunked;
      - `flags`: the transformer kwargs for the forward (`isolate_modalities`, `spatio_temporal_guidance_blocks`);
      - `cache_context`: the transformer cache-context name, matching the reference pipeline.
    """
    active = set(video_guider.active_predictions()) | set(audio_guider.active_predictions())

    # STG blocks are shared (one perturbed forward feeds both modalities); take them from whichever guider
    # requested STG, video first. If the two disagreed there would be no single forward that satisfies both --
    # a wrinkle of the two-guider design (a single dual-modality guider would own this unambiguously).
    stg_blocks = None
    for guider in (video_guider, audio_guider):
        if guider.is_stg_enabled():
            stg_blocks = guider.spatio_temporal_guidance_blocks
            break

    # CFG forward: `pred_cond` is always present (it is the base for every delta); `pred_uncond` joins the same
    # forward -- batched as [uncond, cond] -- iff either modality wants CFG.
    cfg_identifiers = ["pred_uncond", "pred_cond"] if "pred_uncond" in active else ["pred_cond"]
    passes = [
        {
            "identifiers": cfg_identifiers,
            "conditioning": ["uncond" if pred == "pred_uncond" else "cond" for pred in cfg_identifiers],
            "flags": {"isolate_modalities": False, "spatio_temporal_guidance_blocks": None},
            "cache_context": "cond_uncond",
        }
    ]
    if "pred_cond_stg" in active:
        passes.append(
            {
                "identifiers": ["pred_cond_stg"],
                "conditioning": ["cond"],
                "flags": {"isolate_modalities": False, "spatio_temporal_guidance_blocks": stg_blocks},
                "cache_context": "uncond_stg",
            }
        )
    if "pred_cond_modality" in active:
        passes.append(
            {
                "identifiers": ["pred_cond_modality"],
                "conditioning": ["cond"],
                "flags": {"isolate_modalities": True, "spatio_temporal_guidance_blocks": None},
                "cache_context": "uncond_modality",
            }
        )
    return passes
