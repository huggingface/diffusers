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

import torch

from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import OutputParam
from .before_denoise import (
    LTX2ConditionPrepareAudioLatentsStep,
    LTX2ConditionPrepareCoordsStep,
    LTX2ConditionSetTimestepsStep,
    LTX2DFRPlanStep,
    LTX2DFRPrepareLatentsStep,
    LTX2TextInputStep,
)
from .decoders import LTX2AudioDecoderStep, LTX2DFRSplitKeyframesStep, LTX2DiffusionVaeDecoderStep
from .denoise import LTX2ConditionDenoiseStep
from .modular_blocks_ltx2 import (
    LTX2AutoConditionEncoderStep,
    LTX2AutoDurationStep,
    LTX2AutoPromptEnhancerStep,
    LTX2TextConditioningStep,
)


# auto_docstring
class LTX2DFRCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "ltx2.5-dfr"
    block_classes = [
        LTX2TextInputStep,
        LTX2DFRPrepareLatentsStep,
        LTX2ConditionSetTimestepsStep,
        LTX2ConditionPrepareAudioLatentsStep,
        LTX2ConditionPrepareCoordsStep,
        LTX2ConditionDenoiseStep,
    ]
    block_names = [
        "input",
        "prepare_latents",
        "set_timesteps",
        "prepare_audio_latents",
        "prepare_coords",
        "denoise",
    ]

    @property
    def description(self):
        return (
            "Core denoise stage for one DFR pass. Identical to `LTX2ConditionCoreDenoiseStep` except for the "
            "prepare-latents block, which appends the generated keyframe slots and the optional spatial detailing "
            "reference. Everything downstream is unchanged: the slot marker rides to the transformer as a "
            "`denoiser_input_fields` output, and the slot coordinates ride in `appended_coords`."
        )


# auto_docstring
class LTX2DFRDecoderStep(SequentialPipelineBlocks):
    model_name = "ltx2.5-dfr"
    block_classes = [LTX2DFRSplitKeyframesStep, LTX2DiffusionVaeDecoderStep, LTX2AudioDecoderStep]
    block_names = ["split_keyframes", "video_decode", "audio_decode"]

    @property
    def description(self):
        return (
            "Decode stage for DFR: splits the generated keyframe slots out of the denoised sequence and trims the "
            "canvas padding, then denoises the video latents with the diffusion decoder and vocodes the audio "
            "latents (or returns latents)."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
            OutputParam(
                "keyframes_latents",
                type_hint=torch.Tensor,
                description=(
                    "Denormalized `[B, C, num_slots, H, W]` generated keyframe slots. Upsample these alongside the "
                    "video latents to seed the detailing pass."
                ),
            ),
        ]


# auto_docstring
class LTX2DFRBlocks(SequentialPipelineBlocks):
    model_name = "ltx2.5-dfr"
    block_classes = [
        LTX2AutoPromptEnhancerStep,
        LTX2TextConditioningStep,
        LTX2AutoDurationStep,
        LTX2DFRPlanStep,
        LTX2AutoConditionEncoderStep,
        LTX2DFRCoreDenoiseStep,
        LTX2DFRDecoderStep,
    ]
    block_names = [
        "prompt_enhancer",
        "text_encoder",
        "duration",
        "plan",
        "condition_encoder",
        "denoise",
        "decode",
    ]
    _workflow_map = {
        "text2video": {"prompt": True},
        "condition": {"conditions": True, "prompt": True},
    }

    @property
    def description(self):
        return (
            "Modular pipeline blocks for one LTX-2.5 Diffusion Fidelity Rendering (DFR) pass (joint video + audio).\n"
            "DFR generates on a canvas padded to a whole number of keyframe segments and spends one extra latent "
            "frame of tokens per segment border on a *keyframe slot*: a single-pixel-frame latent the model fills "
            "in. Relaxing the effective temporal compression at those positions means the surrounding video is "
            "conditioned on genuinely new frames rather than interpolated ones.\n"
            "The full recipe is two passes of these blocks. The first runs at half the target resolution and "
            "returns `videos`/`keyframes_latents` as latents; upsample both with `LTX2LatentUpsamplePipeline`, load "
            "the spatial detailing IC-LoRA, and run the blocks again at full resolution with `latents`, "
            "`keyframes_latents` and `detailing_reference_latents` supplied. Needs a transformer whose config sets "
            "`use_keyframes_abs_pos_embedding`, which LTX-2.5 checkpoints ship."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
            OutputParam(
                "keyframes_latents",
                type_hint=torch.Tensor,
                description=(
                    "Denormalized `[B, C, num_slots, H, W]` generated keyframe slots. Upsample these alongside the "
                    "video latents to seed the detailing pass."
                ),
            ),
        ]
