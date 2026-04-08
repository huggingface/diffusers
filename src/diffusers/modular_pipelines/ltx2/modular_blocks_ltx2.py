# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...utils import logging
from ..modular_pipeline import AutoPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, OutputParam
from .before_denoise import (
    LTX2DisableAdapterStep,
    LTX2EnableAdapterStep,
    LTX2InputStep,
    LTX2PrepareAudioLatentsStep,
    LTX2PrepareCoordinatesStep,
    LTX2PrepareLatentsStep,
    LTX2SetTimestepsStep,
    LTX2Stage2PrepareLatentsStep,
    LTX2Stage2SetTimestepsStep,
)
from .decoders import LTX2AudioDecoderStep, LTX2VideoDecoderStep
from .denoise import LTX2DenoiseLoopWrapper, LTX2DenoiseStep, LTX2LoopAfterDenoiser, LTX2LoopBeforeDenoiser, LTX2LoopDenoiser
from .encoders import LTX2ConditionEncoderStep, LTX2ConnectorStep, LTX2TextEncoderStep
from .modular_blocks_ltx2_upsample import LTX2UpsampleCoreBlocks


logger = logging.get_logger(__name__)


# ====================
# 1. AUTO CONDITION ENCODER (skip if no conditions)
# ====================


class LTX2AutoConditionEncoderStep(AutoPipelineBlocks):
    """Auto block that runs condition encoding when conditions or image inputs are provided.

    - When `conditions` is provided: runs condition encoder for arbitrary frame conditioning
    - When `image` is provided: runs condition encoder (converts image to condition at frame 0)
    - When neither is provided: step is skipped (T2V mode)
    """

    block_classes = [LTX2ConditionEncoderStep, LTX2ConditionEncoderStep]
    block_names = ["conditional_encoder", "image_encoder"]
    block_trigger_inputs = ["conditions", "image"]


# ====================
# 2. CORE DENOISE
# ====================


class LTX2CoreDenoiseStep(SequentialPipelineBlocks):
    """Core denoising block: input prep -> timesteps -> latents -> audio latents -> coordinates -> denoise loop."""

    model_name = "ltx2"
    block_classes = [
        LTX2InputStep,
        LTX2SetTimestepsStep,
        LTX2PrepareLatentsStep,
        LTX2PrepareAudioLatentsStep,
        LTX2PrepareCoordinatesStep,
        LTX2DenoiseStep,
    ]
    block_names = [
        "input",
        "set_timesteps",
        "prepare_latents",
        "prepare_audio_latents",
        "prepare_coordinates",
        "denoise",
    ]

    @property
    def description(self):
        return "Core denoise block that takes encoded conditions and runs the full denoising process."

    @property
    def outputs(self):
        return [
            OutputParam("latents"),
            OutputParam("audio_latents"),
        ]


# ====================
# 3. BLOCKS (T2V only)
# ====================


class LTX2Blocks(SequentialPipelineBlocks):
    """Modular pipeline blocks for LTX2 text-to-video generation."""

    model_name = "ltx2"
    block_classes = [
        LTX2TextEncoderStep,
        LTX2ConnectorStep,
        LTX2CoreDenoiseStep,
        LTX2VideoDecoderStep,
        LTX2AudioDecoderStep,
    ]
    block_names = ["text_encoder", "connector", "denoise", "video_decode", "audio_decode"]

    @property
    def description(self):
        return "Modular pipeline blocks for LTX2 text-to-video generation."

    @property
    def outputs(self):
        return [OutputParam("videos"), OutputParam("audio"), OutputParam("audio_sample_rate")]


# ====================
# 4. AUTO BLOCKS (T2V + I2V + Conditional)
# ====================


class LTX2AutoBlocks(SequentialPipelineBlocks):
    """Modular pipeline blocks for LTX2 with unified T2V, I2V, and conditional generation.

    Workflow map:
        - text2video: prompt only
        - image2video: image + prompt (auto-converts to condition at frame 0)
        - conditional: conditions + prompt (arbitrary frame conditioning)
    """

    model_name = "ltx2"
    block_classes = [
        LTX2TextEncoderStep,
        LTX2ConnectorStep,
        LTX2AutoConditionEncoderStep,
        LTX2CoreDenoiseStep,
        LTX2VideoDecoderStep,
        LTX2AudioDecoderStep,
    ]
    block_names = ["text_encoder", "connector", "condition_encoder", "denoise", "video_decode", "audio_decode"]

    _workflow_map = {
        "text2video": {"prompt": True},
        "image2video": {"image": True, "prompt": True},
        "conditional": {"conditions": True, "prompt": True},
    }

    @property
    def description(self):
        return (
            "Unified modular pipeline blocks for LTX2 supporting text-to-video, "
            "image-to-video, and conditional/FLF2V generation."
        )

    @property
    def outputs(self):
        return [OutputParam("videos"), OutputParam("audio")]


# ====================
# 5. STAGE 2 CORE DENOISE
# ====================


class LTX2Stage2CoreDenoiseStep(SequentialPipelineBlocks):
    """Core denoise for Stage 2: uses distilled sigmas with no dynamic shifting."""

    model_name = "ltx2"
    block_classes = [
        LTX2InputStep,
        LTX2Stage2SetTimestepsStep,
        LTX2Stage2PrepareLatentsStep,
        LTX2PrepareAudioLatentsStep,
        LTX2PrepareCoordinatesStep,
        LTX2DenoiseStep,
    ]
    block_names = [
        "input",
        "set_timesteps",
        "prepare_latents",
        "prepare_audio_latents",
        "prepare_coordinates",
        "denoise",
    ]

    @property
    def description(self):
        return "Stage 2 core denoise block using distilled sigmas and no dynamic shifting."

    @property
    def outputs(self):
        return [
            OutputParam("latents"),
            OutputParam("audio_latents"),
        ]


# ====================
# 6. STAGE 1 BLOCKS
# ====================


class LTX2Stage1Blocks(SequentialPipelineBlocks):
    """Stage 1 blocks: text encoding -> conditioning -> denoise -> latent output.

    Outputs latents and audio_latents for downstream processing (upsample + stage2).
    Supports T2V, I2V, and conditional generation modes.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2TextEncoderStep,
        LTX2ConnectorStep,
        LTX2AutoConditionEncoderStep,
        LTX2CoreDenoiseStep,
    ]
    block_names = ["text_encoder", "connector", "condition_encoder", "denoise"]

    _workflow_map = {
        "text2video": {"prompt": True},
        "image2video": {"image": True, "prompt": True},
        "conditional": {"conditions": True, "prompt": True},
    }

    @property
    def description(self):
        return (
            "Stage 1 modular pipeline blocks for LTX2: text encoding, conditioning, "
            "and denoising. Outputs latents for upsample + stage2 workflow."
        )

    @property
    def outputs(self):
        return [OutputParam("latents"), OutputParam("audio_latents")]


# ====================
# 7. STAGE 2 BLOCKS
# ====================


class LTX2Stage2Blocks(SequentialPipelineBlocks):
    """Stage 2 blocks: text encoding -> denoise (distilled) -> decode video + audio.

    Expects pre-computed latents (from upsample) and audio_latents (from stage1).
    Uses fixed distilled sigmas with no dynamic shifting and guidance_scale=1.0.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2TextEncoderStep,
        LTX2ConnectorStep,
        LTX2Stage2CoreDenoiseStep,
        LTX2VideoDecoderStep,
        LTX2AudioDecoderStep,
    ]
    block_names = ["text_encoder", "connector", "denoise", "video_decode", "audio_decode"]

    @property
    def description(self):
        return (
            "Stage 2 modular pipeline blocks for LTX2: re-encodes text, "
            "denoises with distilled sigmas, and decodes video + audio."
        )

    @property
    def expected_components(self):
        # Override guider default for stage2: guidance_scale=1.0 (no CFG)
        components = [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 1.0}),
                default_creation_method="from_config",
            ),
        ]
        for block in self.sub_blocks.values():
            for component in block.expected_components:
                if component not in components:
                    components.append(component)
        return components

    @property
    def outputs(self):
        return [OutputParam("videos"), OutputParam("audio")]


# ====================
# 8. STAGE 2 FULL DENOISE (uses stage2_guider)
# ====================


class LTX2Stage2FullDenoiseStep(LTX2DenoiseLoopWrapper):
    """Denoise step for Stage 2 within the full pipeline, using stage2_guider (guidance_scale=1.0)."""

    block_classes = [
        LTX2LoopBeforeDenoiser,
        LTX2LoopDenoiser(
            guider_config=FrozenDict({
                **LTX2LoopDenoiser._default_guider_config,
                "guidance_scale": 1.0,
                "audio_guidance_scale": 1.0,
                "modality_guidance_scale": 1.0,
            }),
        ),
        LTX2LoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Stage 2 denoise step using stage2_guider (guidance_scale=1.0).\n"
            "Used within LTX2FullPipelineBlocks to avoid conflict with the Stage 1 guider."
        )


# ====================
# 9. STAGE 2 FULL CORE DENOISE
# ====================


class LTX2Stage2FullCoreDenoiseStep(SequentialPipelineBlocks):
    """Core denoise for Stage 2 within the full pipeline: distilled sigmas, no dynamic shifting, stage2_guider."""

    model_name = "ltx2"
    block_classes = [
        LTX2InputStep,
        LTX2Stage2SetTimestepsStep,
        LTX2Stage2PrepareLatentsStep,
        LTX2PrepareAudioLatentsStep,
        LTX2PrepareCoordinatesStep,
        LTX2Stage2FullDenoiseStep,
    ]
    block_names = [
        "input",
        "set_timesteps",
        "prepare_latents",
        "prepare_audio_latents",
        "prepare_coordinates",
        "denoise",
    ]

    @property
    def description(self):
        return "Stage 2 core denoise for full pipeline: distilled sigmas, no dynamic shifting, stage2_guider."

    @property
    def outputs(self):
        return [
            OutputParam("latents"),
            OutputParam("audio_latents"),
        ]


# ====================
# 10. STAGE 2 INTERNAL BLOCKS (no text encoder/connector)
# ====================


class LTX2Stage2InternalBlocks(SequentialPipelineBlocks):
    """Stage 2 blocks without text encoder/connector — reads connector_* embeddings from state.

    Used within LTX2FullPipelineBlocks where Stage 1 already encoded text.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2Stage2FullCoreDenoiseStep,
        LTX2VideoDecoderStep,
        LTX2AudioDecoderStep,
    ]
    block_names = ["denoise", "video_decode", "audio_decode"]

    @property
    def description(self):
        return "Stage 2 internal blocks (no text encoding): denoise with stage2_guider + decode."

    @property
    def outputs(self):
        return [OutputParam("videos"), OutputParam("audio")]


# ====================
# 11. FULL PIPELINE BLOCKS (all-in-one)
# ====================


class LTX2FullPipelineBlocks(SequentialPipelineBlocks):
    """All-in-one mega-block: stage1 -> upsample -> stage2 in a single pipe() call.

    LoRA adapters are automatically disabled for stage1 and re-enabled for stage2.
    Uses two guiders: `guider` (guidance_scale=4.0) for stage1 and
    `stage2_guider` (guidance_scale=1.0) for stage2.

    Required components: text_encoder, tokenizer, transformer, connectors, vae, audio_vae,
    vocoder, scheduler, guider, stage2_guider, latent_upsampler.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2DisableAdapterStep,
        LTX2Stage1Blocks,
        LTX2UpsampleCoreBlocks,
        LTX2EnableAdapterStep,
        LTX2Stage2InternalBlocks,
    ]
    block_names = ["disable_lora", "stage1", "upsample", "enable_lora", "stage2"]

    _workflow_map = {
        "text2video": {"prompt": True},
        "image2video": {"image": True, "prompt": True},
        "conditional": {"conditions": True, "prompt": True},
    }

    @property
    def description(self):
        return (
            "All-in-one LTX2 pipeline: stage1 (denoise) -> upsample -> stage2 (distilled denoise + decode). "
            "LoRA adapters toggled automatically via stage_2_adapter parameter."
        )

    @property
    def outputs(self):
        return [OutputParam("videos"), OutputParam("audio")]
