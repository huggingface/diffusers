import copy
from typing import Any

import torch

from ...models.autoencoders.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer
from ...models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from ...models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
from ...schedulers import UniPCMultistepScheduler
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Cosmos3OmniModularPipeline


logger = logging.get_logger(__name__)


class Cosmos3PrepareTextSegmentsStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Builds cond/uncond text segments and runtime device/dtype before denoising."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="cond_input_ids", required=True),
            InputParam(name="uncond_input_ids", required=True),
            InputParam(name="guidance_scale", type_hint=float, default=6.0),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("device"),
            OutputParam("dtype"),
            OutputParam("cond_text_segment"),
            OutputParam("uncond_text_segment"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        components._current_timestep = None
        components._interrupt = False
        components._guidance_scale = block_state.guidance_scale

        block_state.device = components._get_execution_device()
        block_state.dtype = components.transformer.dtype
        block_state.cond_text_segment = components._prepare_text_segment(
            block_state.cond_input_ids, device=block_state.device
        )
        block_state.uncond_text_segment = components._prepare_text_segment(
            block_state.uncond_input_ids, device=block_state.device
        )

        self.set_block_state(state, block_state)
        return components, state


class Cosmos3PrepareLatentsStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return (
            "Prepares vision/sound/action latents and conditioning masks. "
            "TODO: split VAE encoding into standalone Cosmos3VaeEncoderStep and Cosmos3ActionVaeEncoderStep."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", Cosmos3OmniTransformer),
            ComponentSpec("vae", AutoencoderKLWan),
            ComponentSpec("sound_tokenizer", Cosmos3AVAEAudioTokenizer),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="image", default=None),
            InputParam(name="video", default=None),
            InputParam(name="condition_frame_indexes_vision", default=(0, 1)),
            InputParam(name="condition_video_keep", default="first"),
            InputParam(name="num_frames", required=True),
            InputParam(name="height", required=True),
            InputParam(name="width", required=True),
            InputParam(name="fps", type_hint=float, default=24.0),
            InputParam(name="latents", default=None),
            InputParam(name="sound_latents", default=None),
            InputParam(name="action_latents", default=None),
            InputParam(name="generator", default=None),
            InputParam(name="enable_sound", type_hint=bool, default=False),
            InputParam(name="action", default=None),
            InputParam(name="device", required=True),
            InputParam(name="dtype", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents"),
            OutputParam("sound_latents"),
            OutputParam("action_latents"),
            OutputParam("fps_vision"),
            OutputParam("fps_sound"),
            OutputParam("vision_condition_mask"),
            OutputParam("sound_condition_mask"),
            OutputParam("action_condition_mask"),
            OutputParam("action_domain_id"),
            OutputParam("action_image_size"),
            OutputParam("raw_action_dim_resolved"),
            OutputParam("action_condition_frame_indexes"),
            OutputParam("vision_condition_indexes_for_pack"),
            OutputParam("has_image_condition"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        (
            block_state.latents,
            block_state.sound_latents,
            block_state.action_latents,
            block_state.fps_vision,
            block_state.fps_sound,
            block_state.vision_condition_mask,
            block_state.sound_condition_mask,
            block_state.action_condition_mask,
            block_state.action_domain_id,
            block_state.action_image_size,
            block_state.raw_action_dim_resolved,
            block_state.action_condition_frame_indexes,
        ) = components.prepare_latents(
            image=block_state.image,
            video=block_state.video,
            condition_frame_indexes_vision=block_state.condition_frame_indexes_vision,
            condition_video_keep=block_state.condition_video_keep,
            num_frames=block_state.num_frames,
            height=block_state.height,
            width=block_state.width,
            fps=block_state.fps,
            latents=block_state.latents,
            sound_latents=block_state.sound_latents,
            action_latents=block_state.action_latents,
            generator=block_state.generator,
            device=block_state.device,
            dtype=block_state.dtype,
            enable_sound=block_state.enable_sound,
            action=block_state.action,
        )

        vision_condition_indexes = torch.nonzero(
            block_state.vision_condition_mask[:, 0, 0] > 0, as_tuple=False
        ).flatten()
        block_state.vision_condition_indexes_for_pack = [int(idx.item()) for idx in vision_condition_indexes]
        block_state.has_image_condition = bool(block_state.vision_condition_indexes_for_pack)

        self.set_block_state(state, block_state)
        return components, state


class Cosmos3PackSequenceStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Builds static packed cond/uncond sequence metadata before denoising."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="cond_text_segment", required=True),
            InputParam(name="uncond_text_segment", required=True),
            InputParam(name="latents", required=True),
            InputParam(name="sound_latents", default=None),
            InputParam(name="action_latents", default=None),
            InputParam(name="fps_vision", required=True),
            InputParam(name="fps_sound", default=None),
            InputParam(name="has_image_condition", required=True),
            InputParam(name="vision_condition_indexes_for_pack", required=True),
            InputParam(name="action_condition_frame_indexes", default=None),
            InputParam(name="device", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("cond_packed_static"),
            OutputParam("uncond_packed_static"),
            OutputParam("num_noisy_vision_tokens"),
            OutputParam("sound_len"),
            OutputParam("action_noisy_len"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        cond_vision_segment = components._prepare_vision_segment(
            input_vision_tokens=block_state.latents,
            has_image_condition=block_state.has_image_condition,
            mrope_offset=block_state.cond_text_segment["vision_start_temporal_offset"],
            vision_fps=block_state.fps_vision,
            curr=block_state.cond_text_segment["und_len"],
            device=block_state.device,
            condition_frame_indexes=block_state.vision_condition_indexes_for_pack,
        )
        cond_sound_segment: dict[str, Any] = {}
        if block_state.sound_latents is not None:
            cond_sound_segment = components._prepare_sound_segment(
                input_sound_tokens=block_state.sound_latents,
                mrope_offset=block_state.cond_text_segment["vision_start_temporal_offset"],
                sound_fps=block_state.fps_sound,
                curr=block_state.cond_text_segment["und_len"] + cond_vision_segment["num_vision_tokens"],
                device=block_state.device,
            )
        cond_action_segment: dict[str, Any] = {}
        if block_state.action_latents is not None:
            cond_action_segment = components._prepare_action_segment(
                input_action_tokens=block_state.action_latents,
                condition_frame_indexes=block_state.action_condition_frame_indexes,
                mrope_offset=block_state.cond_text_segment["vision_start_temporal_offset"],
                action_fps=block_state.fps_vision,
                curr=block_state.cond_text_segment["und_len"]
                + cond_vision_segment["num_vision_tokens"]
                + cond_sound_segment.get("sound_len", 0),
                device=block_state.device,
            )
        cond_mrope_segments = [
            block_state.cond_text_segment["text_mrope_ids"],
            cond_vision_segment["vision_mrope_ids"],
        ]
        if cond_sound_segment:
            cond_mrope_segments.append(cond_sound_segment["sound_mrope_ids"])
        if cond_action_segment:
            cond_mrope_segments.append(cond_action_segment["action_mrope_ids"])
        block_state.cond_packed_static = {
            **block_state.cond_text_segment,
            **cond_vision_segment,
            **cond_sound_segment,
            **cond_action_segment,
            "position_ids": torch.cat(cond_mrope_segments, dim=1),
            "sequence_length": block_state.cond_text_segment["und_len"]
            + cond_vision_segment["num_vision_tokens"]
            + cond_sound_segment.get("sound_len", 0)
            + cond_action_segment.get("action_len", 0),
        }

        uncond_vision_segment = components._prepare_vision_segment(
            input_vision_tokens=block_state.latents,
            has_image_condition=block_state.has_image_condition,
            mrope_offset=block_state.uncond_text_segment["vision_start_temporal_offset"],
            vision_fps=block_state.fps_vision,
            curr=block_state.uncond_text_segment["und_len"],
            device=block_state.device,
            condition_frame_indexes=block_state.vision_condition_indexes_for_pack,
        )
        uncond_sound_segment: dict[str, Any] = {}
        if block_state.sound_latents is not None:
            uncond_sound_segment = components._prepare_sound_segment(
                input_sound_tokens=block_state.sound_latents,
                mrope_offset=block_state.uncond_text_segment["vision_start_temporal_offset"],
                sound_fps=block_state.fps_sound,
                curr=block_state.uncond_text_segment["und_len"] + uncond_vision_segment["num_vision_tokens"],
                device=block_state.device,
            )
        uncond_action_segment: dict[str, Any] = {}
        if block_state.action_latents is not None:
            uncond_action_segment = components._prepare_action_segment(
                input_action_tokens=block_state.action_latents,
                condition_frame_indexes=block_state.action_condition_frame_indexes,
                mrope_offset=block_state.uncond_text_segment["vision_start_temporal_offset"],
                action_fps=block_state.fps_vision,
                curr=block_state.uncond_text_segment["und_len"]
                + uncond_vision_segment["num_vision_tokens"]
                + uncond_sound_segment.get("sound_len", 0),
                device=block_state.device,
            )
        uncond_mrope_segments = [
            block_state.uncond_text_segment["text_mrope_ids"],
            uncond_vision_segment["vision_mrope_ids"],
        ]
        if uncond_sound_segment:
            uncond_mrope_segments.append(uncond_sound_segment["sound_mrope_ids"])
        if uncond_action_segment:
            uncond_mrope_segments.append(uncond_action_segment["action_mrope_ids"])
        block_state.uncond_packed_static = {
            **block_state.uncond_text_segment,
            **uncond_vision_segment,
            **uncond_sound_segment,
            **uncond_action_segment,
            "position_ids": torch.cat(uncond_mrope_segments, dim=1),
            "sequence_length": block_state.uncond_text_segment["und_len"]
            + uncond_vision_segment["num_vision_tokens"]
            + uncond_sound_segment.get("sound_len", 0)
            + uncond_action_segment.get("action_len", 0),
        }

        block_state.num_noisy_vision_tokens = cond_vision_segment["num_noisy_vision_tokens"]
        block_state.sound_len = cond_sound_segment.get("sound_len")
        block_state.action_noisy_len = cond_action_segment.get("num_noisy_action_tokens")

        self.set_block_state(state, block_state)
        return components, state


class Cosmos3SetTimestepsStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Initializes scheduler timesteps and modality schedulers."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", UniPCMultistepScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps", required=True),
            InputParam(name="device", required=True),
            InputParam(name="sound_latents", default=None),
            InputParam(name="action_latents", default=None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps"),
            OutputParam("sound_scheduler"),
            OutputParam("action_scheduler"),
            OutputParam("num_warmup_steps"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        components.scheduler.set_timesteps(block_state.num_inference_steps, device=block_state.device)

        block_state.timesteps = components.scheduler.timesteps
        block_state.sound_scheduler = (
            copy.deepcopy(components.scheduler) if block_state.sound_latents is not None else None
        )
        block_state.action_scheduler = (
            copy.deepcopy(components.scheduler) if block_state.action_latents is not None else None
        )
        block_state.num_warmup_steps = (
            len(block_state.timesteps) - block_state.num_inference_steps * components.scheduler.order
        )

        self.set_block_state(state, block_state)
        return components, state
