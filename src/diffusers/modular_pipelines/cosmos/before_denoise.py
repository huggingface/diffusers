import copy
from typing import Any

import torch

from ...models.autoencoders.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer
from ...models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from ...models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
from ...pipelines.cosmos.pipeline_cosmos3_omni import _EMBODIMENT_TO_DOMAIN_ID
from ...schedulers import UniPCMultistepScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
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
        return "Prepares noisy vision/sound/action latents and conditioning masks from pre-encoded vision tokens."

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
            InputParam(name="x0_tokens_vision", default=None),
            InputParam(name="vision_condition_frames", default=None),
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
            InputParam(name="action_image_size", default=None),
            InputParam(name="action_condition_frame_indexes", default=None),
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

        action = block_state.action
        action_mode = action.mode if action is not None else None
        x0_tokens_vision = block_state.x0_tokens_vision
        if x0_tokens_vision is None:
            if block_state.num_frames < 1:
                raise ValueError(f"`num_frames` must be >= 1, got {block_state.num_frames}.")
            sf_spatial = int(components.vae.config.scale_factor_spatial)
            if block_state.height % sf_spatial != 0 or block_state.width % sf_spatial != 0:
                raise ValueError(
                    f"`height` and `width` must be multiples of {sf_spatial}, got ({block_state.height}, {block_state.width})."
                )

            # Match the monolithic prepare_latents text-only path exactly:
            # build a zero vision clip and run VAE encode once to get canonical latent shape/layout.
            vision_tensor = torch.zeros(
                1,
                3,
                block_state.num_frames,
                block_state.height,
                block_state.width,
                device=block_state.device,
                dtype=block_state.dtype,
            )
            x0_tokens_vision = components._encode_video(vision_tensor).contiguous().float()
        else:
            x0_tokens_vision = x0_tokens_vision.to(device=block_state.device, dtype=torch.float32)
        vision_shape = tuple(x0_tokens_vision.shape)

        block_state.raw_action_dim_resolved = (
            int(action.raw_action_dim) if action is not None and action.raw_action_dim is not None else None
        )
        if (
            block_state.raw_action_dim_resolved is not None
            and block_state.raw_action_dim_resolved > components.transformer.config.action_dim
        ):
            raise ValueError(
                f"raw_action_dim={block_state.raw_action_dim_resolved} exceeds the model's trained action_dim="
                f"{components.transformer.config.action_dim}; this checkpoint cannot represent that action width."
            )

        block_state.fps_vision = float(block_state.fps)

        condition_frames_vision = block_state.vision_condition_frames or []
        vision_condition_mask = torch.zeros(
            (x0_tokens_vision.shape[2], 1, 1), device=block_state.device, dtype=block_state.dtype
        )
        for frame_idx in condition_frames_vision:
            if 0 <= frame_idx < vision_condition_mask.shape[0]:
                vision_condition_mask[frame_idx, 0, 0] = 1.0
        block_state.vision_condition_mask = vision_condition_mask

        if block_state.latents is None:
            pure_noise = randn_tensor(
                vision_shape, generator=block_state.generator, device=block_state.device, dtype=block_state.dtype
            )
            block_state.latents = (
                vision_condition_mask * x0_tokens_vision.to(device=block_state.device, dtype=block_state.dtype)
                + (1.0 - vision_condition_mask) * pure_noise
            )
        else:
            block_state.latents = block_state.latents.to(device=block_state.device, dtype=block_state.dtype)

        block_state.fps_sound = None
        block_state.sound_condition_mask = None
        if block_state.enable_sound:
            if getattr(components, "sound_tokenizer", None) is None:
                raise ValueError("`enable_sound=True` requires a sound-capable checkpoint with a `sound_tokenizer`.")
            if not getattr(components.transformer.config, "sound_gen", False):
                raise ValueError("`enable_sound=True` but the transformer was not trained with `sound_gen=True`.")
            sound_dim = components.transformer.config.sound_dim
            block_state.fps_sound = float(components.transformer.config.sound_latent_fps)
            n_audio_samples = int(
                block_state.num_frames / block_state.fps * components.sound_tokenizer.config.sampling_rate
            )
            hop_size = components.sound_tokenizer._hop_size
            t_sound = (n_audio_samples + hop_size - 1) // hop_size
            x0_tokens_sound = torch.zeros(sound_dim, t_sound, device=block_state.device, dtype=block_state.dtype)
            block_state.sound_condition_mask = torch.zeros(
                (x0_tokens_sound.shape[1], 1), device=block_state.device, dtype=block_state.dtype
            )
            if block_state.sound_latents is None:
                pure_noise_sound = randn_tensor(
                    tuple(x0_tokens_sound.shape),
                    generator=block_state.generator,
                    device=block_state.device,
                    dtype=block_state.dtype,
                )
                block_state.sound_latents = (
                    block_state.sound_condition_mask.T * x0_tokens_sound
                    + (1.0 - block_state.sound_condition_mask.T) * pure_noise_sound
                )
            else:
                block_state.sound_latents = block_state.sound_latents.to(
                    device=block_state.device, dtype=block_state.dtype
                )

        block_state.action_condition_mask = None
        block_state.action_domain_id = None
        if action_mode is not None:
            action_chunk_size = action.chunk_size
            action_dim = components.transformer.action_dim
            if action_mode == "forward_dynamics":
                raw_actions = action.raw_actions
                if raw_actions is None:
                    raise ValueError("action_mode='forward_dynamics' requires an action tensor.")
                raw_actions = raw_actions.to(device=block_state.device, dtype=block_state.dtype)
                if raw_actions.shape[-1] > action_dim:
                    raise ValueError(
                        f"Cosmos3 action dimension {raw_actions.shape[-1]} exceeds model action_dim={action_dim}."
                    )
                if raw_actions.shape[0] < action_chunk_size:
                    raw_actions = torch.cat(
                        [raw_actions, raw_actions[-1:].expand(action_chunk_size - raw_actions.shape[0], -1)],
                        dim=0,
                    )
                raw_actions = raw_actions[:action_chunk_size]
                if raw_actions.shape[-1] < action_dim:
                    action_padding = torch.zeros(
                        raw_actions.shape[0],
                        action_dim - raw_actions.shape[-1],
                        dtype=raw_actions.dtype,
                        device=raw_actions.device,
                    )
                    raw_actions = torch.cat([raw_actions, action_padding], dim=-1)
                x0_tokens_action = raw_actions
            else:
                x0_tokens_action = torch.zeros(
                    action_chunk_size, action_dim, device=block_state.device, dtype=block_state.dtype
                )

            if action.domain_name not in _EMBODIMENT_TO_DOMAIN_ID:
                raise ValueError(
                    f"Unknown Cosmos3 action domain_name={action.domain_name!r}; expected one of {sorted(_EMBODIMENT_TO_DOMAIN_ID)}."
                )
            block_state.action_domain_id = torch.tensor(
                [_EMBODIMENT_TO_DOMAIN_ID[action.domain_name]],
                dtype=torch.long,
                device=block_state.device,
            )
            condition_frames = block_state.action_condition_frame_indexes or []
            block_state.action_condition_mask = torch.zeros(
                (x0_tokens_action.shape[0], 1), device=block_state.device, dtype=block_state.dtype
            )
            for frame_idx in condition_frames:
                if 0 <= frame_idx < block_state.action_condition_mask.shape[0]:
                    block_state.action_condition_mask[frame_idx, 0] = 1.0
            if block_state.action_latents is None:
                pure_noise_action = randn_tensor(
                    tuple(x0_tokens_action.shape),
                    generator=block_state.generator,
                    device=block_state.device,
                    dtype=block_state.dtype,
                )
                block_state.action_latents = (
                    block_state.action_condition_mask * x0_tokens_action
                    + (1.0 - block_state.action_condition_mask) * pure_noise_action
                )
                if block_state.raw_action_dim_resolved is not None:
                    block_state.action_latents[:, block_state.raw_action_dim_resolved :] = 0
            else:
                block_state.action_latents = block_state.action_latents.to(
                    device=block_state.device, dtype=block_state.dtype
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
