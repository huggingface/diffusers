import copy

import torch

from ...models.autoencoders.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer
from ...models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from ...models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
from ...pipelines.cosmos.pipeline_cosmos3_omni import _EMBODIMENT_TO_DOMAIN_ID
from ...schedulers import UniPCMultistepScheduler
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Cosmos3OmniModularPipeline


class Cosmos3PrepareTextSegmentsStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Builds cond/uncond text segments before denoising."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="cond_input_ids", required=True),
            InputParam(name="uncond_input_ids", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("cond_text_segment"),
            OutputParam("uncond_text_segment"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        block_state.cond_text_segment = components._prepare_text_segment(block_state.cond_input_ids, device=device)
        block_state.uncond_text_segment = components._prepare_text_segment(block_state.uncond_input_ids, device=device)
        self.set_block_state(state, block_state)
        return components, state


class Cosmos3VisionPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Prepares noisy vision latents and the vision conditioning mask."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", Cosmos3OmniTransformer),
            ComponentSpec("vae", AutoencoderKLWan),
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
            InputParam(name="generator", default=None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents"),
            OutputParam("fps_vision"),
            OutputParam("vision_condition_mask"),
            OutputParam("vision_condition_indexes_for_pack"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        dtype = components.transformer.dtype

        x0_tokens_vision = block_state.x0_tokens_vision
        if x0_tokens_vision is None:
            if block_state.num_frames < 1:
                raise ValueError(f"num_frames must be >= 1, got {block_state.num_frames}.")
            sf_spatial = int(components.vae.config.scale_factor_spatial)
            if block_state.height % sf_spatial != 0 or block_state.width % sf_spatial != 0:
                raise ValueError(
                    f"height and width must be multiples of {sf_spatial}, got ({block_state.height}, {block_state.width})."
                )
            vision_tensor = torch.zeros(
                1,
                3,
                block_state.num_frames,
                block_state.height,
                block_state.width,
                device=device,
                dtype=dtype,
            )
            x0_tokens_vision = components._encode_video(vision_tensor).contiguous().float()
        else:
            x0_tokens_vision = x0_tokens_vision.to(device=device, dtype=torch.float32)

        block_state.fps_vision = float(block_state.fps)
        condition_frames = block_state.vision_condition_frames or []
        block_state.vision_condition_mask = torch.zeros((x0_tokens_vision.shape[2], 1, 1), device=device, dtype=dtype)
        for frame_idx in condition_frames:
            if 0 <= frame_idx < block_state.vision_condition_mask.shape[0]:
                block_state.vision_condition_mask[frame_idx, 0, 0] = 1.0

        if block_state.latents is None:
            pure_noise = randn_tensor(
                tuple(x0_tokens_vision.shape), generator=block_state.generator, device=device, dtype=dtype
            )
            block_state.latents = (
                block_state.vision_condition_mask * x0_tokens_vision.to(device=device, dtype=dtype)
                + (1.0 - block_state.vision_condition_mask) * pure_noise
            )
        else:
            block_state.latents = block_state.latents.to(device=device, dtype=dtype)

        vision_condition_indexes = torch.nonzero(
            block_state.vision_condition_mask[:, 0, 0] > 0, as_tuple=False
        ).flatten()
        block_state.vision_condition_indexes_for_pack = [int(idx.item()) for idx in vision_condition_indexes]

        self.set_block_state(state, block_state)
        return components, state


class Cosmos3SoundPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Prepares noisy sound latents and the sound conditioning mask."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", Cosmos3OmniTransformer),
            ComponentSpec("sound_tokenizer", Cosmos3AVAEAudioTokenizer),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="num_frames", required=True),
            InputParam(name="fps", type_hint=float, default=24.0),
            InputParam(name="sound_latents", default=None),
            InputParam(name="generator", default=None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("sound_latents"),
            OutputParam("fps_sound"),
            OutputParam("sound_condition_mask"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        dtype = components.transformer.dtype

        if components.sound_tokenizer is None:
            raise ValueError("Sound generation requires a sound-capable checkpoint with a sound_tokenizer.")
        if not components.transformer.config.sound_gen:
            raise ValueError("Sound generation requires a transformer trained with sound_gen=True.")

        sound_dim = components.transformer.config.sound_dim
        block_state.fps_sound = float(components.transformer.config.sound_latent_fps)
        n_audio_samples = int(
            block_state.num_frames / block_state.fps * components.sound_tokenizer.config.sampling_rate
        )
        hop_size = components.sound_tokenizer._hop_size
        t_sound = (n_audio_samples + hop_size - 1) // hop_size
        x0_tokens_sound = torch.zeros(sound_dim, t_sound, device=device, dtype=dtype)
        block_state.sound_condition_mask = torch.zeros((x0_tokens_sound.shape[1], 1), device=device, dtype=dtype)

        if block_state.sound_latents is None:
            pure_noise = randn_tensor(
                tuple(x0_tokens_sound.shape), generator=block_state.generator, device=device, dtype=dtype
            )
            block_state.sound_latents = (
                block_state.sound_condition_mask.T * x0_tokens_sound
                + (1.0 - block_state.sound_condition_mask.T) * pure_noise
            )
        else:
            block_state.sound_latents = block_state.sound_latents.to(device=device, dtype=dtype)

        self.set_block_state(state, block_state)
        return components, state


class Cosmos3ActionPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Prepares noisy action latents and the action conditioning mask."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="action", required=True),
            InputParam(name="action_condition_frame_indexes", default=None),
            InputParam(name="action_latents", default=None),
            InputParam(name="generator", default=None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("action_latents"),
            OutputParam("action_condition_mask"),
            OutputParam("action_domain_id"),
            OutputParam("raw_action_dim_resolved"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        dtype = components.transformer.dtype
        action = block_state.action

        if not components.transformer.config.action_gen:
            raise ValueError("action requires a transformer trained with action_gen=True.")

        block_state.raw_action_dim_resolved = int(action.raw_action_dim) if action.raw_action_dim is not None else None
        if (
            block_state.raw_action_dim_resolved is not None
            and block_state.raw_action_dim_resolved > components.transformer.config.action_dim
        ):
            raise ValueError(
                f"raw_action_dim={block_state.raw_action_dim_resolved} exceeds the model action_dim="
                f"{components.transformer.config.action_dim}."
            )

        action_chunk_size = action.chunk_size
        action_dim = components.transformer.action_dim
        if action.mode == "forward_dynamics":
            raw_actions = action.raw_actions
            if raw_actions is None:
                raise ValueError("action_mode='forward_dynamics' requires an action tensor.")
            raw_actions = raw_actions.to(device=device, dtype=dtype)
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
            x0_tokens_action = torch.zeros(action_chunk_size, action_dim, device=device, dtype=dtype)

        if action.domain_name not in _EMBODIMENT_TO_DOMAIN_ID:
            raise ValueError(
                f"Unknown Cosmos3 action domain_name={action.domain_name!r}; expected one of {sorted(_EMBODIMENT_TO_DOMAIN_ID)}."
            )
        block_state.action_domain_id = torch.tensor(
            [_EMBODIMENT_TO_DOMAIN_ID[action.domain_name]], dtype=torch.long, device=device
        )
        condition_frames = block_state.action_condition_frame_indexes or []
        block_state.action_condition_mask = torch.zeros((x0_tokens_action.shape[0], 1), device=device, dtype=dtype)
        for frame_idx in condition_frames:
            if 0 <= frame_idx < block_state.action_condition_mask.shape[0]:
                block_state.action_condition_mask[frame_idx, 0] = 1.0

        if block_state.action_latents is None:
            pure_noise = randn_tensor(
                tuple(x0_tokens_action.shape), generator=block_state.generator, device=device, dtype=dtype
            )
            block_state.action_latents = (
                block_state.action_condition_mask * x0_tokens_action
                + (1.0 - block_state.action_condition_mask) * pure_noise
            )
            if block_state.raw_action_dim_resolved is not None:
                block_state.action_latents[:, block_state.raw_action_dim_resolved :] = 0
        else:
            block_state.action_latents = block_state.action_latents.to(device=device, dtype=dtype)

        self.set_block_state(state, block_state)
        return components, state


class Cosmos3VisionPackSequenceStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Builds separate cond/uncond vision sequence segments."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="cond_text_segment", required=True),
            InputParam(name="uncond_text_segment", required=True),
            InputParam(name="latents", required=True),
            InputParam(name="fps_vision", required=True),
            InputParam(name="vision_condition_indexes_for_pack", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("cond_vision_segment"),
            OutputParam("uncond_vision_segment"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        has_image_condition = bool(block_state.vision_condition_indexes_for_pack)

        block_state.cond_vision_segment = components._prepare_vision_segment(
            input_vision_tokens=block_state.latents,
            has_image_condition=has_image_condition,
            mrope_offset=block_state.cond_text_segment["vision_start_temporal_offset"],
            vision_fps=block_state.fps_vision,
            curr=block_state.cond_text_segment["und_len"],
            device=device,
            condition_frame_indexes=block_state.vision_condition_indexes_for_pack,
        )
        block_state.uncond_vision_segment = components._prepare_vision_segment(
            input_vision_tokens=block_state.latents,
            has_image_condition=has_image_condition,
            mrope_offset=block_state.uncond_text_segment["vision_start_temporal_offset"],
            vision_fps=block_state.fps_vision,
            curr=block_state.uncond_text_segment["und_len"],
            device=device,
            condition_frame_indexes=block_state.vision_condition_indexes_for_pack,
        )

        self.set_block_state(state, block_state)
        return components, state


class Cosmos3SoundPackSequenceStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Builds separate cond/uncond sound sequence segments."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="cond_text_segment", required=True),
            InputParam(name="uncond_text_segment", required=True),
            InputParam(name="cond_vision_segment", required=True),
            InputParam(name="uncond_vision_segment", required=True),
            InputParam(name="sound_latents", required=True),
            InputParam(name="fps_sound", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("cond_sound_segment"),
            OutputParam("uncond_sound_segment"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        block_state.cond_sound_segment = components._prepare_sound_segment(
            input_sound_tokens=block_state.sound_latents,
            mrope_offset=block_state.cond_text_segment["vision_start_temporal_offset"],
            sound_fps=block_state.fps_sound,
            curr=block_state.cond_text_segment["und_len"] + block_state.cond_vision_segment["num_vision_tokens"],
            device=device,
        )
        block_state.uncond_sound_segment = components._prepare_sound_segment(
            input_sound_tokens=block_state.sound_latents,
            mrope_offset=block_state.uncond_text_segment["vision_start_temporal_offset"],
            sound_fps=block_state.fps_sound,
            curr=block_state.uncond_text_segment["und_len"] + block_state.uncond_vision_segment["num_vision_tokens"],
            device=device,
        )

        self.set_block_state(state, block_state)
        return components, state


class Cosmos3ActionPackSequenceStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Builds separate cond/uncond action sequence segments."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="cond_text_segment", required=True),
            InputParam(name="uncond_text_segment", required=True),
            InputParam(name="cond_vision_segment", required=True),
            InputParam(name="uncond_vision_segment", required=True),
            InputParam(name="action_latents", required=True),
            InputParam(name="action_condition_frame_indexes", default=None),
            InputParam(name="fps_vision", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("cond_action_segment"),
            OutputParam("uncond_action_segment"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        block_state.cond_action_segment = components._prepare_action_segment(
            input_action_tokens=block_state.action_latents,
            condition_frame_indexes=block_state.action_condition_frame_indexes,
            mrope_offset=block_state.cond_text_segment["vision_start_temporal_offset"],
            action_fps=block_state.fps_vision,
            curr=block_state.cond_text_segment["und_len"] + block_state.cond_vision_segment["num_vision_tokens"],
            device=device,
        )
        block_state.uncond_action_segment = components._prepare_action_segment(
            input_action_tokens=block_state.action_latents,
            condition_frame_indexes=block_state.action_condition_frame_indexes,
            mrope_offset=block_state.uncond_text_segment["vision_start_temporal_offset"],
            action_fps=block_state.fps_vision,
            curr=block_state.uncond_text_segment["und_len"] + block_state.uncond_vision_segment["num_vision_tokens"],
            device=device,
        )

        self.set_block_state(state, block_state)
        return components, state


class Cosmos3SoundActionPackSequenceStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Builds separate cond/uncond action sequence segments after sound segments."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="cond_text_segment", required=True),
            InputParam(name="uncond_text_segment", required=True),
            InputParam(name="cond_vision_segment", required=True),
            InputParam(name="uncond_vision_segment", required=True),
            InputParam(name="cond_sound_segment", required=True),
            InputParam(name="uncond_sound_segment", required=True),
            InputParam(name="action_latents", required=True),
            InputParam(name="action_condition_frame_indexes", default=None),
            InputParam(name="fps_vision", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("cond_action_segment"),
            OutputParam("uncond_action_segment"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        block_state.cond_action_segment = components._prepare_action_segment(
            input_action_tokens=block_state.action_latents,
            condition_frame_indexes=block_state.action_condition_frame_indexes,
            mrope_offset=block_state.cond_text_segment["vision_start_temporal_offset"],
            action_fps=block_state.fps_vision,
            curr=block_state.cond_text_segment["und_len"]
            + block_state.cond_vision_segment["num_vision_tokens"]
            + block_state.cond_sound_segment["sound_len"],
            device=device,
        )
        block_state.uncond_action_segment = components._prepare_action_segment(
            input_action_tokens=block_state.action_latents,
            condition_frame_indexes=block_state.action_condition_frame_indexes,
            mrope_offset=block_state.uncond_text_segment["vision_start_temporal_offset"],
            action_fps=block_state.fps_vision,
            curr=block_state.uncond_text_segment["und_len"]
            + block_state.uncond_vision_segment["num_vision_tokens"]
            + block_state.uncond_sound_segment["sound_len"],
            device=device,
        )

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
        device = components._execution_device
        components.scheduler.set_timesteps(block_state.num_inference_steps, device=device)
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
