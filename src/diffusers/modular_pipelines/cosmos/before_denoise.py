import copy

import torch

from ...models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
from ...pipelines.cosmos.pipeline_cosmos3_omni import _EMBODIMENT_TO_DOMAIN_ID, CosmosActionCondition
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
            InputParam(name="cond_input_ids", required=True, description="Token IDs for the conditional prompt."),
            InputParam(name="uncond_input_ids", required=True, description="Token IDs for the unconditional prompt."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "cond_text_segment",
                type_hint=dict,
                kwargs_type="denoiser_input_fields",
                description="Conditional text segment for the denoiser.",
            ),
            OutputParam(
                "uncond_text_segment",
                type_hint=dict,
                kwargs_type="denoiser_input_fields",
                description="Unconditional text segment for the denoiser.",
            ),
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
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="x0_tokens_vision",
                type_hint=torch.Tensor,
                default=None,
                description="Vision latents encoded from the conditioning image or video.",
            ),
            InputParam(
                name="vision_condition_frames",
                type_hint=list[int],
                default=None,
                description="Latent-frame indexes fixed by visual conditioning.",
            ),
            InputParam(name="num_frames", type_hint=int, required=True, description="Number of frames to generate."),
            InputParam(
                name="height", type_hint=int, required=True, description="Height of the generated video in pixels."
            ),
            InputParam(
                name="width", type_hint=int, required=True, description="Width of the generated video in pixels."
            ),
            InputParam(name="fps", type_hint=float, default=24.0, description="Frame rate of the generated video."),
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                default=None,
                description="Pre-generated noisy vision latents.",
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="Noisy vision latents for denoising."),
            OutputParam("fps_vision", type_hint=float, description="Frame rate used to pack vision latents."),
            OutputParam(
                "vision_condition_mask",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Mask marking conditioned vision latent frames.",
            ),
            OutputParam(
                "vision_condition_indexes_for_pack",
                type_hint=list[int],
                description="Indexes of conditioned vision latent frames.",
            ),
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
            sf_spatial = components.vae_scale_factor_spatial
            if block_state.height % sf_spatial != 0 or block_state.width % sf_spatial != 0:
                raise ValueError(
                    f"height and width must be multiples of {sf_spatial}, got ({block_state.height}, {block_state.width})."
                )
            latent_shape = (
                1,
                components.num_channels_latents,
                (block_state.num_frames - 1) // components.vae_scale_factor_temporal + 1,
                block_state.height // sf_spatial,
                block_state.width // sf_spatial,
            )
            x0_tokens_vision = torch.zeros(latent_shape, device=device, dtype=torch.float32)
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
            ComponentSpec("scheduler", UniPCMultistepScheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="num_frames", type_hint=int, required=True, description="Number of frames to generate."),
            InputParam(name="fps", type_hint=float, default=24.0, description="Frame rate of the generated video."),
            InputParam(
                name="sound_latents",
                type_hint=torch.Tensor,
                default=None,
                description="Pre-generated noisy sound latents.",
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("sound_latents", type_hint=torch.Tensor, description="Noisy sound latents for denoising."),
            OutputParam("fps_sound", type_hint=float, description="Frame rate of the sound latent sequence."),
            OutputParam(
                "sound_condition_mask",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Mask marking conditioned sound latent frames.",
            ),
            OutputParam("sound_scheduler", description="Scheduler used to update sound latents."),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        dtype = components.transformer.dtype

        if not components.transformer.config.sound_gen:
            raise ValueError("Sound generation requires a transformer trained with sound_gen=True.")

        sound_dim = components.transformer.config.sound_dim
        block_state.fps_sound = float(components.transformer.config.sound_latent_fps)
        n_audio_samples = int(block_state.num_frames / block_state.fps * components.sound_sampling_rate)
        hop_size = components.sound_hop_size
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

        block_state.sound_scheduler = copy.deepcopy(components.scheduler)

        self.set_block_state(state, block_state)
        return components, state


class Cosmos3ActionPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Prepares noisy action latents and the action conditioning mask."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", Cosmos3OmniTransformer),
            ComponentSpec("scheduler", UniPCMultistepScheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="action",
                type_hint=CosmosActionCondition,
                required=True,
                description="Action-conditioning metadata.",
            ),
            InputParam(
                name="action_condition_frame_indexes",
                type_hint=list[int],
                default=None,
                description="Action-frame indexes fixed by action conditioning.",
            ),
            InputParam(
                name="action_latents",
                type_hint=torch.Tensor,
                default=None,
                description="Pre-generated noisy action latents.",
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("action_latents", type_hint=torch.Tensor, description="Noisy action latents for denoising."),
            OutputParam(
                "action_condition_mask",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Mask marking conditioned action latent frames.",
            ),
            OutputParam(
                "action_domain_ids",
                type_hint=list[torch.Tensor],
                kwargs_type="denoiser_input_fields",
                description="Embodiment domain IDs for action conditioning.",
            ),
            OutputParam(
                "raw_action_dim_resolved",
                type_hint=int,
                kwargs_type="denoiser_input_fields",
                description="Unpadded action-vector dimension.",
            ),
            OutputParam("action_scheduler", description="Scheduler used to update action latents."),
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
        block_state.action_domain_ids = [
            torch.tensor([_EMBODIMENT_TO_DOMAIN_ID[action.domain_name]], dtype=torch.long, device=device)
        ]
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

        block_state.action_scheduler = copy.deepcopy(components.scheduler)

        self.set_block_state(state, block_state)
        return components, state


class Cosmos3VisionPackSequenceStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Builds separate cond/uncond vision sequence segments."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="cond_text_segment", type_hint=dict, required=True, description="Conditional text segment."
            ),
            InputParam(
                name="uncond_text_segment",
                type_hint=dict,
                required=True,
                description="Unconditional text segment.",
            ),
            InputParam(
                name="latents", type_hint=torch.Tensor, required=True, description="Noisy vision latents to pack."
            ),
            InputParam(
                name="fps_vision",
                type_hint=float,
                required=True,
                description="Frame rate used to pack vision latents.",
            ),
            InputParam(
                name="vision_condition_indexes_for_pack",
                type_hint=list[int],
                required=True,
                description="Indexes of conditioned vision latent frames.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "cond_vision_segment",
                type_hint=dict,
                kwargs_type="denoiser_input_fields",
                description="Conditional vision segment for the denoiser.",
            ),
            OutputParam(
                "uncond_vision_segment",
                type_hint=dict,
                kwargs_type="denoiser_input_fields",
                description="Unconditional vision segment for the denoiser.",
            ),
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
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="cond_text_segment", type_hint=dict, required=True, description="Conditional text segment."
            ),
            InputParam(
                name="uncond_text_segment",
                type_hint=dict,
                required=True,
                description="Unconditional text segment.",
            ),
            InputParam(
                name="cond_sequence_length",
                type_hint=int,
                required=True,
                description="Conditional multimodal sequence length.",
            ),
            InputParam(
                name="uncond_sequence_length",
                type_hint=int,
                required=True,
                description="Unconditional multimodal sequence length.",
            ),
            InputParam(
                name="sound_latents", type_hint=torch.Tensor, required=True, description="Noisy sound latents to pack."
            ),
            InputParam(
                name="fps_sound",
                type_hint=float,
                required=True,
                description="Frame rate of the sound latent sequence.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "cond_sound_segment",
                type_hint=dict,
                kwargs_type="denoiser_input_fields",
                description="Conditional sound segment for the denoiser.",
            ),
            OutputParam(
                "uncond_sound_segment",
                type_hint=dict,
                kwargs_type="denoiser_input_fields",
                description="Unconditional sound segment for the denoiser.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        block_state.cond_sound_segment = components._prepare_sound_segment(
            input_sound_tokens=block_state.sound_latents,
            mrope_offset=block_state.cond_text_segment["vision_start_temporal_offset"],
            sound_fps=block_state.fps_sound,
            curr=block_state.cond_sequence_length,
            device=device,
        )
        block_state.uncond_sound_segment = components._prepare_sound_segment(
            input_sound_tokens=block_state.sound_latents,
            mrope_offset=block_state.uncond_text_segment["vision_start_temporal_offset"],
            sound_fps=block_state.fps_sound,
            curr=block_state.uncond_sequence_length,
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
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="cond_text_segment", type_hint=dict, required=True, description="Conditional text segment."
            ),
            InputParam(
                name="uncond_text_segment",
                type_hint=dict,
                required=True,
                description="Unconditional text segment.",
            ),
            InputParam(
                name="cond_sequence_length",
                type_hint=int,
                required=True,
                description="Conditional multimodal sequence length.",
            ),
            InputParam(
                name="uncond_sequence_length",
                type_hint=int,
                required=True,
                description="Unconditional multimodal sequence length.",
            ),
            InputParam(
                name="action_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Noisy action latents to pack.",
            ),
            InputParam(
                name="action_condition_frame_indexes",
                type_hint=list[int],
                default=None,
                description="Action-frame indexes fixed by action conditioning.",
            ),
            InputParam(
                name="fps_vision",
                type_hint=float,
                required=True,
                description="Frame rate used to pack vision latents.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "cond_action_segment",
                type_hint=dict,
                kwargs_type="denoiser_input_fields",
                description="Conditional action segment for the denoiser.",
            ),
            OutputParam(
                "uncond_action_segment",
                type_hint=dict,
                kwargs_type="denoiser_input_fields",
                description="Unconditional action segment for the denoiser.",
            ),
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
            curr=block_state.cond_sequence_length,
            device=device,
        )
        block_state.uncond_action_segment = components._prepare_action_segment(
            input_action_tokens=block_state.action_latents,
            condition_frame_indexes=block_state.action_condition_frame_indexes,
            mrope_offset=block_state.uncond_text_segment["vision_start_temporal_offset"],
            action_fps=block_state.fps_vision,
            curr=block_state.uncond_sequence_length,
            device=device,
        )

        self.set_block_state(state, block_state)
        return components, state


class Cosmos3VisionDenoiseInputStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Assembles text and vision sequence metadata for the denoising loop."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="cond_text_segment", type_hint=dict, required=True, description="Conditional text segment."
            ),
            InputParam(
                name="uncond_text_segment",
                type_hint=dict,
                required=True,
                description="Unconditional text segment.",
            ),
            InputParam(
                name="cond_vision_segment", type_hint=dict, required=True, description="Conditional vision segment."
            ),
            InputParam(
                name="uncond_vision_segment",
                type_hint=dict,
                required=True,
                description="Unconditional vision segment.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "cond_position_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Conditional multimodal RoPE position IDs.",
            ),
            OutputParam(
                "uncond_position_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Unconditional multimodal RoPE position IDs.",
            ),
            OutputParam(
                "cond_sequence_length",
                type_hint=int,
                kwargs_type="denoiser_input_fields",
                description="Conditional multimodal sequence length.",
            ),
            OutputParam(
                "uncond_sequence_length",
                type_hint=int,
                kwargs_type="denoiser_input_fields",
                description="Unconditional multimodal sequence length.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.cond_position_ids = torch.cat(
            [
                block_state.cond_text_segment["text_mrope_ids"],
                block_state.cond_vision_segment["vision_mrope_ids"],
            ],
            dim=1,
        )
        block_state.uncond_position_ids = torch.cat(
            [
                block_state.uncond_text_segment["text_mrope_ids"],
                block_state.uncond_vision_segment["vision_mrope_ids"],
            ],
            dim=1,
        )
        block_state.cond_sequence_length = (
            block_state.cond_text_segment["und_len"] + block_state.cond_vision_segment["num_vision_tokens"]
        )
        block_state.uncond_sequence_length = (
            block_state.uncond_text_segment["und_len"] + block_state.uncond_vision_segment["num_vision_tokens"]
        )
        self.set_block_state(state, block_state)
        return components, state


class Cosmos3SoundDenoiseInputStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Appends sound sequence metadata to the denoising-loop inputs."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="cond_position_ids",
                type_hint=torch.Tensor,
                required=True,
                description="Conditional multimodal RoPE position IDs.",
            ),
            InputParam(
                name="uncond_position_ids",
                type_hint=torch.Tensor,
                required=True,
                description="Unconditional multimodal RoPE position IDs.",
            ),
            InputParam(
                name="cond_sequence_length",
                type_hint=int,
                required=True,
                description="Conditional multimodal sequence length.",
            ),
            InputParam(
                name="uncond_sequence_length",
                type_hint=int,
                required=True,
                description="Unconditional multimodal sequence length.",
            ),
            InputParam(
                name="cond_sound_segment", type_hint=dict, required=True, description="Conditional sound segment."
            ),
            InputParam(
                name="uncond_sound_segment",
                type_hint=dict,
                required=True,
                description="Unconditional sound segment.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "cond_position_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Conditional multimodal RoPE position IDs.",
            ),
            OutputParam(
                "uncond_position_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Unconditional multimodal RoPE position IDs.",
            ),
            OutputParam(
                "cond_sequence_length",
                type_hint=int,
                kwargs_type="denoiser_input_fields",
                description="Conditional multimodal sequence length.",
            ),
            OutputParam(
                "uncond_sequence_length",
                type_hint=int,
                kwargs_type="denoiser_input_fields",
                description="Unconditional multimodal sequence length.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.cond_position_ids = torch.cat(
            [block_state.cond_position_ids, block_state.cond_sound_segment["sound_mrope_ids"]], dim=1
        )
        block_state.uncond_position_ids = torch.cat(
            [block_state.uncond_position_ids, block_state.uncond_sound_segment["sound_mrope_ids"]], dim=1
        )
        block_state.cond_sequence_length += block_state.cond_sound_segment["sound_len"]
        block_state.uncond_sequence_length += block_state.uncond_sound_segment["sound_len"]
        self.set_block_state(state, block_state)
        return components, state


class Cosmos3ActionDenoiseInputStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Appends action sequence metadata to the denoising-loop inputs."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="cond_position_ids",
                type_hint=torch.Tensor,
                required=True,
                description="Conditional multimodal RoPE position IDs.",
            ),
            InputParam(
                name="uncond_position_ids",
                type_hint=torch.Tensor,
                required=True,
                description="Unconditional multimodal RoPE position IDs.",
            ),
            InputParam(
                name="cond_sequence_length",
                type_hint=int,
                required=True,
                description="Conditional multimodal sequence length.",
            ),
            InputParam(
                name="uncond_sequence_length",
                type_hint=int,
                required=True,
                description="Unconditional multimodal sequence length.",
            ),
            InputParam(
                name="cond_action_segment", type_hint=dict, required=True, description="Conditional action segment."
            ),
            InputParam(
                name="uncond_action_segment",
                type_hint=dict,
                required=True,
                description="Unconditional action segment.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "cond_position_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Conditional multimodal RoPE position IDs.",
            ),
            OutputParam(
                "uncond_position_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Unconditional multimodal RoPE position IDs.",
            ),
            OutputParam(
                "cond_sequence_length",
                type_hint=int,
                kwargs_type="denoiser_input_fields",
                description="Conditional multimodal sequence length.",
            ),
            OutputParam(
                "uncond_sequence_length",
                type_hint=int,
                kwargs_type="denoiser_input_fields",
                description="Unconditional multimodal sequence length.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.cond_position_ids = torch.cat(
            [block_state.cond_position_ids, block_state.cond_action_segment["action_mrope_ids"]], dim=1
        )
        block_state.uncond_position_ids = torch.cat(
            [block_state.uncond_position_ids, block_state.uncond_action_segment["action_mrope_ids"]], dim=1
        )
        block_state.cond_sequence_length += block_state.cond_action_segment["action_len"]
        block_state.uncond_sequence_length += block_state.uncond_action_segment["action_len"]
        self.set_block_state(state, block_state)
        return components, state


class Cosmos3SetTimestepsStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Initializes scheduler timesteps."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", UniPCMultistepScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="Scheduler timesteps for denoising."),
            OutputParam("num_warmup_steps", type_hint=int, description="Number of scheduler warmup steps."),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        components.scheduler.set_timesteps(block_state.num_inference_steps, device=device)
        block_state.timesteps = components.scheduler.timesteps
        block_state.num_warmup_steps = (
            len(block_state.timesteps) - block_state.num_inference_steps * components.scheduler.order
        )
        self.set_block_state(state, block_state)
        return components, state
