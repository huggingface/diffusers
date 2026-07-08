import torch
from transformers import AutoTokenizer

from ...configuration_utils import FrozenDict
from ...models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from ...pipelines.cosmos.pipeline_cosmos3_omni import (
    _ACTION_RESOLUTION_BINS,
    CosmosActionCondition,
)
from ...utils import logging
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Cosmos3OmniModularPipeline


logger = logging.get_logger(__name__)


class Cosmos3TextEncoderStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Prepares non-action prompt token IDs for downstream text-segment packing."

    @staticmethod
    def _check_inputs(block_state) -> None:
        prompt = block_state.prompt
        negative_prompt = block_state.negative_prompt

        if not isinstance(prompt, (str, list)) or (
            isinstance(prompt, list) and not all(isinstance(p, str) for p in prompt)
        ):
            raise ValueError(f"`prompt` must be a str or list of str, got {type(prompt).__name__}.")
        if negative_prompt is not None and not isinstance(negative_prompt, (str, list)):
            raise ValueError(
                f"`negative_prompt` must be a str, list of str, or None, got {type(negative_prompt).__name__}."
            )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_tokenizer", AutoTokenizer),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="prompt", type_hint=str, required=True),
            InputParam(name="negative_prompt", default=None),
            InputParam(name="num_frames", default=None),
            InputParam(name="height", default=None),
            InputParam(name="width", default=None),
            InputParam(name="fps", type_hint=float, default=24.0),
            InputParam(name="use_system_prompt", type_hint=bool, default=True),
            InputParam(name="add_resolution_template", type_hint=bool, default=True),
            InputParam(name="add_duration_template", type_hint=bool, default=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("num_frames"),
            OutputParam("height"),
            OutputParam("width"),
            OutputParam("cond_input_ids"),
            OutputParam("uncond_input_ids"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        if block_state.num_frames is None:
            block_state.num_frames = 189
        if block_state.height is None:
            block_state.height = 720
        if block_state.width is None:
            block_state.width = 1280

        self._check_inputs(block_state)
        if isinstance(block_state.prompt, list):
            block_state.prompt = block_state.prompt[0]
        if isinstance(block_state.negative_prompt, list):
            block_state.negative_prompt = block_state.negative_prompt[0]

        if components.requires_safety_checker:
            if getattr(components, "safety_checker", None) is None:
                raise ValueError(
                    "Cosmos3 requires a safety checker by default. Call `pipe.enable_safety_checker()` to load it "
                    "(or pass your own), or opt out explicitly with `pipe.disable_safety_checker()`."
                )
            device = components._execution_device
            components.safety_checker.to(device)
            try:
                if not components.safety_checker.check_text_safety(block_state.prompt):
                    raise ValueError(
                        f"Cosmos Guardrail detected unsafe text in the prompt: {block_state.prompt}. "
                        "Please ensure that the prompt abides by the NVIDIA Open Model License Agreement."
                    )
            finally:
                components.safety_checker.to("cpu")

        block_state.cond_input_ids, block_state.uncond_input_ids = components.tokenize_prompt(
            block_state.prompt,
            block_state.negative_prompt,
            num_frames=block_state.num_frames,
            height=block_state.height,
            width=block_state.width,
            fps=block_state.fps,
            use_system_prompt=block_state.use_system_prompt,
            add_resolution_template=block_state.add_resolution_template,
            add_duration_template=block_state.add_duration_template,
            action_mode=None,
            action_view_point=None,
        )

        self.set_block_state(state, block_state)
        return components, state


class Cosmos3ActionTextStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Prepares action prompt token IDs from prompt + action metadata."

    @staticmethod
    def _check_inputs(block_state) -> None:
        prompt = block_state.prompt
        negative_prompt = block_state.negative_prompt
        action = block_state.action
        num_frames = block_state.num_frames
        height = block_state.height
        width = block_state.width
        if not isinstance(prompt, (str, list)) or (
            isinstance(prompt, list) and not all(isinstance(p, str) for p in prompt)
        ):
            raise ValueError(f"`prompt` must be a str or list of str, got {type(prompt).__name__}.")
        if negative_prompt is not None and not isinstance(negative_prompt, (str, list)):
            raise ValueError(
                f"`negative_prompt` must be a str, list of str, or None, got {type(negative_prompt).__name__}."
            )
        if action is None:
            raise ValueError("`action` is required for Cosmos3ActionTextStep.")
        if action.image is None and action.video is None:
            raise ValueError("`action.image` or `action.video` must be provided for action-conditioned generation.")
        if num_frames is not None:
            raise ValueError("`num_frames` has to be None if action is not None.")
        if height is not None or width is not None:
            raise ValueError("`height` and `width` have to be None if action is not None.")

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_tokenizer", AutoTokenizer),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 16, "resample": "bilinear"}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="prompt", type_hint=str, required=True),
            InputParam(name="negative_prompt", default=None),
            InputParam(name="action", type_hint=CosmosActionCondition, required=True),
            InputParam(name="num_frames", default=None),
            InputParam(name="height", default=None),
            InputParam(name="width", default=None),
            InputParam(name="fps", type_hint=float, default=24.0),
            InputParam(name="use_system_prompt", type_hint=bool, default=True),
            InputParam(name="add_resolution_template", type_hint=bool, default=True),
            InputParam(name="add_duration_template", type_hint=bool, default=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("action_mode"),
            OutputParam("num_frames"),
            OutputParam("height"),
            OutputParam("width"),
            OutputParam("cond_input_ids"),
            OutputParam("uncond_input_ids"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self._check_inputs(block_state)

        action = block_state.action
        block_state.action_mode = action.mode
        block_state.num_frames = action.chunk_size + 1
        conditioning_clip = [action.image] if action.image is not None else action.video
        probe = components.video_processor.preprocess_video(conditioning_clip)
        source_h, source_w = int(probe.shape[-2]), int(probe.shape[-1])
        resolution_key = str(action.resolution_tier)
        block_state.height, block_state.width = VideoProcessor.classify_height_width_bin(
            source_h, source_w, ratios=_ACTION_RESOLUTION_BINS[resolution_key]
        )

        if isinstance(block_state.prompt, list):
            block_state.prompt = block_state.prompt[0]
        if isinstance(block_state.negative_prompt, list):
            block_state.negative_prompt = block_state.negative_prompt[0]

        if components.requires_safety_checker:
            if getattr(components, "safety_checker", None) is None:
                raise ValueError(
                    "Cosmos3 requires a safety checker by default. Call `pipe.enable_safety_checker()` to load it "
                    "(or pass your own), or opt out explicitly with `pipe.disable_safety_checker()`."
                )
            device = components._execution_device
            components.safety_checker.to(device)
            try:
                if not components.safety_checker.check_text_safety(block_state.prompt):
                    raise ValueError(
                        f"Cosmos Guardrail detected unsafe text in the prompt: {block_state.prompt}. "
                        "Please ensure that the prompt abides by the NVIDIA Open Model License Agreement."
                    )
            finally:
                components.safety_checker.to("cpu")

        block_state.cond_input_ids, block_state.uncond_input_ids = components.tokenize_prompt(
            block_state.prompt,
            block_state.negative_prompt,
            num_frames=block_state.num_frames,
            height=block_state.height,
            width=block_state.width,
            fps=block_state.fps,
            use_system_prompt=block_state.use_system_prompt,
            add_resolution_template=block_state.add_resolution_template,
            add_duration_template=block_state.add_duration_template,
            action_mode=block_state.action_mode,
            action_view_point=action.view_point,
        )

        self.set_block_state(state, block_state)
        return components, state


class Cosmos3ImageVaeEncoderStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Encodes non-action image-to-video conditioning into Cosmos3 vision latents."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 16, "resample": "bilinear"}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="image", default=None),
            InputParam(name="num_frames", required=True),
            InputParam(name="height", required=True),
            InputParam(name="width", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("x0_tokens_vision"),
            OutputParam("vision_condition_frames"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.device = components._execution_device
        block_state.dtype = components.transformer.dtype

        if block_state.image is None:
            raise ValueError("`Cosmos3ImageVaeEncoderStep` requires an `image` input.")
        if block_state.num_frames == 1:
            raise ValueError(
                "`image` conditioning requires `num_frames` > 1; image-to-image generation is not supported."
            )
        if block_state.num_frames < 1:
            raise ValueError(f"`num_frames` must be >= 1, got {block_state.num_frames}.")

        sf = int(components.vae.config.scale_factor_spatial)
        if block_state.height % sf != 0 or block_state.width % sf != 0:
            raise ValueError(
                f"`height` and `width` must be multiples of {sf}, got ({block_state.height}, {block_state.width})."
            )

        conditioning_frame_2d = components.video_processor.preprocess(
            block_state.image, height=block_state.height, width=block_state.width
        ).to(device=block_state.device, dtype=block_state.dtype)

        vision_tensor = torch.zeros(
            1,
            3,
            block_state.num_frames,
            block_state.height,
            block_state.width,
            dtype=block_state.dtype,
            device=block_state.device,
        )
        vision_tensor[:, :, 0] = conditioning_frame_2d
        vision_tensor[:, :, 1:] = conditioning_frame_2d.unsqueeze(2).expand(-1, -1, block_state.num_frames - 1, -1, -1)

        block_state.x0_tokens_vision = components._encode_video(vision_tensor).contiguous().float()
        block_state.vision_condition_frames = [0]

        self.set_block_state(state, block_state)
        return components, state


class Cosmos3VideoVaeEncoderStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Encodes non-action video conditioning into Cosmos3 vision latents."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 16, "resample": "bilinear"}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="video", default=None),
            InputParam(name="condition_frame_indexes_vision", default=(0, 1)),
            InputParam(name="condition_video_keep", default="first"),
            InputParam(name="num_frames", required=True),
            InputParam(name="height", required=True),
            InputParam(name="width", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("x0_tokens_vision"),
            OutputParam("vision_condition_frames"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.device = components._execution_device
        block_state.dtype = components.transformer.dtype

        if block_state.video is None:
            raise ValueError("`Cosmos3VideoVaeEncoderStep` requires a `video` input.")
        if block_state.num_frames == 1:
            raise ValueError("`video` conditioning requires `num_frames` > 1.")
        if block_state.num_frames < 1:
            raise ValueError(f"`num_frames` must be >= 1, got {block_state.num_frames}.")

        sf = int(components.vae.config.scale_factor_spatial)
        if block_state.height % sf != 0 or block_state.width % sf != 0:
            raise ValueError(
                f"`height` and `width` must be multiples of {sf}, got ({block_state.height}, {block_state.width})."
            )

        if not isinstance(block_state.condition_frame_indexes_vision, (list, tuple)) or isinstance(
            block_state.condition_frame_indexes_vision, (str, bytes)
        ):
            raise ValueError(
                "`condition_frame_indexes_vision` must be a list/tuple of non-negative ints, e.g. [0, 1]; got "
                f"{block_state.condition_frame_indexes_vision!r}."
            )
        if not all(isinstance(index, int) and index >= 0 for index in block_state.condition_frame_indexes_vision):
            raise ValueError(
                "`condition_frame_indexes_vision` must be a list/tuple of non-negative ints, e.g. [0, 1]; got "
                f"{block_state.condition_frame_indexes_vision!r}."
            )
        if block_state.condition_video_keep not in {"first", "last"}:
            raise ValueError("`condition_video_keep` must be either 'first' or 'last'.")

        indexes = tuple(block_state.condition_frame_indexes_vision)
        if not indexes:
            raise ValueError("`condition_frame_indexes_vision` must contain at least one index.")
        latent_t = (block_state.num_frames - 1) // int(components.vae.config.scale_factor_temporal) + 1
        if max(indexes) >= latent_t:
            raise ValueError(
                f"`condition_frame_indexes_vision` {indexes} contains an index outside the latent timeline "
                f"(latent_frames={latent_t} for num_frames={block_state.num_frames})."
            )

        condition_indexes_vision = indexes
        conditioning_frames_3d = components.video_processor.preprocess_video(
            block_state.video, height=block_state.height, width=block_state.width
        ).to(device=block_state.device, dtype=block_state.dtype)
        temporal_compression = int(components.vae.config.scale_factor_temporal)
        max_cond_frames = max(condition_indexes_vision) * temporal_compression + 1
        if block_state.condition_video_keep == "first":
            conditioning_frames_3d = conditioning_frames_3d[:, :, :max_cond_frames]
        else:
            conditioning_frames_3d = conditioning_frames_3d[:, :, -max_cond_frames:]

        vision_tensor = torch.zeros(
            1,
            3,
            block_state.num_frames,
            block_state.height,
            block_state.width,
            dtype=block_state.dtype,
            device=block_state.device,
        )
        t_fill = min(conditioning_frames_3d.shape[2], block_state.num_frames)
        vision_tensor[:, :, :t_fill] = conditioning_frames_3d[:, :, :t_fill]
        if t_fill < block_state.num_frames:
            vision_tensor[:, :, t_fill:] = vision_tensor[:, :, t_fill - 1 : t_fill].expand(
                -1, -1, block_state.num_frames - t_fill, -1, -1
            )
        vision_condition_frames = list(condition_indexes_vision)

        block_state.x0_tokens_vision = components._encode_video(vision_tensor).contiguous().float()
        block_state.vision_condition_frames = vision_condition_frames

        self.set_block_state(state, block_state)
        return components, state


class Cosmos3ActionVisionVaeEncoderStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return (
            "Prepares action-conditioned vision latents and action frame metadata. "
            "Only the action visual reference (image/video) is VAE-encoded; action vectors are handled separately."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 16, "resample": "bilinear"}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="action", type_hint=CosmosActionCondition, required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("x0_tokens_vision"),
            OutputParam("vision_condition_frames"),
            OutputParam("action_condition_frame_indexes"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.device = components._execution_device
        block_state.dtype = components.vae.dtype

        action = block_state.action
        target_frames = action.chunk_size + 1
        conditioning_clip = [action.image] if action.image is not None else action.video
        vision_tensor, action_image_size, _, _ = components._prepare_action_video_conditioning(
            conditioning_clip,
            action.resolution_tier,
            target_frames,
            device=block_state.device,
            dtype=block_state.dtype,
        )

        if action.mode == "forward_dynamics":
            vision_condition_frames = [0]
            action_condition_frame_indexes = list(range(action.chunk_size))
        elif action.mode == "policy":
            vision_condition_frames = [0]
            action_condition_frame_indexes = []
        elif action.mode == "inverse_dynamics":
            latent_frames = (target_frames - 1) // int(components.vae.config.scale_factor_temporal) + 1
            vision_condition_frames = list(range(latent_frames))
            action_condition_frame_indexes = []
        else:
            raise ValueError(
                f"Unsupported action_mode={action.mode!r}; expected one of ['forward_dynamics', 'inverse_dynamics', 'policy']."
            )

        x0_tokens_vision = components._encode_video(vision_tensor).contiguous().float()
        if action_image_size is not None:
            x0_tokens_vision = components._remove_action_video_padding_from_latent(x0_tokens_vision, action_image_size)

        block_state.x0_tokens_vision = x0_tokens_vision
        block_state.vision_condition_frames = vision_condition_frames
        block_state.action_condition_frame_indexes = action_condition_frame_indexes

        self.set_block_state(state, block_state)
        return components, state
