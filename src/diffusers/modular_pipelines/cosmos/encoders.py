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

        if not isinstance(prompt, str):
            raise ValueError(
                f"`prompt` must be a str; batched prompts are not supported, got {type(prompt).__name__}."
            )
        if negative_prompt is not None and not isinstance(negative_prompt, str):
            raise ValueError(
                "`negative_prompt` must be a str or None; batched prompts are not supported, "
                f"got {type(negative_prompt).__name__}."
            )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_tokenizer", AutoTokenizer),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", description="The text prompt that guides Cosmos3 generation."),
            InputParam.template(
                "negative_prompt", description="The negative text prompt used for classifier-free guidance."
            ),
            InputParam(name="num_frames", type_hint=int, default=None, description="Number of frames to generate."),
            InputParam(
                name="height",
                type_hint=int,
                default=None,
                description="Height of the generated video or image in pixels.",
            ),
            InputParam(
                name="width",
                type_hint=int,
                default=None,
                description="Width of the generated video or image in pixels.",
            ),
            InputParam(name="fps", type_hint=float, default=24.0, description="Frame rate of the generated video."),
            InputParam(
                name="use_system_prompt",
                type_hint=bool,
                default=True,
                description="Whether to prepend the Cosmos3 system prompt.",
            ),
            InputParam(
                name="add_resolution_template",
                type_hint=bool,
                default=True,
                description="Whether to add resolution metadata to the prompt.",
            ),
            InputParam(
                name="add_duration_template",
                type_hint=bool,
                default=True,
                description="Whether to add duration metadata to the prompt.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("num_frames", type_hint=int, description="Number of frames to generate."),
            OutputParam("height", type_hint=int, description="Height of the generated video or image in pixels."),
            OutputParam("width", type_hint=int, description="Width of the generated video or image in pixels."),
            OutputParam("cond_input_ids", type_hint=torch.Tensor, description="Token IDs for the conditional prompt."),
            OutputParam(
                "uncond_input_ids", type_hint=torch.Tensor, description="Token IDs for the unconditional prompt."
            ),
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
        if not isinstance(prompt, str):
            raise ValueError(
                f"`prompt` must be a str; batched prompts are not supported, got {type(prompt).__name__}."
            )
        if negative_prompt is not None and not isinstance(negative_prompt, str):
            raise ValueError(
                "`negative_prompt` must be a str or None; batched prompts are not supported, "
                f"got {type(negative_prompt).__name__}."
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
            InputParam.template("prompt", description="The text prompt that guides Cosmos3 generation."),
            InputParam.template(
                "negative_prompt", description="The negative text prompt used for classifier-free guidance."
            ),
            InputParam(
                name="action",
                type_hint=CosmosActionCondition,
                required=True,
                description="Action-conditioning metadata and its reference visual input.",
            ),
            InputParam(name="num_frames", type_hint=int, default=None, description="Number of frames to generate."),
            InputParam(
                name="height",
                type_hint=int,
                default=None,
                description="Height of the generated video or image in pixels.",
            ),
            InputParam(
                name="width",
                type_hint=int,
                default=None,
                description="Width of the generated video or image in pixels.",
            ),
            InputParam(name="fps", type_hint=float, default=24.0, description="Frame rate of the generated video."),
            InputParam(
                name="use_system_prompt",
                type_hint=bool,
                default=True,
                description="Whether to prepend the Cosmos3 system prompt.",
            ),
            InputParam(
                name="add_resolution_template",
                type_hint=bool,
                default=True,
                description="Whether to add resolution metadata to the prompt.",
            ),
            InputParam(
                name="add_duration_template",
                type_hint=bool,
                default=True,
                description="Whether to add duration metadata to the prompt.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("action_mode", type_hint=str, description="Requested action-generation mode."),
            OutputParam("num_frames", type_hint=int, description="Number of frames to generate."),
            OutputParam("height", type_hint=int, description="Height of the generated video or image in pixels."),
            OutputParam("width", type_hint=int, description="Width of the generated video or image in pixels."),
            OutputParam("cond_input_ids", type_hint=torch.Tensor, description="Token IDs for the conditional prompt."),
            OutputParam(
                "uncond_input_ids", type_hint=torch.Tensor, description="Token IDs for the unconditional prompt."
            ),
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
            InputParam(name="image", default=None, description="Reference image for image-to-video conditioning."),
            InputParam(name="num_frames", type_hint=int, required=True, description="Number of frames to generate."),
            InputParam(
                name="height", type_hint=int, required=True, description="Height of the generated video in pixels."
            ),
            InputParam(
                name="width", type_hint=int, required=True, description="Width of the generated video in pixels."
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "x0_tokens_vision",
                type_hint=torch.Tensor,
                description="Vision latents encoded from the conditioning image or video.",
            ),
            OutputParam(
                "vision_condition_frames",
                type_hint=list[int],
                description="Latent-frame indexes fixed by visual conditioning.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        dtype = components.transformer.dtype

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
        ).to(device=device, dtype=dtype)

        vision_tensor = torch.zeros(
            1,
            3,
            block_state.num_frames,
            block_state.height,
            block_state.width,
            dtype=dtype,
            device=device,
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
            InputParam(name="video", default=None, description="Reference video for video-to-video conditioning."),
            InputParam(
                name="condition_frame_indexes_vision",
                type_hint=tuple[int, ...] | list[int],
                default=(0, 1),
                description="Latent-frame indexes to preserve from the conditioning video.",
            ),
            InputParam(
                name="condition_video_keep",
                type_hint=str,
                default="first",
                description="Which end of a longer conditioning video to use: `first` or `last`.",
            ),
            InputParam(name="num_frames", type_hint=int, required=True, description="Number of frames to generate."),
            InputParam(
                name="height", type_hint=int, required=True, description="Height of the generated video in pixels."
            ),
            InputParam(
                name="width", type_hint=int, required=True, description="Width of the generated video in pixels."
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "x0_tokens_vision",
                type_hint=torch.Tensor,
                description="Vision latents encoded from the conditioning image or video.",
            ),
            OutputParam(
                "vision_condition_frames",
                type_hint=list[int],
                description="Latent-frame indexes fixed by visual conditioning.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        dtype = components.transformer.dtype

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
        ).to(device=device, dtype=dtype)
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
            dtype=dtype,
            device=device,
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
            InputParam(
                name="action",
                type_hint=CosmosActionCondition,
                required=True,
                description="Action-conditioning metadata and its reference visual input.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "x0_tokens_vision",
                type_hint=torch.Tensor,
                description="Vision latents encoded from the conditioning image or video.",
            ),
            OutputParam(
                "vision_condition_frames",
                type_hint=list[int],
                description="Latent-frame indexes fixed by visual conditioning.",
            ),
            OutputParam(
                "action_condition_frame_indexes",
                type_hint=list[int],
                description="Action-frame indexes fixed by action conditioning.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        dtype = components.vae.dtype

        action = block_state.action
        target_frames = action.chunk_size + 1
        conditioning_clip = [action.image] if action.image is not None else action.video
        vision_tensor, action_image_size, _, _ = components._prepare_action_video_conditioning(
            conditioning_clip,
            action.resolution_tier,
            target_frames,
            device=device,
            dtype=dtype,
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
