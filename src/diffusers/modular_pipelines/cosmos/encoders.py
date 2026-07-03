import torch
from transformers import AutoTokenizer

from ...pipelines.cosmos.pipeline_cosmos3_omni import (
    _ACTION_RESOLUTION_BINS,
    CosmosActionCondition,
    CosmosSafetyChecker,
)
from ...utils import logging
from ...video_processor import VideoProcessor
from ..modular_pipeline import AutoPipelineBlocks, ModularPipelineBlocks, PipelineState
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
            InputParam(name="guidance_scale", type_hint=float, default=6.0),
            InputParam(name="use_system_prompt", type_hint=bool, default=True),
            InputParam(name="add_resolution_template", type_hint=bool, default=True),
            InputParam(name="add_duration_template", type_hint=bool, default=True),
            InputParam(name="enable_safety_check", type_hint=bool, default=True),
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

        device = components._get_execution_device()
        if block_state.enable_safety_check and getattr(components, "safety_checker", None) is None:
            try:
                components._ensure_safety_checker()
            except ImportError:
                pass
        if block_state.enable_safety_check and isinstance(components.safety_checker, CosmosSafetyChecker):
            components.safety_checker.to(device)
            try:
                if not components.safety_checker.check_text_safety(block_state.prompt):
                    raise ValueError(
                        f"Cosmos Guardrail detected unsafe text in the prompt: {block_state.prompt}. "
                        "Please ensure that the prompt abides by the NVIDIA Open Model License Agreement."
                    )
            finally:
                components.safety_checker.to("cpu")

        block_state.action_mode = None
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
            InputParam(name="action", type_hint=CosmosActionCondition, required=True),
            InputParam(name="fps", type_hint=float, default=24.0),
            InputParam(name="guidance_scale", type_hint=float, default=6.0),
            InputParam(name="use_system_prompt", type_hint=bool, default=True),
            InputParam(name="add_resolution_template", type_hint=bool, default=True),
            InputParam(name="add_duration_template", type_hint=bool, default=True),
            InputParam(name="enable_safety_check", type_hint=bool, default=True),
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

        device = components._get_execution_device()
        if block_state.enable_safety_check and getattr(components, "safety_checker", None) is None:
            try:
                components._ensure_safety_checker()
            except ImportError:
                pass
        if block_state.enable_safety_check and isinstance(components.safety_checker, CosmosSafetyChecker):
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


class Cosmos3AutoTextEncoderStep(AutoPipelineBlocks):
    model_name = "cosmos3-omni"
    block_classes = [Cosmos3ActionTextStep, Cosmos3TextEncoderStep]
    block_names = ["action_text", "text"]
    block_trigger_inputs = ["action", None]

    @property
    def description(self):
        return (
            "Auto text encoder block for Cosmos3.\n"
            + " - `Cosmos3ActionTextStep` runs when `action` is provided.\n"
            + " - `Cosmos3TextEncoderStep` runs otherwise."
        )
