# Copyright 2026 chinoll and The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor

from ...configuration_utils import FrozenDict
from ...models import HiDreamO1Transformer2DModel
from ...schedulers import UniPCMultistepScheduler
from ...utils import numpy_to_pil
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState, SequentialPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import HiDreamO1ModularPipeline, HiDreamO1Patchifier
from .utils import (
    FULL_NOISE_SCALE,
    PATCH_SIZE,
    TIMESTEP_TOKEN_NUM,
    add_special_tokens,
    find_closest_resolution,
    get_rope_index_fix_point,
    get_tokenizer,
    retrieve_timesteps,
    set_scheduler_shift,
    to_device,
)


def _build_text_to_image_sample(
    components: HiDreamO1ModularPipeline,
    prompt: str,
    height: int,
    width: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tokenizer = get_tokenizer(components.processor)
    model_config = components.transformer.qwen_config
    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)

    messages = [{"role": "user", "content": prompt}]
    template_caption = (
        components.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + tokenizer.boi_token
        + tokenizer.tms_token * TIMESTEP_TOKEN_NUM
    )
    input_ids = tokenizer.encode(template_caption, return_tensors="pt", add_special_tokens=False)

    image_grid_thw = torch.tensor([1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64).unsqueeze(0)
    vision_tokens = torch.full((1, image_len), image_token_id, dtype=input_ids.dtype)
    vision_tokens[0, 0] = vision_start_token_id
    input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

    position_ids, _ = get_rope_index_fix_point(
        1,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        input_ids=input_ids_pad,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=None,
        skip_vision_start_token=[1],
    )

    text_seq_len = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]
    token_types = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
    start = text_seq_len - TIMESTEP_TOKEN_NUM
    token_types[0, start : start + image_len + TIMESTEP_TOKEN_NUM] = 1
    token_types[0, text_seq_len - TIMESTEP_TOKEN_NUM : text_seq_len] = 3

    sample = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "token_types": (token_types > 0).to(token_types.dtype),
        "vinput_mask": token_types == 1,
    }
    return to_device(sample, device)


def _forward_transformer(
    components: HiDreamO1ModularPipeline,
    sample: dict[str, torch.Tensor],
    patches: torch.Tensor,
    timestep: torch.Tensor,
    attention_kwargs: dict[str, Any] | None,
) -> torch.Tensor:
    outputs = components.transformer(
        input_ids=sample["input_ids"],
        position_ids=sample["position_ids"],
        vinputs=patches,
        timestep=timestep.reshape(-1),
        token_types=sample["token_types"],
        attention_kwargs=attention_kwargs,
    )
    return outputs.sample[0, sample["vinput_mask"][0]].unsqueeze(0)


class HiDreamO1PromptSampleStep(ModularPipelineBlocks):
    model_name = "hidream-o1"

    @property
    def description(self) -> str:
        return "Prepare HiDream-O1 text-to-image prompt samples and multimodal RoPE metadata."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("processor", AutoProcessor),
            ComponentSpec("transformer", HiDreamO1Transformer2DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt"),
            InputParam.template("height", default=2048),
            InputParam.template("width", default=2048),
            InputParam("guidance_scale", type_hint=float, default=5.0, description="Classifier-free guidance scale."),
            InputParam(
                "use_resolution_binning",
                type_hint=bool,
                default=True,
                description="Whether to snap height and width to one of the official high-resolution buckets.",
            ),
            InputParam.template("output_type"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("height", type_hint=int, description="Resolved image height."),
            OutputParam("width", type_hint=int, description="Resolved image width."),
            OutputParam("samples", type_hint=list, description="Conditional and optional unconditional O1 samples."),
        ]

    @staticmethod
    def check_inputs(prompt: str, height: int, width: int, output_type: str, use_resolution_binning: bool):
        if not isinstance(prompt, str):
            raise TypeError("`prompt` must be a string. Batched prompts are not implemented for HiDream-O1.")
        if output_type not in {"pil", "np", "pt"}:
            raise ValueError("`output_type` must be one of 'pil', 'np', or 'pt'.")
        if height <= 0 or width <= 0:
            raise ValueError("`height` and `width` must be positive.")
        if not use_resolution_binning and (height % PATCH_SIZE != 0 or width % PATCH_SIZE != 0):
            raise ValueError(f"`height` and `width` must be divisible by {PATCH_SIZE} when resolution binning is off.")

    @torch.no_grad()
    def __call__(self, components: HiDreamO1ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        self.check_inputs(
            block_state.prompt,
            block_state.height,
            block_state.width,
            block_state.output_type,
            block_state.use_resolution_binning,
        )
        if block_state.use_resolution_binning:
            block_state.width, block_state.height = find_closest_resolution(block_state.width, block_state.height)

        add_special_tokens(get_tokenizer(components.processor))

        device = components._execution_device
        cond_sample = _build_text_to_image_sample(
            components, block_state.prompt, block_state.height, block_state.width, device
        )
        block_state.samples = [cond_sample]
        if block_state.guidance_scale > 1.0:
            block_state.samples.append(
                _build_text_to_image_sample(components, " ", block_state.height, block_state.width, device)
            )

        self.set_block_state(state, block_state)
        return components, state


class HiDreamO1PrepareImageNoiseStep(ModularPipelineBlocks):
    model_name = "hidream-o1"

    @property
    def description(self) -> str:
        return "Prepare initial raw RGB image noise and pack it into O1 patch tokens."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "patchifier",
                HiDreamO1Patchifier,
                config=FrozenDict({"patch_size": PATCH_SIZE}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", HiDreamO1Transformer2DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam.template("generator"),
            InputParam.template("latents"),
            InputParam(
                "noise_scale_start",
                type_hint=float,
                default=FULL_NOISE_SCALE,
                description="Scale applied to the initial image noise before patchification.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("patches", type_hint=torch.Tensor, description="Initial raw RGB image patch tokens."),
        ]

    @torch.no_grad()
    def __call__(self, components: HiDreamO1ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        if block_state.latents is not None:
            block_state.patches = block_state.latents
            self.set_block_state(state, block_state)
            return components, state

        device = components._execution_device
        dtype = components.transformer.dtype
        image_noise = randn_tensor(
            (1, 3, block_state.height, block_state.width),
            generator=block_state.generator,
            device=device,
            dtype=torch.float32,
        )
        image_noise = block_state.noise_scale_start * image_noise.to(device=device, dtype=dtype)
        block_state.patches = components.patchifier.pack_image(image_noise)

        self.set_block_state(state, block_state)
        return components, state


class HiDreamO1SetTimestepsStep(ModularPipelineBlocks):
    model_name = "hidream-o1"

    @property
    def description(self) -> str:
        return "Set the scheduler timesteps and O1 noise scale schedule."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "scheduler",
                UniPCMultistepScheduler,
                config=FrozenDict({"prediction_type": "sample", "use_flow_sigmas": True, "flow_shift": 3.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps"),
            InputParam("shift", type_hint=float, default=3.0, description="Flow matching timestep shift."),
            InputParam("timesteps", type_hint=list, description="Optional custom timestep schedule."),
            InputParam.template("sigmas"),
            InputParam(
                "noise_scale_start",
                type_hint=float,
                default=FULL_NOISE_SCALE,
                description="Initial scheduler stochastic noise scale.",
            ),
            InputParam(
                "noise_scale_end",
                type_hint=float,
                description="Final scheduler stochastic noise scale.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="Timesteps used by the scheduler."),
            OutputParam("num_inference_steps", type_hint=int, description="Resolved number of inference steps."),
            OutputParam(
                "noise_scale_schedule", type_hint=list, description="Per-step scheduler stochastic noise scale."
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: HiDreamO1ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        block_state.num_inference_steps = (
            50 if block_state.num_inference_steps is None else block_state.num_inference_steps
        )
        block_state.noise_scale_end = (
            block_state.noise_scale_start if block_state.noise_scale_end is None else block_state.noise_scale_end
        )

        set_scheduler_shift(components.scheduler, block_state.shift)
        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(
            components.scheduler,
            block_state.num_inference_steps,
            device,
            block_state.timesteps,
            block_state.sigmas,
        )

        if len(block_state.timesteps) > 1:
            block_state.noise_scale_schedule = [
                block_state.noise_scale_start
                + (block_state.noise_scale_end - block_state.noise_scale_start)
                * step
                / (len(block_state.timesteps) - 1)
                for step in range(len(block_state.timesteps))
            ]
        else:
            block_state.noise_scale_schedule = [block_state.noise_scale_start]

        self.set_block_state(state, block_state)
        return components, state


class HiDreamO1DenoiseStep(ModularPipelineBlocks):
    model_name = "hidream-o1"

    @torch.compiler.disable
    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        if total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

    @property
    def description(self) -> str:
        return "Iteratively denoise O1 raw RGB patch tokens."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", HiDreamO1Transformer2DModel),
            ComponentSpec(
                "scheduler",
                UniPCMultistepScheduler,
                config=FrozenDict({"prediction_type": "sample", "use_flow_sigmas": True, "flow_shift": 3.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "samples", required=True, type_hint=list, description="Conditional and optional unconditional samples."
            ),
            InputParam("patches", required=True, type_hint=torch.Tensor, description="Raw RGB image patch tokens."),
            InputParam.template("timesteps", required=True),
            InputParam.template("num_inference_steps", required=True),
            InputParam("guidance_scale", type_hint=float, default=5.0, description="Classifier-free guidance scale."),
            InputParam.template("generator"),
            InputParam.template("attention_kwargs"),
            InputParam(
                "noise_scale_schedule", type_hint=list, description="Per-step scheduler stochastic noise scale."
            ),
            InputParam(
                "noise_clip_std", type_hint=float, default=0.0, description="Scheduler stochastic noise clipping std."
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("patches", type_hint=torch.Tensor, description="Denoised raw RGB image patch tokens."),
        ]

    @torch.no_grad()
    def __call__(self, components: HiDreamO1ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        dtype = components.transformer.dtype
        autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        attention_kwargs = {} if block_state.attention_kwargs is None else dict(block_state.attention_kwargs)

        step_kwargs = {}
        step_signature = set(inspect.signature(components.scheduler.step).parameters.keys())
        if "generator" in step_signature:
            step_kwargs["generator"] = block_state.generator

        with self.progress_bar(total=len(block_state.timesteps)) as progress_bar:
            for step_idx, step_t in enumerate(block_state.timesteps):
                step_t = step_t.to(device=device, dtype=torch.float32)
                t_pixeldit = 1.0 - step_t / 1000.0

                with torch.autocast(device.type, dtype=dtype, enabled=autocast_enabled, cache_enabled=False):
                    x_pred_cond = _forward_transformer(
                        components, block_state.samples[0], block_state.patches.clone(), t_pixeldit, attention_kwargs
                    )

                if len(block_state.samples) > 1:
                    with torch.autocast(device.type, dtype=dtype, enabled=autocast_enabled, cache_enabled=False):
                        x_pred_uncond = _forward_transformer(
                            components,
                            block_state.samples[1],
                            block_state.patches.clone(),
                            t_pixeldit,
                            attention_kwargs,
                        )
                    model_output = x_pred_uncond + block_state.guidance_scale * (x_pred_cond - x_pred_uncond)
                else:
                    model_output = x_pred_cond

                current_step_kwargs = dict(step_kwargs)
                if "s_noise" in step_signature:
                    current_step_kwargs["s_noise"] = block_state.noise_scale_schedule[step_idx]
                if "noise_clip_std" in step_signature:
                    current_step_kwargs["noise_clip_std"] = block_state.noise_clip_std

                block_state.patches = components.scheduler.step(
                    model_output.float(),
                    step_t,
                    block_state.patches.float(),
                    return_dict=False,
                    **current_step_kwargs,
                )[0].to(dtype)
                progress_bar.update()

        self.set_block_state(state, block_state)
        return components, state


class HiDreamO1DecodeStep(ModularPipelineBlocks):
    model_name = "hidream-o1"

    @property
    def description(self) -> str:
        return "Unpack denoised RGB patches and postprocess images."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "patchifier",
                HiDreamO1Patchifier,
                config=FrozenDict({"patch_size": PATCH_SIZE}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "patches", required=True, type_hint=torch.Tensor, description="Denoised raw RGB image patch tokens."
            ),
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam.template("output_type"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("images"),
        ]

    @torch.no_grad()
    def __call__(self, components: HiDreamO1ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        image = (block_state.patches + 1) / 2
        image = components.patchifier.unpack_image(image.float(), block_state.height, block_state.width)

        if block_state.output_type == "pt":
            block_state.images = image
        else:
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            image = np.clip(image, 0, 1)
            if block_state.output_type == "pil":
                block_state.images = numpy_to_pil(image)
            else:
                block_state.images = image

        self.set_block_state(state, block_state)
        return components, state


class HiDreamO1AutoBlocks(SequentialPipelineBlocks):
    """
    Modular text-to-image pipeline for HiDream-O1.
    """

    block_classes = [
        HiDreamO1PromptSampleStep(),
        HiDreamO1PrepareImageNoiseStep(),
        HiDreamO1SetTimestepsStep(),
        HiDreamO1DenoiseStep(),
        HiDreamO1DecodeStep(),
    ]
    block_names = [
        "prompt_sample",
        "prepare_image_noise",
        "set_timesteps",
        "denoise",
        "decode",
    ]
    _workflow_map = {"text2image": {"prompt": "prompt"}}

    @property
    def description(self):
        return "Modular text-to-image pipeline for HiDream-O1 raw RGB patch generation."

    @property
    def outputs(self):
        return [
            OutputParam.template("images"),
        ]
