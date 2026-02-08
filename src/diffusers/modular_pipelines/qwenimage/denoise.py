# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, List, Optional, Tuple

import torch

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models import QwenImageControlNetModel, QwenImageTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ..modular_pipeline import BlockState, LoopSequentialPipelineBlocks, ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import QwenImageModularPipeline


logger = logging.get_logger(__name__)


def _validate_area_scalar(
    value: Any,
    field_name: str,
    area_index: int,
) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"`area_composition[{area_index}]['{field_name}']` must be int or float, but got {type(value)}."
        )

    scalar = float(value)
    if scalar < 0:
        raise ValueError(
            f"`area_composition[{area_index}]['{field_name}']` must be non-negative, but got {value}."
        )

    return scalar


def _normalize_area_value_to_latent_grid(
    value: float,
    image_size: int,
    latent_size: int,
    coordinate_space: str,
) -> int:
    if coordinate_space == "percentage":
        return max(int(round(value * latent_size)), 0)

    if coordinate_space == "comfy_latent":
        return max(int(round(value / 2.0)), 0)

    return max(int(round(value * latent_size / image_size)), 0)


def _build_area_masks_from_composition(
    area_composition: List[Dict[str, Any]],
    latent_height: int,
    latent_width: int,
    image_height: int,
    image_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[int], List[torch.Tensor], List[float]]:
    valid_area_indices: List[int] = []
    area_masks: List[torch.Tensor] = []
    area_strengths: List[float] = []

    for area_index, area in enumerate(area_composition):
        if not isinstance(area, dict):
            raise ValueError(
                f"`area_composition[{area_index}]` must be a dictionary, but got {type(area)}."
            )

        required_fields = ["x", "y", "width", "height"]
        for field_name in required_fields:
            if field_name not in area:
                raise ValueError(f"`area_composition[{area_index}]` is missing required field `{field_name}`.")

        x_value = _validate_area_scalar(area["x"], field_name="x", area_index=area_index)
        y_value = _validate_area_scalar(area["y"], field_name="y", area_index=area_index)
        width_value = _validate_area_scalar(area["width"], field_name="width", area_index=area_index)
        height_value = _validate_area_scalar(area["height"], field_name="height", area_index=area_index)

        values = [x_value, y_value, width_value, height_value]
        if all(value <= 1.0 for value in values):
            coordinate_space = "percentage"
        else:
            # ComfyUI internal representation uses latent coordinates on a /8 grid.
            # QwenImage denoiser grid is /16, so divide by 2 when users pass Comfy latent-space values directly.
            max_extent_like_latent = max(x_value + width_value, y_value + height_value)
            max_latent_extent = max(latent_width, latent_height) * 2 + 1e-6
            coordinate_space = "comfy_latent" if max_extent_like_latent <= max_latent_extent else "pixel"

        x = _normalize_area_value_to_latent_grid(
            x_value,
            image_size=image_width,
            latent_size=latent_width,
            coordinate_space=coordinate_space,
        )
        y = _normalize_area_value_to_latent_grid(
            y_value,
            image_size=image_height,
            latent_size=latent_height,
            coordinate_space=coordinate_space,
        )
        w = _normalize_area_value_to_latent_grid(
            width_value,
            image_size=image_width,
            latent_size=latent_width,
            coordinate_space=coordinate_space,
        )
        h = _normalize_area_value_to_latent_grid(
            height_value,
            image_size=image_height,
            latent_size=latent_height,
            coordinate_space=coordinate_space,
        )

        if w <= 0 or h <= 0:
            continue

        x0 = min(max(0, x), latent_width)
        y0 = min(max(0, y), latent_height)
        x1 = min(latent_width, x0 + w)
        y1 = min(latent_height, y0 + h)

        if x1 <= x0 or y1 <= y0:
            continue

        area_mask = torch.zeros((latent_height, latent_width), device=device, dtype=dtype)
        patch = torch.ones((y1 - y0, x1 - x0), device=device, dtype=dtype)

        # Matches ComfyUI get_area_and_mult behavior for area-only conditioning.
        fuzz = 8
        area_height = y1 - y0
        area_width = x1 - x0
        rr_y = min(fuzz, area_height // 4)
        rr_x = min(fuzz, area_width // 4)

        if rr_y > 0:
            if y0 != 0:
                for t_idx in range(rr_y):
                    patch[t_idx, :] *= (t_idx + 1) / rr_y
            if y1 < latent_height:
                for t_idx in range(rr_y):
                    patch[area_height - 1 - t_idx, :] *= (t_idx + 1) / rr_y

        if rr_x > 0:
            if x0 != 0:
                for t_idx in range(rr_x):
                    patch[:, t_idx] *= (t_idx + 1) / rr_x
            if x1 < latent_width:
                for t_idx in range(rr_x):
                    patch[:, area_width - 1 - t_idx] *= (t_idx + 1) / rr_x

        area_mask[y0:y1, x0:x1] = patch

        valid_area_indices.append(area_index)
        area_masks.append(area_mask)
        area_strengths.append(max(float(area.get("strength", 1.0)), 0.0))

    return valid_area_indices, area_masks, area_strengths


def _run_cfg_denoiser(
    components: QwenImageModularPipeline,
    latent_model_input: torch.Tensor,
    timestep: torch.Tensor,
    attention_kwargs: Optional[Dict[str, Any]],
    cond_prompt_embeds: torch.Tensor,
    cond_prompt_embeds_mask: torch.Tensor,
    uncond_prompt_embeds: Optional[torch.Tensor],
    uncond_prompt_embeds_mask: Optional[torch.Tensor],
    num_inference_steps: int,
    step_index: int,
    t: torch.Tensor,
    additional_cond_kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    transformer_dtype = components.transformer.dtype
    latent_model_input = latent_model_input.to(dtype=transformer_dtype)
    timestep = timestep.to(dtype=transformer_dtype)

    if cond_prompt_embeds is not None:
        cond_prompt_embeds = cond_prompt_embeds.to(dtype=transformer_dtype)
    if uncond_prompt_embeds is not None:
        uncond_prompt_embeds = uncond_prompt_embeds.to(dtype=transformer_dtype)

    guider_inputs = {
        "encoder_hidden_states": (cond_prompt_embeds, uncond_prompt_embeds),
        "encoder_hidden_states_mask": (cond_prompt_embeds_mask, uncond_prompt_embeds_mask),
    }

    components.guider.set_state(step=step_index, num_inference_steps=num_inference_steps, timestep=t)
    guider_state = components.guider.prepare_inputs(guider_inputs)

    for guider_state_batch in guider_state:
        components.guider.prepare_models(components.transformer)
        cond_kwargs = {input_name: getattr(guider_state_batch, input_name) for input_name in guider_inputs.keys()}
        guider_state_batch.noise_pred = components.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            attention_kwargs=attention_kwargs,
            return_dict=False,
            **cond_kwargs,
            **additional_cond_kwargs,
        )[0]
        components.guider.cleanup_models(components.transformer)

    guider_output = components.guider(guider_state)
    return guider_output.pred, guider_output.pred_cond


def _rescale_noise_prediction(pred: torch.Tensor, pred_cond: torch.Tensor) -> torch.Tensor:
    pred = torch.nan_to_num(pred)
    pred_cond = torch.nan_to_num(pred_cond)

    pred_cond_norm = torch.norm(pred_cond, dim=-1, keepdim=True)
    pred_norm = torch.norm(pred, dim=-1, keepdim=True)

    noise_pred = pred * (pred_cond_norm / pred_norm.clamp(min=1e-6))
    noise_pred = torch.nan_to_num(noise_pred)
    return noise_pred


# ====================
# 1. LOOP STEPS (run at each denoising step)
# ====================


# loop step:before denoiser
class QwenImageLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that prepares the latent input for the denoiser. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImageDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        # one timestep
        block_state.timestep = t.expand(block_state.latents.shape[0]).to(block_state.latents.dtype)
        block_state.latent_model_input = block_state.latents
        return components, block_state


class QwenImageEditLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "qwenimage-edit"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that prepares the latent input for the denoiser. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImageDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The initial latents to use for the denoising process. Can be generated in prepare_latent step.",
            ),
            InputParam.template("image_latents"),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        # one timestep

        block_state.latent_model_input = torch.cat([block_state.latents, block_state.image_latents], dim=1)
        block_state.timestep = t.expand(block_state.latents.shape[0]).to(block_state.latents.dtype)
        return components, block_state


class QwenImageLoopBeforeDenoiserControlNet(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec("controlnet", QwenImageControlNetModel),
        ]

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that runs the controlnet before the denoiser. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImageDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "control_image_latents",
                required=True,
                type_hint=torch.Tensor,
                description="The control image to use for the denoising process. Can be generated in prepare_controlnet_inputs step.",
            ),
            InputParam.template("controlnet_conditioning_scale", note="updated in prepare_controlnet_inputs step."),
            InputParam(
                name="controlnet_keep",
                required=True,
                type_hint=List[float],
                description="The controlnet keep values. Can be generated in prepare_controlnet_inputs step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: int):
        # cond_scale for the timestep (controlnet input)
        if isinstance(block_state.controlnet_keep[i], list):
            block_state.cond_scale = [
                c * s for c, s in zip(block_state.controlnet_conditioning_scale, block_state.controlnet_keep[i])
            ]
        else:
            controlnet_cond_scale = block_state.controlnet_conditioning_scale
            if isinstance(controlnet_cond_scale, list):
                controlnet_cond_scale = controlnet_cond_scale[0]
            block_state.cond_scale = controlnet_cond_scale * block_state.controlnet_keep[i]

        # run controlnet for the guidance batch
        controlnet_block_samples = components.controlnet(
            hidden_states=block_state.latent_model_input,
            controlnet_cond=block_state.control_image_latents,
            conditioning_scale=block_state.cond_scale,
            timestep=block_state.timestep / 1000,
            img_shapes=block_state.img_shapes,
            encoder_hidden_states=block_state.prompt_embeds,
            encoder_hidden_states_mask=block_state.prompt_embeds_mask,
            return_dict=False,
        )

        block_state.additional_cond_kwargs["controlnet_block_samples"] = controlnet_block_samples

        return components, block_state


# loop step:denoiser
class QwenImageLoopDenoiser(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that denoise the latent input for the denoiser. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImageDenoiseLoopWrapper`)"
        )

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", QwenImageTransformer2DModel),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam.template("attention_kwargs"),
            InputParam.template("denoiser_input_fields"),
            InputParam(
                "img_shapes",
                required=True,
                type_hint=List[Tuple[int, int]],
                description="The shape of the image latents for RoPE calculation. can be generated in prepare_additional_inputs step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        guider_inputs = {
            "encoder_hidden_states": (
                getattr(block_state, "prompt_embeds", None),
                getattr(block_state, "negative_prompt_embeds", None),
            ),
            "encoder_hidden_states_mask": (
                getattr(block_state, "prompt_embeds_mask", None),
                getattr(block_state, "negative_prompt_embeds_mask", None),
            ),
        }

        transformer_args = set(inspect.signature(components.transformer.forward).parameters.keys())
        additional_cond_kwargs = {}
        for field_name, field_value in block_state.denoiser_input_fields.items():
            if field_name in transformer_args and field_name not in guider_inputs:
                additional_cond_kwargs[field_name] = field_value
        block_state.additional_cond_kwargs.update(additional_cond_kwargs)

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs(guider_inputs)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = {input_name: getattr(guider_state_batch, input_name) for input_name in guider_inputs.keys()}

            # YiYi TODO: add cache context
            guider_state_batch.noise_pred = components.transformer(
                hidden_states=block_state.latent_model_input,
                timestep=block_state.timestep / 1000,
                attention_kwargs=block_state.attention_kwargs,
                return_dict=False,
                **cond_kwargs,
                **block_state.additional_cond_kwargs,
            )[0]

            components.guider.cleanup_models(components.transformer)

        guider_output = components.guider(guider_state)

        # apply guidance rescale
        pred_cond_norm = torch.norm(guider_output.pred_cond, dim=-1, keepdim=True)
        pred_norm = torch.norm(guider_output.pred, dim=-1, keepdim=True)
        block_state.noise_pred = guider_output.pred * (pred_cond_norm / pred_norm)

        return components, block_state


class QwenImageAreaCompositionLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that prepares area-composition data before denoiser prediction. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImageDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                name="latents",
                required=True,
                type_hint=torch.Tensor,
                description="The current latent tensor in the denoising loop.",
            ),
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam(
                name="area_composition",
                type_hint=List[Dict[str, Any]],
                description=(
                    "Optional regional prompt configuration with entries containing area coordinates and prompts."
                ),
            ),
            InputParam(
                name="area_prompt_embeds",
                type_hint=torch.Tensor,
                description="Regional prompt embeddings with shape [num_areas, batch, seq_len, hidden_dim].",
            ),
            InputParam(
                name="area_prompt_embeds_mask",
                type_hint=torch.Tensor,
                description="Regional prompt masks with shape [num_areas, batch, seq_len].",
            ),
            InputParam(
                name="area_negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="Regional negative prompt embeddings with shape [num_areas, batch, seq_len, hidden_dim].",
            ),
            InputParam(
                name="area_negative_prompt_embeds_mask",
                type_hint=torch.Tensor,
                description="Regional negative prompt masks with shape [num_areas, batch, seq_len].",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                name="area_composition_enabled",
                type_hint=bool,
                description="Whether area composition should be applied for this loop step.",
            ),
            OutputParam(
                name="area_valid_indices",
                type_hint=List[int],
                description="Indices of valid area definitions after shape conversion.",
            ),
            OutputParam(
                name="area_token_weights",
                type_hint=torch.Tensor,
                description="Flattened per-area token weights used to merge regional predictions.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.area_composition_enabled = False
        block_state.area_valid_indices = []
        block_state.area_token_weights = None

        if not (
            getattr(block_state, "area_composition", None)
            and getattr(block_state, "area_prompt_embeds", None) is not None
        ):
            return components, block_state

        latent_sequence_length = block_state.latents.size(1)
        requested_image_height = int(getattr(block_state, "height", 0) or 0)
        requested_image_width = int(getattr(block_state, "width", 0) or 0)

        img_shapes = getattr(block_state, "img_shapes", None)
        if not (isinstance(img_shapes, list) and len(img_shapes) > 0):
            return components, block_state

        first_batch_shapes = img_shapes[0]
        if not (isinstance(first_batch_shapes, list) and len(first_batch_shapes) > 0):
            return components, block_state

        first_shape = first_batch_shapes[0]
        if not (isinstance(first_shape, (list, tuple)) and len(first_shape) >= 2):
            return components, block_state

        latent_height = int(first_shape[-2])
        latent_width = int(first_shape[-1])

        if latent_height <= 0 or latent_width <= 0:
            return components, block_state

        if latent_height * latent_width != latent_sequence_length:
            return components, block_state

        fallback_image_height = int(getattr(block_state, "image_height", 0) or 0)
        fallback_image_width = int(getattr(block_state, "image_width", 0) or 0)
        image_height = requested_image_height or fallback_image_height or latent_height * 16
        image_width = requested_image_width or fallback_image_width or latent_width * 16

        valid_area_indices, area_masks, area_strengths = _build_area_masks_from_composition(
            area_composition=block_state.area_composition,
            latent_height=latent_height,
            latent_width=latent_width,
            image_height=image_height,
            image_width=image_width,
            device=block_state.latents.device,
            dtype=block_state.latents.dtype,
        )
        if len(area_masks) == 0 or len(valid_area_indices) == 0:
            return components, block_state

        area_token_weights = []
        for area_mask, strength in zip(area_masks, area_strengths):
            token_weight = area_mask.view(-1) * max(strength, 0.0)
            area_token_weights.append(token_weight)
        area_token_weights = torch.stack(area_token_weights, dim=0)

        block_state.area_composition_enabled = True
        block_state.area_valid_indices = valid_area_indices
        block_state.area_token_weights = area_token_weights

        return components, block_state


class QwenImageAreaCompositionLoopDenoiser(QwenImageLoopDenoiser):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that applies global + regional denoiser predictions and merges them. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImageDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return super().inputs + [
            InputParam(
                name="area_composition_enabled",
                type_hint=bool,
                description="Whether area composition should run for this step.",
            ),
            InputParam(
                name="area_valid_indices",
                type_hint=List[int],
                description="Indices of valid area entries.",
            ),
            InputParam(
                name="area_token_weights",
                type_hint=torch.Tensor,
                description="Flattened per-area token weights used to merge regional predictions.",
            ),
            InputParam(
                name="area_prompt_embeds",
                type_hint=torch.Tensor,
                description="Regional prompt embeddings with shape [num_areas, batch, seq_len, hidden_dim].",
            ),
            InputParam(
                name="area_prompt_embeds_mask",
                type_hint=torch.Tensor,
                description="Regional prompt masks with shape [num_areas, batch, seq_len].",
            ),
            InputParam(
                name="area_negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="Regional negative prompt embeddings with shape [num_areas, batch, seq_len, hidden_dim].",
            ),
            InputParam(
                name="area_negative_prompt_embeds_mask",
                type_hint=torch.Tensor,
                description="Regional negative prompt masks with shape [num_areas, batch, seq_len].",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        guider_inputs = {
            "encoder_hidden_states": (
                getattr(block_state, "prompt_embeds", None),
                getattr(block_state, "negative_prompt_embeds", None),
            ),
            "encoder_hidden_states_mask": (
                getattr(block_state, "prompt_embeds_mask", None),
                getattr(block_state, "negative_prompt_embeds_mask", None),
            ),
        }

        transformer_args = set(inspect.signature(components.transformer.forward).parameters.keys())
        additional_cond_kwargs = {}
        for field_name, field_value in block_state.denoiser_input_fields.items():
            if field_name in transformer_args and field_name not in guider_inputs:
                additional_cond_kwargs[field_name] = field_value
        block_state.additional_cond_kwargs.update(additional_cond_kwargs)

        pred, pred_cond = _run_cfg_denoiser(
            components=components,
            latent_model_input=block_state.latent_model_input,
            timestep=block_state.timestep / 1000,
            attention_kwargs=block_state.attention_kwargs,
            cond_prompt_embeds=getattr(block_state, "prompt_embeds", None),
            cond_prompt_embeds_mask=getattr(block_state, "prompt_embeds_mask", None),
            uncond_prompt_embeds=getattr(block_state, "negative_prompt_embeds", None),
            uncond_prompt_embeds_mask=getattr(block_state, "negative_prompt_embeds_mask", None),
            num_inference_steps=block_state.num_inference_steps,
            step_index=i,
            t=t,
            additional_cond_kwargs=additional_cond_kwargs,
        )

        pred = pred[:, : block_state.latents.size(1)]
        pred_cond = pred_cond[:, : block_state.latents.size(1)]

        should_apply_area = bool(getattr(block_state, "area_composition_enabled", False))
        area_token_weights = getattr(block_state, "area_token_weights", None)
        valid_area_indices = getattr(block_state, "area_valid_indices", [])

        if should_apply_area and area_token_weights is not None and len(valid_area_indices) > 0:
            area_noise_preds = []

            for area_idx in valid_area_indices:
                area_pred, area_pred_cond = _run_cfg_denoiser(
                    components=components,
                    latent_model_input=block_state.latent_model_input,
                    timestep=block_state.timestep / 1000,
                    attention_kwargs=block_state.attention_kwargs,
                    cond_prompt_embeds=block_state.area_prompt_embeds[area_idx],
                    cond_prompt_embeds_mask=block_state.area_prompt_embeds_mask[area_idx],
                    uncond_prompt_embeds=(
                        block_state.area_negative_prompt_embeds[area_idx]
                        if getattr(block_state, "area_negative_prompt_embeds", None) is not None
                        else getattr(block_state, "negative_prompt_embeds", None)
                    ),
                    uncond_prompt_embeds_mask=(
                        block_state.area_negative_prompt_embeds_mask[area_idx]
                        if getattr(block_state, "area_negative_prompt_embeds_mask", None) is not None
                        else getattr(block_state, "negative_prompt_embeds_mask", None)
                    ),
                    num_inference_steps=block_state.num_inference_steps,
                    step_index=i,
                    t=t,
                    additional_cond_kwargs=additional_cond_kwargs,
                )
                area_pred = area_pred[:, : block_state.latents.size(1)]
                area_pred_cond = area_pred_cond[:, : block_state.latents.size(1)]

                area_noise_preds.append(_rescale_noise_prediction(pred=area_pred, pred_cond=area_pred_cond))

            area_weights = area_token_weights[: len(area_noise_preds)].to(device=pred.device, dtype=pred.dtype)
            base_weight = torch.ones((pred.shape[1],), device=pred.device, dtype=pred.dtype)
            merged_pred = pred * base_weight.unsqueeze(0).unsqueeze(-1)
            weight_sum = base_weight.unsqueeze(0).unsqueeze(-1)

            for area_idx, area_noise_pred in enumerate(area_noise_preds):
                weight = area_weights[area_idx].unsqueeze(0).unsqueeze(-1)
                merged_pred = merged_pred + area_noise_pred * weight
                weight_sum = weight_sum + weight

            pred = merged_pred / weight_sum.clamp(min=1e-6)

        block_state.noise_pred = _rescale_noise_prediction(pred=pred, pred_cond=pred_cond)
        return components, block_state


class QwenImageEditLoopDenoiser(ModularPipelineBlocks):
    model_name = "qwenimage-edit"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that denoise the latent input for the denoiser. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImageDenoiseLoopWrapper`)"
        )

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", QwenImageTransformer2DModel),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam.template("attention_kwargs"),
            InputParam.template("denoiser_input_fields"),
            InputParam(
                "img_shapes",
                required=True,
                type_hint=List[Tuple[int, int]],
                description="The shape of the image latents for RoPE calculation. Can be generated in prepare_additional_inputs step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        guider_inputs = {
            "encoder_hidden_states": (
                getattr(block_state, "prompt_embeds", None),
                getattr(block_state, "negative_prompt_embeds", None),
            ),
            "encoder_hidden_states_mask": (
                getattr(block_state, "prompt_embeds_mask", None),
                getattr(block_state, "negative_prompt_embeds_mask", None),
            ),
        }

        transformer_args = set(inspect.signature(components.transformer.forward).parameters.keys())
        additional_cond_kwargs = {}
        for field_name, field_value in block_state.denoiser_input_fields.items():
            if field_name in transformer_args and field_name not in guider_inputs:
                additional_cond_kwargs[field_name] = field_value
        block_state.additional_cond_kwargs.update(additional_cond_kwargs)

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs(guider_inputs)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = {input_name: getattr(guider_state_batch, input_name) for input_name in guider_inputs.keys()}

            # YiYi TODO: add cache context
            guider_state_batch.noise_pred = components.transformer(
                hidden_states=block_state.latent_model_input,
                timestep=block_state.timestep / 1000,
                attention_kwargs=block_state.attention_kwargs,
                return_dict=False,
                **cond_kwargs,
                **block_state.additional_cond_kwargs,
            )[0]

            components.guider.cleanup_models(components.transformer)

        guider_output = components.guider(guider_state)

        pred = guider_output.pred[:, : block_state.latents.size(1)]
        pred_cond = guider_output.pred_cond[:, : block_state.latents.size(1)]

        # apply guidance rescale
        pred_cond_norm = torch.norm(pred_cond, dim=-1, keepdim=True)
        pred_norm = torch.norm(pred, dim=-1, keepdim=True)
        block_state.noise_pred = pred * (pred_cond_norm / pred_norm)

        return components, block_state


# loop step:after denoiser
class QwenImageLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that updates the latents. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImageDenoiseLoopWrapper`)"
        )

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam.template("latents"),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        latents_dtype = block_state.latents.dtype
        block_state.latents = components.scheduler.step(
            block_state.noise_pred,
            t,
            block_state.latents,
            return_dict=False,
        )[0]

        if block_state.latents.dtype != latents_dtype:
            if torch.backends.mps.is_available():
                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                block_state.latents = block_state.latents.to(latents_dtype)

        return components, block_state


class QwenImageLoopAfterDenoiserInpaint(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return (
            "step within the denoising loop that updates the latents using mask and image_latents for inpainting. "
            "This block should be used to compose the `sub_blocks` attribute of a `LoopSequentialPipelineBlocks` "
            "object (e.g. `QwenImageDenoiseLoopWrapper`)"
        )

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "mask",
                required=True,
                type_hint=torch.Tensor,
                description="The mask to use for the inpainting process. Can be generated in inpaint prepare latents step.",
            ),
            InputParam.template("image_latents"),
            InputParam(
                "initial_noise",
                required=True,
                type_hint=torch.Tensor,
                description="The initial noise to use for the inpainting process. Can be generated in inpaint prepare latents step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam.template("latents"),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.init_latents_proper = block_state.image_latents
        if i < len(block_state.timesteps) - 1:
            block_state.noise_timestep = block_state.timesteps[i + 1]
            block_state.init_latents_proper = components.scheduler.scale_noise(
                block_state.init_latents_proper, torch.tensor([block_state.noise_timestep]), block_state.initial_noise
            )

        block_state.latents = (
            1 - block_state.mask
        ) * block_state.init_latents_proper + block_state.mask * block_state.latents

        return components, block_state


# ====================
# 2. DENOISE LOOP WRAPPER: define the denoising loop logic
# ====================
class QwenImageDenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return (
            "Pipeline block that iteratively denoise the latents over `timesteps`. "
            "The specific steps with each iteration can be customized with `sub_blocks` attributes"
        )

    @property
    def loop_expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def loop_inputs(self) -> List[InputParam]:
        return [
            InputParam(
                name="timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps to use for the denoising process. Can be generated in set_timesteps step.",
            ),
            InputParam.template("num_inference_steps", required=True),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.num_warmup_steps = max(
            len(block_state.timesteps) - block_state.num_inference_steps * components.scheduler.order, 0
        )

        block_state.additional_cond_kwargs = {}

        with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                if i == len(block_state.timesteps) - 1 or (
                    (i + 1) > block_state.num_warmup_steps and (i + 1) % components.scheduler.order == 0
                ):
                    progress_bar.update()

        self.set_block_state(state, block_state)

        return components, state


# ====================
# 3. DENOISE STEPS: compose the denoising loop with loop wrapper + loop steps
# ====================


# Qwen Image (text2image, image2image)


# auto_docstring
class QwenImageDenoiseStep(QwenImageDenoiseLoopWrapper):
    """
    Denoise step that iteratively denoise the latents.
      Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method At each iteration, it runs blocks
      defined in `sub_blocks` sequencially:
       - `QwenImageLoopBeforeDenoiser`
       - `QwenImageLoopDenoiser`
       - `QwenImageLoopAfterDenoiser`
      This block supports text2image and image2image tasks for QwenImage.

      Components:
          guider (`ClassifierFreeGuidance`) transformer (`QwenImageTransformer2DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process. Can be generated in set_timesteps step.
          num_inference_steps (`int`):
              The number of denoising steps.
          latents (`Tensor`):
              The initial latents to use for the denoising process. Can be generated in prepare_latent step.
          attention_kwargs (`Dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          img_shapes (`List`):
              The shape of the image latents for RoPE calculation. can be generated in prepare_additional_inputs step.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "qwenimage"

    block_classes = [
        QwenImageLoopBeforeDenoiser,
        QwenImageLoopDenoiser,
        QwenImageLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents.\n"
            "Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method\n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImageLoopBeforeDenoiser`\n"
            " - `QwenImageLoopDenoiser`\n"
            " - `QwenImageLoopAfterDenoiser`\n"
            "This block supports text2image and image2image tasks for QwenImage."
        )


class QwenImageAreaCompositionDenoiseStep(QwenImageDenoiseLoopWrapper):
    model_name = "qwenimage"

    block_classes = [
        QwenImageLoopBeforeDenoiser,
        QwenImageAreaCompositionLoopBeforeDenoiser,
        QwenImageAreaCompositionLoopDenoiser,
        QwenImageLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "before_area_composition", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step with area composition that iteratively denoise the latents.\n"
            "Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method\n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImageLoopBeforeDenoiser`\n"
            " - `QwenImageAreaCompositionLoopBeforeDenoiser`\n"
            " - `QwenImageAreaCompositionLoopDenoiser`\n"
            " - `QwenImageLoopAfterDenoiser`\n"
            "This block supports text2image and image2image tasks for QwenImage with area composition."
        )


# Qwen Image (inpainting)
# auto_docstring
class QwenImageInpaintDenoiseStep(QwenImageDenoiseLoopWrapper):
    """
    Denoise step that iteratively denoise the latents.
      Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method At each iteration, it runs blocks
      defined in `sub_blocks` sequencially:
       - `QwenImageLoopBeforeDenoiser`
       - `QwenImageLoopDenoiser`
       - `QwenImageLoopAfterDenoiser`
       - `QwenImageLoopAfterDenoiserInpaint`
      This block supports inpainting tasks for QwenImage.

      Components:
          guider (`ClassifierFreeGuidance`) transformer (`QwenImageTransformer2DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process. Can be generated in set_timesteps step.
          num_inference_steps (`int`):
              The number of denoising steps.
          latents (`Tensor`):
              The initial latents to use for the denoising process. Can be generated in prepare_latent step.
          attention_kwargs (`Dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          img_shapes (`List`):
              The shape of the image latents for RoPE calculation. can be generated in prepare_additional_inputs step.
          mask (`Tensor`):
              The mask to use for the inpainting process. Can be generated in inpaint prepare latents step.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          initial_noise (`Tensor`):
              The initial noise to use for the inpainting process. Can be generated in inpaint prepare latents step.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "qwenimage"
    block_classes = [
        QwenImageLoopBeforeDenoiser,
        QwenImageLoopDenoiser,
        QwenImageLoopAfterDenoiser,
        QwenImageLoopAfterDenoiserInpaint,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser", "after_denoiser_inpaint"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImageLoopBeforeDenoiser`\n"
            " - `QwenImageLoopDenoiser`\n"
            " - `QwenImageLoopAfterDenoiser`\n"
            " - `QwenImageLoopAfterDenoiserInpaint`\n"
            "This block supports inpainting tasks for QwenImage."
        )


# Qwen Image (text2image, image2image) with controlnet
# auto_docstring
class QwenImageControlNetDenoiseStep(QwenImageDenoiseLoopWrapper):
    """
    Denoise step that iteratively denoise the latents.
      Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method At each iteration, it runs blocks
      defined in `sub_blocks` sequencially:
       - `QwenImageLoopBeforeDenoiser`
       - `QwenImageLoopBeforeDenoiserControlNet`
       - `QwenImageLoopDenoiser`
       - `QwenImageLoopAfterDenoiser`
      This block supports text2img/img2img tasks with controlnet for QwenImage.

      Components:
          guider (`ClassifierFreeGuidance`) controlnet (`QwenImageControlNetModel`) transformer
          (`QwenImageTransformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process. Can be generated in set_timesteps step.
          num_inference_steps (`int`):
              The number of denoising steps.
          latents (`Tensor`):
              The initial latents to use for the denoising process. Can be generated in prepare_latent step.
          control_image_latents (`Tensor`):
              The control image to use for the denoising process. Can be generated in prepare_controlnet_inputs step.
          controlnet_conditioning_scale (`float`, *optional*, defaults to 1.0):
              Scale for ControlNet conditioning. (updated in prepare_controlnet_inputs step.)
          controlnet_keep (`List`):
              The controlnet keep values. Can be generated in prepare_controlnet_inputs step.
          attention_kwargs (`Dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          img_shapes (`List`):
              The shape of the image latents for RoPE calculation. can be generated in prepare_additional_inputs step.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "qwenimage"
    block_classes = [
        QwenImageLoopBeforeDenoiser,
        QwenImageLoopBeforeDenoiserControlNet,
        QwenImageLoopDenoiser,
        QwenImageLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "before_denoiser_controlnet", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImageLoopBeforeDenoiser`\n"
            " - `QwenImageLoopBeforeDenoiserControlNet`\n"
            " - `QwenImageLoopDenoiser`\n"
            " - `QwenImageLoopAfterDenoiser`\n"
            "This block supports text2img/img2img tasks with controlnet for QwenImage."
        )


# Qwen Image (inpainting) with controlnet
# auto_docstring
class QwenImageInpaintControlNetDenoiseStep(QwenImageDenoiseLoopWrapper):
    """
    Denoise step that iteratively denoise the latents.
      Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method At each iteration, it runs blocks
      defined in `sub_blocks` sequencially:
       - `QwenImageLoopBeforeDenoiser`
       - `QwenImageLoopBeforeDenoiserControlNet`
       - `QwenImageLoopDenoiser`
       - `QwenImageLoopAfterDenoiser`
       - `QwenImageLoopAfterDenoiserInpaint`
      This block supports inpainting tasks with controlnet for QwenImage.

      Components:
          guider (`ClassifierFreeGuidance`) controlnet (`QwenImageControlNetModel`) transformer
          (`QwenImageTransformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process. Can be generated in set_timesteps step.
          num_inference_steps (`int`):
              The number of denoising steps.
          latents (`Tensor`):
              The initial latents to use for the denoising process. Can be generated in prepare_latent step.
          control_image_latents (`Tensor`):
              The control image to use for the denoising process. Can be generated in prepare_controlnet_inputs step.
          controlnet_conditioning_scale (`float`, *optional*, defaults to 1.0):
              Scale for ControlNet conditioning. (updated in prepare_controlnet_inputs step.)
          controlnet_keep (`List`):
              The controlnet keep values. Can be generated in prepare_controlnet_inputs step.
          attention_kwargs (`Dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          img_shapes (`List`):
              The shape of the image latents for RoPE calculation. can be generated in prepare_additional_inputs step.
          mask (`Tensor`):
              The mask to use for the inpainting process. Can be generated in inpaint prepare latents step.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          initial_noise (`Tensor`):
              The initial noise to use for the inpainting process. Can be generated in inpaint prepare latents step.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "qwenimage"
    block_classes = [
        QwenImageLoopBeforeDenoiser,
        QwenImageLoopBeforeDenoiserControlNet,
        QwenImageLoopDenoiser,
        QwenImageLoopAfterDenoiser,
        QwenImageLoopAfterDenoiserInpaint,
    ]
    block_names = [
        "before_denoiser",
        "before_denoiser_controlnet",
        "denoiser",
        "after_denoiser",
        "after_denoiser_inpaint",
    ]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImageLoopBeforeDenoiser`\n"
            " - `QwenImageLoopBeforeDenoiserControlNet`\n"
            " - `QwenImageLoopDenoiser`\n"
            " - `QwenImageLoopAfterDenoiser`\n"
            " - `QwenImageLoopAfterDenoiserInpaint`\n"
            "This block supports inpainting tasks with controlnet for QwenImage."
        )


# Qwen Image Edit (image2image)
# auto_docstring
class QwenImageEditDenoiseStep(QwenImageDenoiseLoopWrapper):
    """
    Denoise step that iteratively denoise the latents.
      Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method At each iteration, it runs blocks
      defined in `sub_blocks` sequencially:
       - `QwenImageEditLoopBeforeDenoiser`
       - `QwenImageEditLoopDenoiser`
       - `QwenImageLoopAfterDenoiser`
      This block supports QwenImage Edit.

      Components:
          guider (`ClassifierFreeGuidance`) transformer (`QwenImageTransformer2DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process. Can be generated in set_timesteps step.
          num_inference_steps (`int`):
              The number of denoising steps.
          latents (`Tensor`):
              The initial latents to use for the denoising process. Can be generated in prepare_latent step.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          attention_kwargs (`Dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          img_shapes (`List`):
              The shape of the image latents for RoPE calculation. Can be generated in prepare_additional_inputs step.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "qwenimage-edit"
    block_classes = [
        QwenImageEditLoopBeforeDenoiser,
        QwenImageEditLoopDenoiser,
        QwenImageLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImageEditLoopBeforeDenoiser`\n"
            " - `QwenImageEditLoopDenoiser`\n"
            " - `QwenImageLoopAfterDenoiser`\n"
            "This block supports QwenImage Edit."
        )


# Qwen Image Edit (inpainting)
# auto_docstring
class QwenImageEditInpaintDenoiseStep(QwenImageDenoiseLoopWrapper):
    """
    Denoise step that iteratively denoise the latents.
      Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method At each iteration, it runs blocks
      defined in `sub_blocks` sequencially:
       - `QwenImageEditLoopBeforeDenoiser`
       - `QwenImageEditLoopDenoiser`
       - `QwenImageLoopAfterDenoiser`
       - `QwenImageLoopAfterDenoiserInpaint`
      This block supports inpainting tasks for QwenImage Edit.

      Components:
          guider (`ClassifierFreeGuidance`) transformer (`QwenImageTransformer2DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process. Can be generated in set_timesteps step.
          num_inference_steps (`int`):
              The number of denoising steps.
          latents (`Tensor`):
              The initial latents to use for the denoising process. Can be generated in prepare_latent step.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          attention_kwargs (`Dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          img_shapes (`List`):
              The shape of the image latents for RoPE calculation. Can be generated in prepare_additional_inputs step.
          mask (`Tensor`):
              The mask to use for the inpainting process. Can be generated in inpaint prepare latents step.
          initial_noise (`Tensor`):
              The initial noise to use for the inpainting process. Can be generated in inpaint prepare latents step.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "qwenimage-edit"
    block_classes = [
        QwenImageEditLoopBeforeDenoiser,
        QwenImageEditLoopDenoiser,
        QwenImageLoopAfterDenoiser,
        QwenImageLoopAfterDenoiserInpaint,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser", "after_denoiser_inpaint"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImageEditLoopBeforeDenoiser`\n"
            " - `QwenImageEditLoopDenoiser`\n"
            " - `QwenImageLoopAfterDenoiser`\n"
            " - `QwenImageLoopAfterDenoiserInpaint`\n"
            "This block supports inpainting tasks for QwenImage Edit."
        )


# Qwen Image Layered (image2image)
# auto_docstring
class QwenImageLayeredDenoiseStep(QwenImageDenoiseLoopWrapper):
    """
    Denoise step that iteratively denoise the latents.
      Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method At each iteration, it runs blocks
      defined in `sub_blocks` sequencially:
       - `QwenImageEditLoopBeforeDenoiser`
       - `QwenImageEditLoopDenoiser`
       - `QwenImageLoopAfterDenoiser`
      This block supports QwenImage Layered.

      Components:
          guider (`ClassifierFreeGuidance`) transformer (`QwenImageTransformer2DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          timesteps (`Tensor`):
              The timesteps to use for the denoising process. Can be generated in set_timesteps step.
          num_inference_steps (`int`):
              The number of denoising steps.
          latents (`Tensor`):
              The initial latents to use for the denoising process. Can be generated in prepare_latent step.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          attention_kwargs (`Dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          img_shapes (`List`):
              The shape of the image latents for RoPE calculation. Can be generated in prepare_additional_inputs step.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "qwenimage-layered"
    block_classes = [
        QwenImageEditLoopBeforeDenoiser,
        QwenImageEditLoopDenoiser,
        QwenImageLoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. \n"
            "Its loop logic is defined in `QwenImageDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequencially:\n"
            " - `QwenImageEditLoopBeforeDenoiser`\n"
            " - `QwenImageEditLoopDenoiser`\n"
            " - `QwenImageLoopAfterDenoiser`\n"
            "This block supports QwenImage Layered."
        )
