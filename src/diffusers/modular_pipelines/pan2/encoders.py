# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import json

import PIL.Image
import torch
from transformers import Qwen2Tokenizer, Qwen3_5TextModel

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLWan
from ...utils import logging
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import PAN2ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# The negative prompt PAN 2 was released with. Its `time_range_sec` fields are filled in per call to span the whole
# clip, see `get_pan2_default_negative_prompt`.
PAN2_DEFAULT_NEGATIVE_PROMPT = {
    "video_style": "low quality, blur, softness, reduced detail, noise, compression, clipped highlights, "
    "blurred sidebar, black border, letterboxing, watermark, subtitle, AIGC text, oily waxy "
    "plastic texture, oversaturation or color cast, color clipping or chroma bleeding, flicker, "
    "jitter, ghosting, motion smear, choppy or laggy motion, slow motion, Ken Burns effect, "
    "unmotivated slow zoom over a frozen scene, pan across a still image, global frame scaling "
    "or drifting, simulated movement from a static photograph, repeated micro-motion loop, "
    "discontinuous shot, scene cut, hard cut, crossfade, fade-in, fade-out, fading transition, "
    "scene replacement, spontaneous object or person pop-in or disappearance, duplicated or "
    "missing people or props, identity or count drift, rigid-object deformation or morphing, "
    "bending or melting rigid structures, structural shape or component drift",
    "lighting": "overexposure, clipped highlights, underexposure, harsh lighting, overly deep skin shadows, "
    "brightness fading, excessive lighting saturation, chroma-clipped highlights, neon-like skin "
    "tones, colored spill overwhelming local material colors, oversaturated bloom, unnatural neon "
    "color casts, hue drift, chroma noise, bad white balance, color pulsing, flickering light, "
    "unnatural lighting",
    "perspective_view": "warped geometry, distorted perspective, inconsistent spatial relationships, "
    "unmotivated layout changes, camera teleport or jump, viewpoint reset, wrong depth or "
    "occlusion",
    "subjects_lifecycle": [
        {
            "subject_id": "<|Subject1|>",
            "display_name": "malformed inconsistent subject",
            "subject_attribute": "broken anatomy, distorted body proportions, reversed pelvis, "
            "twisted torso, deformed arms or legs, reversed joints, fused "
            "feet or toes, warped or collapsing faces, transparent or "
            "floating head and body, missing body parts, detached or "
            "disembodied body parts, floating hands or limbs, detached "
            "articulated parts, disconnected joints or supports, hands "
            "disconnected from wrists or forearms, discontinuous "
            "limb-to-torso connections, blurred or fused fingers, missing "
            "thumbs, extra or missing fingers, malformed wrists, extra "
            "limbs, duplicated hands, mirrored face, identity drift, wrong "
            "relative sizes, human-object fusion, unintended body-material "
            "fusion, loss of separate material boundaries, opaque white "
            "nail polish, gray nail polish, chalky opaque fingernails, "
            "artificial acrylic nails",
        }
    ],
    "init_frame_visual_description": "blur or softness, reduced detail, noise or compression artifacts, clipped "
    "highlights, blurred sidebar, black border or letterboxing, oily waxy "
    "plastic surfaces, oversaturation or color cast, exaggerated local "
    "contrast, exposure fading, bad white balance, smeared dense textures, "
    "blurred or distorted face, transparent or missing body parts, detached or "
    "disembodied body parts, floating hands or limbs, detached articulated "
    "parts, disconnected joints or supports, disconnected wrists or forearms, "
    "discontinuous limb-to-torso connections, twisted torso, fused or missing "
    "fingers, missing thumbs, malformed wrists, duplicated limbs, deformed "
    "arms or legs, reversed knees, fused toes, warped geometry, deformed or "
    "asymmetrically warped rigid objects, bent rigid structures, incorrect "
    "rigid part geometry, topology, proportions, or component count, wrong "
    "occlusion, floating or hovering objects, unsupported objects, object or "
    "human-object fusion, unintended fusion of distinct surfaces or layers, "
    "loss of separate material boundaries, merged objects, hand-object "
    "penetration, object-object penetration, penetration, collision, "
    "impossible contact, wrong sizes.",
    "events": [
        {
            "event_index": 0,
            "time_range_sec": "0 to 10",
            "event_description": "blur, softness, reduced detail, compression artifacts, clipped "
            "highlights, flicker, jitter, ghosting, motion smear, choppy or laggy "
            "motion, slow motion, Ken Burns effect, unmotivated slow zoom over a "
            "frozen scene, pan across a still image, global frame scaling or drifting, "
            "simulated movement from a static photograph, repeated micro-motion loop, "
            "oily oversaturated or color-pulsing surfaces, shimmering surface detail, "
            "fingers fusing or changing count, hands smearing, facial features "
            "merging, reversed joints, upper body rotating independently of attached "
            "hips and feet, corkscrewed waist, limbs changing length or attachment "
            "after occlusion, transparent body or head, detached or disembodied body "
            "parts, floating hands or limbs, detached articulated parts, disconnected "
            "joints or supports, hands disconnected from wrists or forearms, body "
            "parts moving independently, vanishing limb-to-torso connections, merged "
            "people or limbs, limbs fused into torso, merged or fused feet and toes, "
            "floating or hovering objects, unsupported objects, object or human-object "
            "fusion, unintended fusion of distinct surfaces or layers, loss of "
            "separate material boundaries, merged objects, hand-object penetration, "
            "object-object penetration, penetration or clipping, objects colliding or "
            "morphing, rigid objects tearing, fragmenting, fusing, bending, melting or "
            "changing dimensions, topology, silhouette, component count or layout, "
            "subjects changing identity, splitting or duplicating, objects appearing "
            "or disappearing without entry, exit or occlusion, changed identity or "
            "count after occlusion, state changes inconsistent with the causing "
            "action, broken physics, uncaused deformation or motion, state changes "
            "preceding the causing interaction, loss of coupling during contact, pose "
            "jumps, discontinuous shot, camera jump, hard scene cut, crossfade, "
            "fade-in, fade-out, fading transition, scene replacement or viewpoint "
            "reset.",
            "camera_motion": [
                {"time_range_sec": "0 to 10", "camera_motion": ["camera_jump", "hard_cut", "viewpoint_reset"]}
            ],
            "contained_actions": [
                {
                    "time_range_sec": "0 to 10",
                    "action": "<|Subject1|> fingers fusing during grip changes, palms "
                    "merging with props, wrists collapsing, crossed limbs fusing, "
                    "intersecting or swapping attachment, duplicated hands or "
                    "feet, detached or disembodied hands or limbs, hands "
                    "disconnected from wrists or forearms, discontinuous "
                    "limb-to-torso connections, merged or fused feet and toes, "
                    "unstable contact, unintended grip slip, hand-object "
                    "penetration, body-object penetration, passage through solid "
                    "surfaces instead of openings, object-object penetration, "
                    "merged or fused objects, rigid objects fragmenting, fusing, "
                    "smearing or changing dimensions during motion, unintended "
                    "fusion of distinct surfaces or layers, loss of separate "
                    "material boundaries, floating or hovering objects, "
                    "unsupported objects, objects duplicating, splitting or "
                    "disappearing during interaction or occlusion, unchanged "
                    "action result, pose jump",
                }
            ],
        }
    ],
    "quality": {
        "video_quality": "low resolution, blur, softness, reduced detail, noise, compression, clipped "
        "highlights, blurred sidebar, black border, letterboxing, oily waxy plastic "
        "surfaces, greasy highlights, oversaturated colors, faded or washed-out "
        "colors, exposure fading, smeared faces, hands or held objects, surface detail "
        "smearing or swimming, material or existing markings changing without a "
        "physical cause, color or shape morphing, object pop-in or pop-out, sudden "
        "disappearance, missing persistent objects, object duplication or count drift, "
        "rigid-object warping or deformation, changing rigid proportions, topology, "
        "silhouette, or component layout",
        "text_logos_overlays": "watermark, subtitle, captions, AIGC text or logo",
        "artifacts": [
            "watermark",
            "template_or_text_dominated",
            "talking_head",
            "visual_quality_issue",
            "edited_or_discontinuous",
            "video_game",
            "animation",
            "ai_generated_content",
            "black_border",
            "ken_burn",
            "time_lapse",
            "blurred_sidebar",
            "hard_scene_cuts",
        ],
    },
}


def get_pan2_default_negative_prompt(num_frames: int, fps: float) -> str:
    r"""PAN 2's default negative prompt as compact one-line JSON, with every time range spanning the clip."""
    span = f"0 to {(num_frames - 1) / fps:g}"

    def fill(node):
        if isinstance(node, dict):
            return {key: span if key == "time_range_sec" else fill(value) for key, value in node.items()}
        if isinstance(node, list):
            return [fill(item) for item in node]
        return node

    return json.dumps(fill(PAN2_DEFAULT_NEGATIVE_PROMPT), ensure_ascii=False, separators=(",", ":"))


def get_pan2_prompt_embeds(
    text_encoder: Qwen3_5TextModel,
    tokenizer: Qwen2Tokenizer,
    prompt: str,
    system_message: str,
    num_hidden_layers_to_skip: int,
    max_sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    # An empty prompt is encoded as a single space, as in training.
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt or " "}]

    # PAN 2 conditions on the tokens after the `<|im_start|>user\n` header: the prompt and the assistant header after it.
    header_ids = tokenizer("<|im_start|>user\n", add_special_tokens=False).input_ids
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True
    ).input_ids
    crop_start = next(
        i + len(header_ids) for i in range(len(input_ids)) if input_ids[i : i + len(header_ids)] == header_ids
    )
    input_ids = torch.tensor([input_ids[: crop_start + max_sequence_length]], device=device)

    prompt_embeds = text_encoder(input_ids=input_ids, output_hidden_states=True).hidden_states[
        -(num_hidden_layers_to_skip + 1)
    ]
    return prompt_embeds[:, crop_start:]


class PAN2TextEncoderStep(ModularPipelineBlocks):
    model_name = "pan2"

    @property
    def description(self) -> str:
        return (
            "Text encoder step that encodes the prompt, and the negative prompt when guidance needs one or one is "
            "passed, with Qwen3.5."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3_5TextModel),
            ComponentSpec("tokenizer", Qwen2Tokenizer),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [ConfigSpec(name="enable_safety_checker", default=True)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", description="The prompt to guide generation, a single string."),
            InputParam.template(
                "negative_prompt",
                description=(
                    "The prompt not to guide generation, a single string. When guidance needs one and none (or an "
                    "empty one) is passed, PAN 2's default negative prompt is used, its time ranges set to the clip's "
                    "duration."
                ),
            ),
            InputParam.template("max_sequence_length", default=1536),
            InputParam(
                "num_frames",
                type_hint=int,
                default=313,
                description="Number of video frames to generate, of the form `4 * n + 1`.",
            ),
            InputParam(
                "fps",
                type_hint=float,
                default=24.0,
                description="The frame rate of the video, which sets its duration in the default negative prompt.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam.template("negative_prompt_embeds"),
        ]

    @staticmethod
    def check_inputs(prompt, negative_prompt):
        # Every prompt is its own attention sequence without padding, so a call takes a single prompt.
        if not isinstance(prompt, str):
            raise ValueError(f"`prompt` must be a single string, got {type(prompt)}.")
        if negative_prompt is not None and not isinstance(negative_prompt, str):
            raise ValueError(f"`negative_prompt` must be a single string, got {type(negative_prompt)}.")

    @torch.no_grad()
    def __call__(
        self, components: PAN2ModularPipeline, state: PipelineState
    ) -> tuple[PAN2ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state.prompt, block_state.negative_prompt)
        device = components._execution_device

        if components.requires_safety_checker:
            components.check_safety_checker_loaded()
            if not components.safety_checker.check_text_safety(block_state.prompt):
                raise ValueError("The PAN 2 guardrail detected unsafe content in the prompt.")

        block_state.prompt_embeds = get_pan2_prompt_embeds(
            components.text_encoder,
            components.tokenizer,
            block_state.prompt,
            system_message=components.system_message,
            num_hidden_layers_to_skip=components.text_encoder_num_hidden_layers_to_skip,
            max_sequence_length=block_state.max_sequence_length,
            device=device,
        )

        # The guider is not a component of this block: in the pipeline, `requires_unconditional_embeds` reads the denoise
        # step's guider and the negative prompt defaults to PAN 2's; standalone, there is no guider and the negative
        # prompt is encoded only when it is passed.
        block_state.negative_prompt_embeds = None
        guidance_off = getattr(components, "guider", None) is not None and not components.requires_unconditional_embeds
        negative_prompt = block_state.negative_prompt
        if not negative_prompt and components.requires_unconditional_embeds:
            negative_prompt = get_pan2_default_negative_prompt(block_state.num_frames, block_state.fps)
        if negative_prompt and guidance_off:
            logger.warning("`negative_prompt` is ignored because the guider does not use unconditional embeddings.")
        elif negative_prompt:
            block_state.negative_prompt_embeds = get_pan2_prompt_embeds(
                components.text_encoder,
                components.tokenizer,
                negative_prompt,
                system_message=components.system_message,
                num_hidden_layers_to_skip=components.text_encoder_num_hidden_layers_to_skip,
                max_sequence_length=block_state.max_sequence_length,
                device=device,
            )

        self.set_block_state(state, block_state)
        return components, state


def resize_and_center_crop(image: PIL.Image.Image, height: int, width: int) -> PIL.Image.Image:
    r"""Scale the image to cover the `(height, width)` canvas, rounding its size, and center crop it, as PAN 2 was trained."""
    scale = max(width / image.width, height / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), resample=PIL.Image.LANCZOS)
    left, top = round((image.width - width) / 2), round((image.height - height) / 2)
    return image.crop((left, top, left + width, top + height))


class PAN2VaeEncoderStep(ModularPipelineBlocks):
    model_name = "pan2"

    @property
    def description(self) -> str:
        return (
            "VAE encoder step that resizes and center-crops the first frame onto the target canvas and encodes it into "
            "normalized image latents."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template(
                "image", required=True, type_hint=PIL.Image.Image, description="The first frame, a single image."
            ),
            InputParam.template("height", description="The height in pixels of the generated video, 704 by default."),
            InputParam.template("width", description="The width in pixels of the generated video, 1248 by default."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "image_latents",
                type_hint=torch.Tensor,
                description="The normalized latents of the first frame, of shape `(1, 48, 1, h, w)`.",
            ),
        ]

    @staticmethod
    def check_inputs(image):
        # A call takes a single prompt, so it takes a single first frame too.
        if not isinstance(image, PIL.Image.Image):
            raise ValueError(f"`image` must be a single PIL image, got {type(image)}.")

    @torch.no_grad()
    def __call__(
        self, components: PAN2ModularPipeline, state: PipelineState
    ) -> tuple[PAN2ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state.image)
        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width
        device = components._execution_device
        vae = components.vae

        image = resize_and_center_crop(block_state.image, block_state.height, block_state.width)
        image = components.video_processor.preprocess(image, height=block_state.height, width=block_state.width)
        image = image.unsqueeze(2).to(device=device, dtype=vae.dtype)
        image_latents = vae.encode(image).latent_dist.mode()

        latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(image_latents)
        latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(image_latents)
        block_state.image_latents = (image_latents - latents_mean) / latents_std

        self.set_block_state(state, block_state)
        return components, state
