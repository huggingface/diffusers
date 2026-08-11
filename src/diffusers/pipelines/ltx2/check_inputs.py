# Copyright 2026 Lightricks and The HuggingFace Team. All rights reserved.
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

import PIL.Image

from .utils import resolve_default_image_crf


class LTX2CheckInputsMixin:
    """Shared input validation for LTX-2 generation pipelines."""

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_on_step_end_tensor_inputs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        spatio_temporal_guidance_blocks=None,
        stg_scale=None,
        audio_stg_scale=None,
        system_prompt=None,
        enable_prompt_enhancement=None,
        num_frames=None,
        min_seconds=1.0,
        max_seconds=20.0,
        image=None,
        image_crf=None,
        latents=None,
        audio_latents=None,
    ):
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

        if latents is not None and latents.ndim != 5:
            raise ValueError(
                f"Only unpacked (5D) video latents of shape `[batch_size, latent_channels, latent_frames,"
                f" latent_height, latent_width] are supported, but got {latents.ndim} dims. If you have packed (3D)"
                f" latents, please unpack them (e.g. using the `_unpack_latents` method)."
            )
        if audio_latents is not None and audio_latents.ndim != 4:
            raise ValueError(
                f"Only unpacked (4D) audio latents of shape `[batch_size, num_channels, audio_length, mel_bins] are"
                f" supported, but got {audio_latents.ndim} dims. If you have packed (3D) latents, please unpack them"
                f" (e.g. using the `_unpack_audio_latents` method)."
            )

        if ((stg_scale > 0.0) or (audio_stg_scale > 0.0)) and not spatio_temporal_guidance_blocks:
            raise ValueError(
                "Spatio-Temporal Guidance (STG) is specified but no STG blocks are supplied. Please supply a list of"
                "block indices at which to apply STG in `spatio_temporal_guidance_blocks`"
            )

        if (
            enable_prompt_enhancement
            and prompt is not None
            and system_prompt is None
            and getattr(self, "prompt_enhancer", None) is None
        ):
            raise ValueError(
                "`system_prompt` must be supplied to enable prompt enhancement when no dedicated "
                "`prompt_enhancer` component is configured (LTX-2.0/2.3)."
            )

        if min_seconds >= max_seconds:
            raise ValueError(
                f"`min_seconds` ({min_seconds}) must be less than `max_seconds` ({max_seconds})."
                " A collapsed range leaves no room for a prediction, and cannot generally be satisfied by a frame"
                " count on the VAE's temporal grid."
            )

        # Auto-duration path: `num_frames` omitted on a pipeline that has a `duration_head`.
        if num_frames is None and getattr(self, "duration_head", None) is not None:
            num_prompts = len(prompt) if isinstance(prompt, list) else 1 if prompt is not None else len(prompt_embeds)
            if num_prompts > 1:
                raise ValueError(
                    f"`num_frames` was omitted so the duration head would auto-predict, but {num_prompts} prompts were"
                    " supplied. The duration head predicts one duration, and prompts with different natural lengths"
                    " cannot share a single frame count. Call the pipeline once per prompt, or pass `num_frames` as an"
                    " integer."
                )

        if latents is None and image is not None:
            crf = image_crf if image_crf is not None else resolve_default_image_crf(self.text_encoder)
            if crf != 0 and not isinstance(image, PIL.Image.Image):
                raise ValueError(
                    f"`image_crf` re-compression requires a `PIL.Image.Image` input, got {type(image)}. "
                    "Pass a PIL image, or set `image_crf=0` to skip re-compression."
                )
