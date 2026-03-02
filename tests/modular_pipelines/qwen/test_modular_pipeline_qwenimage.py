# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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


import PIL
import pytest

from diffusers.modular_pipelines import (
    QwenImageAutoBlocks,
    QwenImageEditAutoBlocks,
    QwenImageEditModularPipeline,
    QwenImageEditPlusAutoBlocks,
    QwenImageEditPlusModularPipeline,
    QwenImageModularPipeline,
)

from ...testing_utils import torch_device
from ..test_modular_pipelines_common import ModularGuiderTesterMixin, ModularPipelineTesterMixin


QWEN_IMAGE_TEXT2IMAGE_WORKFLOWS = {
    "text2image": [
        ("text_encoder", "QwenImageTextEncoderStep"),
        ("denoise.input", "QwenImageTextInputsStep"),
        ("denoise.prepare_latents", "QwenImagePrepareLatentsStep"),
        ("denoise.set_timesteps", "QwenImageSetTimestepsStep"),
        ("denoise.prepare_rope_inputs", "QwenImageRoPEInputsStep"),
        ("denoise.denoise", "QwenImageDenoiseStep"),
        ("denoise.after_denoise", "QwenImageAfterDenoiseStep"),
        ("decode.decode", "QwenImageDecoderStep"),
        ("decode.postprocess", "QwenImageProcessImagesOutputStep"),
    ],
    "image2image": [
        ("text_encoder", "QwenImageTextEncoderStep"),
        ("vae_encoder.preprocess", "QwenImageProcessImagesInputStep"),
        ("vae_encoder.encode", "QwenImageVaeEncoderStep"),
        ("denoise.input.text_inputs", "QwenImageTextInputsStep"),
        ("denoise.input.additional_inputs", "QwenImageAdditionalInputsStep"),
        ("denoise.prepare_latents", "QwenImagePrepareLatentsStep"),
        ("denoise.set_timesteps", "QwenImageSetTimestepsWithStrengthStep"),
        ("denoise.prepare_img2img_latents", "QwenImagePrepareLatentsWithStrengthStep"),
        ("denoise.prepare_rope_inputs", "QwenImageRoPEInputsStep"),
        ("denoise.denoise", "QwenImageDenoiseStep"),
        ("denoise.after_denoise", "QwenImageAfterDenoiseStep"),
        ("decode.decode", "QwenImageDecoderStep"),
        ("decode.postprocess", "QwenImageProcessImagesOutputStep"),
    ],
    "inpainting": [
        ("text_encoder", "QwenImageTextEncoderStep"),
        ("vae_encoder.preprocess", "QwenImageInpaintProcessImagesInputStep"),
        ("vae_encoder.encode", "QwenImageVaeEncoderStep"),
        ("denoise.input.text_inputs", "QwenImageTextInputsStep"),
        ("denoise.input.additional_inputs", "QwenImageAdditionalInputsStep"),
        ("denoise.prepare_latents", "QwenImagePrepareLatentsStep"),
        ("denoise.set_timesteps", "QwenImageSetTimestepsWithStrengthStep"),
        ("denoise.prepare_inpaint_latents.add_noise_to_latents", "QwenImagePrepareLatentsWithStrengthStep"),
        ("denoise.prepare_inpaint_latents.create_mask_latents", "QwenImageCreateMaskLatentsStep"),
        ("denoise.prepare_rope_inputs", "QwenImageRoPEInputsStep"),
        ("denoise.denoise", "QwenImageInpaintDenoiseStep"),
        ("denoise.after_denoise", "QwenImageAfterDenoiseStep"),
        ("decode.decode", "QwenImageDecoderStep"),
        ("decode.postprocess", "QwenImageInpaintProcessImagesOutputStep"),
    ],
    "controlnet_text2image": [
        ("text_encoder", "QwenImageTextEncoderStep"),
        ("controlnet_vae_encoder", "QwenImageControlNetVaeEncoderStep"),
        ("denoise.input", "QwenImageTextInputsStep"),
        ("denoise.controlnet_input", "QwenImageControlNetInputsStep"),
        ("denoise.prepare_latents", "QwenImagePrepareLatentsStep"),
        ("denoise.set_timesteps", "QwenImageSetTimestepsStep"),
        ("denoise.prepare_rope_inputs", "QwenImageRoPEInputsStep"),
        ("denoise.controlnet_before_denoise", "QwenImageControlNetBeforeDenoiserStep"),
        ("denoise.controlnet_denoise", "QwenImageControlNetDenoiseStep"),
        ("denoise.after_denoise", "QwenImageAfterDenoiseStep"),
        ("decode.decode", "QwenImageDecoderStep"),
        ("decode.postprocess", "QwenImageProcessImagesOutputStep"),
    ],
    "controlnet_image2image": [
        ("text_encoder", "QwenImageTextEncoderStep"),
        ("vae_encoder.preprocess", "QwenImageProcessImagesInputStep"),
        ("vae_encoder.encode", "QwenImageVaeEncoderStep"),
        ("controlnet_vae_encoder", "QwenImageControlNetVaeEncoderStep"),
        ("denoise.input.text_inputs", "QwenImageTextInputsStep"),
        ("denoise.input.additional_inputs", "QwenImageAdditionalInputsStep"),
        ("denoise.controlnet_input", "QwenImageControlNetInputsStep"),
        ("denoise.prepare_latents", "QwenImagePrepareLatentsStep"),
        ("denoise.set_timesteps", "QwenImageSetTimestepsWithStrengthStep"),
        ("denoise.prepare_img2img_latents", "QwenImagePrepareLatentsWithStrengthStep"),
        ("denoise.prepare_rope_inputs", "QwenImageRoPEInputsStep"),
        ("denoise.controlnet_before_denoise", "QwenImageControlNetBeforeDenoiserStep"),
        ("denoise.controlnet_denoise", "QwenImageControlNetDenoiseStep"),
        ("denoise.after_denoise", "QwenImageAfterDenoiseStep"),
        ("decode.decode", "QwenImageDecoderStep"),
        ("decode.postprocess", "QwenImageProcessImagesOutputStep"),
    ],
    "controlnet_inpainting": [
        ("text_encoder", "QwenImageTextEncoderStep"),
        ("vae_encoder.preprocess", "QwenImageInpaintProcessImagesInputStep"),
        ("vae_encoder.encode", "QwenImageVaeEncoderStep"),
        ("controlnet_vae_encoder", "QwenImageControlNetVaeEncoderStep"),
        ("denoise.input.text_inputs", "QwenImageTextInputsStep"),
        ("denoise.input.additional_inputs", "QwenImageAdditionalInputsStep"),
        ("denoise.controlnet_input", "QwenImageControlNetInputsStep"),
        ("denoise.prepare_latents", "QwenImagePrepareLatentsStep"),
        ("denoise.set_timesteps", "QwenImageSetTimestepsWithStrengthStep"),
        ("denoise.prepare_inpaint_latents.add_noise_to_latents", "QwenImagePrepareLatentsWithStrengthStep"),
        ("denoise.prepare_inpaint_latents.create_mask_latents", "QwenImageCreateMaskLatentsStep"),
        ("denoise.prepare_rope_inputs", "QwenImageRoPEInputsStep"),
        ("denoise.controlnet_before_denoise", "QwenImageControlNetBeforeDenoiserStep"),
        ("denoise.controlnet_denoise", "QwenImageInpaintControlNetDenoiseStep"),
        ("denoise.after_denoise", "QwenImageAfterDenoiseStep"),
        ("decode.decode", "QwenImageDecoderStep"),
        ("decode.postprocess", "QwenImageInpaintProcessImagesOutputStep"),
    ],
}


class TestQwenImageModularPipelineFast(ModularPipelineTesterMixin, ModularGuiderTesterMixin):
    pipeline_class = QwenImageModularPipeline
    pipeline_blocks_class = QwenImageAutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-qwenimage-modular"

    params = frozenset(["prompt", "height", "width", "negative_prompt", "attention_kwargs", "image", "mask_image"])
    batch_params = frozenset(["prompt", "negative_prompt", "image", "mask_image"])
    expected_workflow_blocks = QWEN_IMAGE_TEXT2IMAGE_WORKFLOWS

    def get_dummy_inputs(self):
        generator = self.get_generator()
        inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=5e-4)


QWEN_IMAGE_EDIT_WORKFLOWS = {
    "image_conditioned": [
        ("text_encoder.resize", "QwenImageEditResizeStep"),
        ("text_encoder.encode", "QwenImageEditTextEncoderStep"),
        ("vae_encoder.resize", "QwenImageEditResizeStep"),
        ("vae_encoder.preprocess", "QwenImageEditProcessImagesInputStep"),
        ("vae_encoder.encode", "QwenImageVaeEncoderStep"),
        ("denoise.input.text_inputs", "QwenImageTextInputsStep"),
        ("denoise.input.additional_inputs", "QwenImageAdditionalInputsStep"),
        ("denoise.prepare_latents", "QwenImagePrepareLatentsStep"),
        ("denoise.set_timesteps", "QwenImageSetTimestepsStep"),
        ("denoise.prepare_rope_inputs", "QwenImageEditRoPEInputsStep"),
        ("denoise.denoise", "QwenImageEditDenoiseStep"),
        ("denoise.after_denoise", "QwenImageAfterDenoiseStep"),
        ("decode.decode", "QwenImageDecoderStep"),
        ("decode.postprocess", "QwenImageProcessImagesOutputStep"),
    ],
    "image_conditioned_inpainting": [
        ("text_encoder.resize", "QwenImageEditResizeStep"),
        ("text_encoder.encode", "QwenImageEditTextEncoderStep"),
        ("vae_encoder.resize", "QwenImageEditResizeStep"),
        ("vae_encoder.preprocess", "QwenImageEditInpaintProcessImagesInputStep"),
        ("vae_encoder.encode", "QwenImageVaeEncoderStep"),
        ("denoise.input.text_inputs", "QwenImageTextInputsStep"),
        ("denoise.input.additional_inputs", "QwenImageAdditionalInputsStep"),
        ("denoise.prepare_latents", "QwenImagePrepareLatentsStep"),
        ("denoise.set_timesteps", "QwenImageSetTimestepsWithStrengthStep"),
        ("denoise.prepare_inpaint_latents.add_noise_to_latents", "QwenImagePrepareLatentsWithStrengthStep"),
        ("denoise.prepare_inpaint_latents.create_mask_latents", "QwenImageCreateMaskLatentsStep"),
        ("denoise.prepare_rope_inputs", "QwenImageEditRoPEInputsStep"),
        ("denoise.denoise", "QwenImageEditInpaintDenoiseStep"),
        ("denoise.after_denoise", "QwenImageAfterDenoiseStep"),
        ("decode.decode", "QwenImageDecoderStep"),
        ("decode.postprocess", "QwenImageInpaintProcessImagesOutputStep"),
    ],
}


class TestQwenImageEditModularPipelineFast(ModularPipelineTesterMixin, ModularGuiderTesterMixin):
    pipeline_class = QwenImageEditModularPipeline
    pipeline_blocks_class = QwenImageEditAutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-qwenimage-edit-modular"

    params = frozenset(["prompt", "height", "width", "negative_prompt", "attention_kwargs", "image", "mask_image"])
    batch_params = frozenset(["prompt", "negative_prompt", "image", "mask_image"])
    expected_workflow_blocks = QWEN_IMAGE_EDIT_WORKFLOWS

    def get_dummy_inputs(self):
        generator = self.get_generator()
        inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "output_type": "pt",
        }
        inputs["image"] = PIL.Image.new("RGB", (32, 32), 0)
        return inputs

    def test_guider_cfg(self):
        super().test_guider_cfg(7e-5)


class TestQwenImageEditPlusModularPipelineFast(ModularPipelineTesterMixin, ModularGuiderTesterMixin):
    pipeline_class = QwenImageEditPlusModularPipeline
    pipeline_blocks_class = QwenImageEditPlusAutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-qwenimage-edit-plus-modular"

    # No `mask_image` yet.
    params = frozenset(["prompt", "height", "width", "negative_prompt", "attention_kwargs", "image"])
    batch_params = frozenset(["prompt", "negative_prompt", "image"])

    def get_dummy_inputs(self):
        generator = self.get_generator()
        inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "output_type": "pt",
        }
        inputs["image"] = PIL.Image.new("RGB", (32, 32), 0)
        return inputs

    def test_multi_images_as_input(self):
        inputs = self.get_dummy_inputs()
        image = inputs.pop("image")
        inputs["image"] = [image, image]

        pipe = self.get_pipeline().to(torch_device)
        _ = pipe(
            **inputs,
        )

    @pytest.mark.xfail(condition=True, reason="Batch of multiple images needs to be revisited", strict=True)
    def test_num_images_per_prompt(self):
        super().test_num_images_per_prompt()

    @pytest.mark.xfail(condition=True, reason="Batch of multiple images needs to be revisited", strict=True)
    def test_inference_batch_consistent():
        super().test_inference_batch_consistent()

    @pytest.mark.xfail(condition=True, reason="Batch of multiple images needs to be revisited", strict=True)
    def test_inference_batch_single_identical():
        super().test_inference_batch_single_identical()

    def test_guider_cfg(self):
        super().test_guider_cfg(1e-6)
