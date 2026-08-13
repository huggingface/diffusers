# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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
import torch

from diffusers.modular_pipelines import Krea2TurboAutoBlocks, Krea2TurboModularPipeline

from ..test_modular_pipelines_common import ModularPipelineTesterMixin


KREA2_TURBO_WORKFLOWS = {
    "text2image": [
        ("text_encoder", "Krea2TurboTextEncoderStep"),
        ("denoise.input", "Krea2TurboTextInputsStep"),
        ("denoise.prepare_latents", "Krea2PrepareLatentsStep"),
        ("denoise.set_timesteps", "Krea2TurboSetTimestepsStep"),
        ("denoise.prepare_position_ids", "Krea2PreparePositionIdsStep"),
        ("denoise.denoise", "Krea2TurboDenoiseStep"),
        ("decode", "Krea2DecodeStep"),
    ],
    "image2image": [
        ("text_encoder", "Krea2TurboTextEncoderStep"),
        ("vae_encoder.preprocess", "Krea2ProcessImagesInputStep"),
        ("vae_encoder.encode", "Krea2VaeEncoderStep"),
        ("denoise.input.text_inputs", "Krea2TurboTextInputsStep"),
        ("denoise.input.image_inputs", "Krea2ImageInputsStep"),
        ("denoise.prepare_latents", "Krea2PrepareLatentsStep"),
        ("denoise.set_timesteps", "Krea2TurboSetTimestepsStep"),
        ("denoise.apply_strength", "Krea2ApplyStrengthStep"),
        ("denoise.prepare_image_latents", "Krea2PrepareImageLatentsStep"),
        ("denoise.prepare_position_ids", "Krea2PreparePositionIdsStep"),
        ("denoise.denoise", "Krea2TurboDenoiseStep"),
        ("decode", "Krea2DecodeStep"),
    ],
    "inpainting": [
        ("text_encoder", "Krea2TurboTextEncoderStep"),
        ("vae_encoder.preprocess", "Krea2InpaintProcessImagesInputStep"),
        ("vae_encoder.encode", "Krea2VaeEncoderStep"),
        ("denoise.input.text_inputs", "Krea2TurboTextInputsStep"),
        ("denoise.input.image_inputs", "Krea2ImageInputsStep"),
        ("denoise.prepare_latents", "Krea2PrepareLatentsStep"),
        ("denoise.set_timesteps", "Krea2TurboSetTimestepsStep"),
        ("denoise.apply_strength", "Krea2ApplyStrengthStep"),
        ("denoise.prepare_inpaint_latents.add_noise", "Krea2PrepareImageLatentsStep"),
        ("denoise.prepare_inpaint_latents.prepare_mask", "Krea2PrepareMaskLatentsStep"),
        ("denoise.prepare_position_ids", "Krea2PreparePositionIdsStep"),
        ("denoise.denoise", "Krea2TurboInpaintDenoiseStep"),
        ("decode", "Krea2InpaintDecodeStep"),
    ],
    "reference": [
        ("text_encoder", "Krea2TurboReferenceTextEncoderStep"),
        ("vae_encoder.preprocess", "Krea2ReferenceProcessImagesInputStep"),
        ("vae_encoder.encode", "Krea2ReferenceVaeEncoderStep"),
        ("denoise.input.text_inputs", "Krea2TurboTextInputsStep"),
        ("denoise.input.reference_inputs", "Krea2ReferenceInputsStep"),
        ("denoise.prepare_latents", "Krea2PrepareLatentsStep"),
        ("denoise.set_timesteps", "Krea2TurboSetTimestepsStep"),
        ("denoise.prepare_position_ids", "Krea2PrepareReferencePositionIdsStep"),
        ("denoise.denoise", "Krea2TurboReferenceDenoiseStep"),
        ("decode", "Krea2DecodeStep"),
    ],
}


class TestKrea2TurboModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = Krea2TurboModularPipeline
    pipeline_blocks_class = Krea2TurboAutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-krea2-turbo-modular-pipe"

    params = frozenset(["prompt", "height", "width", "image", "mask_image", "reference_image"])
    batch_params = frozenset(["prompt", "image", "mask_image"])
    expected_workflow_blocks = KREA2_TURBO_WORKFLOWS

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=5e-3)

    def test_image2image(self):
        pipe = self.get_pipeline().to("cpu")
        inputs = self.get_dummy_inputs()
        inputs["image"] = PIL.Image.new("RGB", (32, 32), "white")
        output = pipe(**inputs, strength=0.8, output="images")
        assert output.shape == (1, 3, 32, 32)

    def test_inpainting(self):
        pipe = self.get_pipeline().to("cpu")
        inputs = self.get_dummy_inputs()
        inputs["image"] = PIL.Image.new("RGB", (32, 32), "white")
        inputs["mask_image"] = PIL.Image.new("L", (32, 32), "black")
        output_low_strength = pipe(**inputs, strength=0.5, output="images")
        inputs["generator"] = self.get_generator(0)
        output_full_strength = pipe(**inputs, strength=1.0, output="images")
        assert output_low_strength.shape == (1, 3, 32, 32)
        assert (output_low_strength - output_full_strength).abs().max() < 1e-6

    def test_reference_image(self):
        pipe = self.get_pipeline().to("cpu")
        inputs = self.get_dummy_inputs()
        inputs["reference_image"] = PIL.Image.new("RGB", (32, 32), "white")
        inputs["reference_image_encoder_resolution"] = 32
        single_reference_output = pipe(**inputs, reference_attention_scale=2.0, output="images")

        inputs["generator"] = self.get_generator(0)
        inputs["reference_image"] = [PIL.Image.new("RGB", (32, 32), "white")]
        single_reference_list_output = pipe(**inputs, reference_attention_scale=[2.0], output="images")

        inputs["generator"] = self.get_generator(0)
        inputs["reference_image"] = [
            PIL.Image.new("RGB", (32, 32), "white"),
            PIL.Image.new("RGB", (32, 32), "black"),
            PIL.Image.new("RGB", (32, 32), "gray"),
        ]
        multi_reference_output = pipe(**inputs, reference_attention_scale=[1.0, 2.0, 0.5], output="images")

        inputs["prompt"] = [inputs["prompt"], inputs["prompt"]]
        inputs["generator"] = [self.get_generator(0), self.get_generator(1)]
        batched_output = pipe(**inputs, reference_attention_scale=[1.0, 2.0, 0.5], output="images")
        assert single_reference_output.shape == (1, 3, 32, 32)
        assert torch.allclose(single_reference_output, single_reference_list_output)
        assert not torch.allclose(single_reference_output, multi_reference_output)
        assert batched_output.shape == (2, 3, 32, 32)
        assert (batched_output[:1] - multi_reference_output).abs().max() < 5e-3
