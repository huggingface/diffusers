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


import torch

from diffusers.modular_pipelines import ZImageAutoBlocks, ZImageModularPipeline
from diffusers.utils import load_image

from ...testing_utils import require_accelerator, slow, torch_device
from ..test_components_manager import ModularPipelineIntegrationTesterMixin
from ..test_modular_pipelines_common import ModularPipelineTesterMixin


ZIMAGE_WORKFLOWS = {
    "text2image": [
        ("text_encoder", "ZImageTextEncoderStep"),
        ("denoise.input", "ZImageTextInputStep"),
        ("denoise.prepare_latents", "ZImagePrepareLatentsStep"),
        ("denoise.set_timesteps", "ZImageSetTimestepsStep"),
        ("denoise.denoise", "ZImageDenoiseStep"),
        ("decode", "ZImageVaeDecoderStep"),
    ],
    "image2image": [
        ("text_encoder", "ZImageTextEncoderStep"),
        ("vae_encoder", "ZImageVaeImageEncoderStep"),
        ("denoise.input", "ZImageTextInputStep"),
        ("denoise.additional_inputs", "ZImageAdditionalInputsStep"),
        ("denoise.prepare_latents", "ZImagePrepareLatentsStep"),
        ("denoise.set_timesteps", "ZImageSetTimestepsStep"),
        ("denoise.set_timesteps_with_strength", "ZImageSetTimestepsWithStrengthStep"),
        ("denoise.prepare_latents_with_image", "ZImagePrepareLatentswithImageStep"),
        ("denoise.denoise", "ZImageDenoiseStep"),
        ("decode", "ZImageVaeDecoderStep"),
    ],
}


class TestZImageModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = ZImageModularPipeline
    pipeline_blocks_class = ZImageAutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-zimage-modular-pipe"

    params = frozenset(["prompt", "height", "width"])
    batch_params = frozenset(["prompt"])
    expected_workflow_blocks = ZIMAGE_WORKFLOWS

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


class TestZImageModularIntegration(ModularPipelineIntegrationTesterMixin):
    """Real-checkpoint runs (slow), ~19GB of bf16 weights: text_encoder 7.5GB, transformer
    11.5GB, vae 0.16GB. On 32GB everything fits and nothing offloads; on 24GB only the text
    encoder yields (its forward's activations eat into the card before the transformer loads);
    on 16GB the transformer must also yield to the vae at decode; and 10GB is smaller than the
    transformer itself - it runs alone through the escape hatch."""

    repo_id = "Tongyi-MAI/Z-Image-Turbo"

    @property
    def offload_cards(self):
        return {
            "32GB": {
                "offload": {},
                "oom": {},
                "final_device": {"text_encoder": "cuda", "transformer": "cuda", "vae": "cuda"},
            },
            "24GB": {
                "offload": {"text_encoder": 1},
                "oom": {},
                "final_device": {"text_encoder": "cpu", "transformer": "cuda", "vae": "cuda"},
            },
            "16GB": {
                "offload": {"text_encoder": 1, "transformer": 1},
                "oom": {},
                "final_device": {"text_encoder": "cpu", "transformer": "cpu", "vae": "cuda"},
            },
            "10GB": {
                "offload": {"text_encoder": 1, "transformer": 1},
                "oom": {},
                "final_device": {"text_encoder": "cpu", "transformer": "cpu", "vae": "cuda"},
            },
        }

    def get_inputs(self):
        return {
            "prompt": "a photo of a cat sitting on a windowsill",
            "generator": torch.Generator("cpu").manual_seed(0),
            "num_inference_steps": 4,
            "height": 1024,
            "width": 1024,
            "output_type": "pt",
        }

    @slow
    @require_accelerator
    def test_text_to_image(self):
        pipe = self.get_pipeline()
        pipe.to(torch_device)
        image = pipe(**self.get_inputs(), output="images")

        assert image.shape == (1, 3, 1024, 1024)
        image_slice = image[0, -1, -3:, -3:].flatten().float().cpu()
        expected_slice = torch.tensor([0.1895, 0.1895, 0.1973, 0.1953, 0.1934, 0.2031, 0.1875, 0.1875, 0.2109])
        max_diff = torch.abs(image_slice - expected_slice).max()
        assert max_diff < 1e-2, f"output slice {image_slice.tolist()} != expected (max diff {max_diff})"

    @slow
    @require_accelerator
    def test_image_to_image(self):
        pipe = self.get_pipeline()
        pipe.to(torch_device)
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint/overture-creations-5sI6fQgYIuo.png"
        )
        image = pipe(**self.get_inputs(), image=init_image, strength=0.6, output="images")

        assert image.shape == (1, 3, 1024, 1024)
        image_slice = image[0, -1, -3:, -3:].flatten().float().cpu()
        expected_slice = torch.tensor([0.4160, 0.4707, 0.3203, 0.4238, 0.4961, 0.4277, 0.4473, 0.3555, 0.3848])
        max_diff = torch.abs(image_slice - expected_slice).max()
        assert max_diff < 1e-2, f"output slice {image_slice.tolist()} != expected (max diff {max_diff})"
