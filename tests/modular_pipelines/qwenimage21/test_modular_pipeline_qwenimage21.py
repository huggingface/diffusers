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

import numpy as np
import PIL

from diffusers.modular_pipelines import QwenImage21AutoBlocks, QwenImage21ModularPipeline

from ...testing_utils import assert_tensors_close, torch_device
from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularGuiderTesterMixin,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


# The tiny VAE compresses 16x and the transformer groups latents in 2x2 blocks, so 32 is the smallest resolution:
# a 2x2 latent, which is exactly one vision slot's worth of target tokens.
IMAGE_SIZE = 32

QWENIMAGE21_TEXT2IMAGE_WORKFLOWS = {
    "text2image": [
        ("text_encoder", "QwenImage21TextEncoderStep"),
        ("denoise.input", "QwenImage21TextInputsStep"),
        ("denoise.prepare_latents", "QwenImage21PrepareLatentsStep"),
        ("denoise.set_timesteps", "QwenImage21SetTimestepsStep"),
        ("denoise.prepare_rope_inputs", "QwenImage21RoPEInputsStep"),
        ("denoise.denoise", "QwenImage21DenoiseStep"),
        ("denoise.unpack_latents", "QwenImage21UnpackLatentsStep"),
        ("decode", "QwenImage21DecodeStep"),
    ],
}

QWENIMAGE21_IMAGE_CONDITIONED_WORKFLOWS = {
    "image_conditioned": [
        ("text_encoder.resize", "QwenImage21ResizeStep"),
        ("text_encoder.encode", "QwenImage21VLTextEncoderStep"),
        ("vae_encoder.resize", "QwenImage21ResizeStep"),
        ("vae_encoder.preprocess", "QwenImage21ProcessImagesInputStep"),
        ("vae_encoder.encode", "QwenImage21VaeEncoderStep"),
        ("denoise.input", "QwenImage21TextInputsStep"),
        ("denoise.additional_inputs", "QwenImage21AdditionalInputsStep"),
        ("denoise.prepare_latents", "QwenImage21PrepareLatentsStep"),
        ("denoise.set_timesteps", "QwenImage21SetTimestepsStep"),
        ("denoise.prepare_rope_inputs", "QwenImage21ImageConditionedRoPEInputsStep"),
        ("denoise.denoise", "QwenImage21ImageConditionedDenoiseStep"),
        ("denoise.unpack_latents", "QwenImage21UnpackLatentsStep"),
        ("decode", "QwenImage21DecodeStep"),
    ],
}


QWENIMAGE21_WORKFLOW_DEFAULTS = {
    "text2image": {
        "components": {
            "text_encoder": "Qwen3VLForConditionalGeneration",
            "processor": "Qwen3VLProcessor",
            "guider": "ClassifierFreeGuidance",
            "scheduler": "FlowMatchEulerDiscreteScheduler",
            "transformer": "QwenImage21Transformer2DModel",
            "vae": "AutoencoderKLQwenImage21",
            "image_processor": "VaeImageProcessor",
        },
        "configs": {"sample_sigmas": None},
        "required_inputs": ["prompt"],
        "inputs": {
            "negative_prompt": None,
            "num_images_per_prompt": 1,
            "latents": None,
            "height": None,
            "width": None,
            "output_resolution": 1024,
            "generator": None,
            "num_inference_steps": 40,
            "sigmas": None,
            "use_kv_cache": True,
            "attention_kwargs": None,
            "output_type": "pil",
        },
        "component_configs": {"guider": {"guidance_scale": 1.0}, "image_processor": {"vae_scale_factor": 16}},
    },
    "image_conditioned": {
        "components": {
            "image_processor": "VaeImageProcessor",
            "text_encoder": "Qwen3VLForConditionalGeneration",
            "processor": "Qwen3VLProcessor",
            "guider": "ClassifierFreeGuidance",
            "vae": "AutoencoderKLQwenImage21",
            "scheduler": "FlowMatchEulerDiscreteScheduler",
            "transformer": "QwenImage21Transformer2DModel",
        },
        "configs": {"sample_sigmas": None},
        "required_inputs": ["image", "prompt"],
        "inputs": {
            "output_resolution": 1024,
            "negative_prompt": None,
            "generator": None,
            "num_images_per_prompt": 1,
            "height": None,
            "width": None,
            "latents": None,
            "num_inference_steps": 40,
            "sigmas": None,
            "use_kv_cache": True,
            "attention_kwargs": None,
            "output_type": "pil",
        },
        "component_configs": {"image_processor": {"vae_scale_factor": 16}, "guider": {"guidance_scale": 1.0}},
    },
}


def get_dummy_condition_image(seed=0):
    array = np.random.RandomState(seed).randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return PIL.Image.fromarray(array).convert("RGBA")


class QwenImage21ModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = QwenImage21ModularPipeline
    pipeline_blocks_class = QwenImage21AutoBlocks
    pretrained_model_name_or_path = "akshan-main/tiny-qwenimage21-modular-pipe"
    params = frozenset(
        ["prompt", "negative_prompt", "height", "width", "output_resolution", "attention_kwargs", "image"]
    )
    batch_params = frozenset(["prompt", "negative_prompt"])
    expected_workflow_blocks = QWENIMAGE21_TEXT2IMAGE_WORKFLOWS
    expected_workflow_defaults = {"text2image": QWENIMAGE21_WORKFLOW_DEFAULTS["text2image"]}

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "generator": self.get_generator(seed),
            "num_inference_steps": 2,
            "height": IMAGE_SIZE,
            "width": IMAGE_SIZE,
            "output_type": "pt",
        }


class TestQwenImage21ModularPipelineFast(QwenImage21ModularPipelineTesterConfig, ModularPipelineTesterMixin):
    def test_kv_cache_matches_uncached_denoising(self):
        # The cache only stores the step-independent text keys and values, so with and without it the loop has
        # to land on the same image.
        pipe = self.get_pipeline().to(torch_device)
        cached = pipe(**self.get_dummy_inputs(), use_kv_cache=True, output="images")
        uncached = pipe(**self.get_dummy_inputs(), use_kv_cache=False, output="images")
        assert_tensors_close(cached, uncached, atol=1e-4, rtol=1e-4)

    def test_sample_sigmas_config_sets_the_schedule(self):
        pipe = self.get_pipeline().to(torch_device)
        sample_sigmas = [1.0, 0.6, 0.2]
        pipe.update_components(sample_sigmas=sample_sigmas)
        timesteps = pipe(**self.get_dummy_inputs(), output="timesteps")
        assert len(timesteps) == len(sample_sigmas)
        # Explicit sigmas take precedence over the configured grid.
        timesteps = pipe(**self.get_dummy_inputs(), sigmas=[1.0, 0.5], output="timesteps")
        assert len(timesteps) == 2


class TestQwenImage21ModularPipelineLoading(QwenImage21ModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestQwenImage21ModularPipelineWorkflow(QwenImage21ModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestQwenImage21ModularPipelineMemory(QwenImage21ModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


class TestQwenImage21ModularPipelineGuider(QwenImage21ModularPipelineTesterConfig, ModularGuiderTesterMixin):
    def test_guider_cfg(self):
        # The tiny model moves the output by about 1e-2 under guidance, right at the default threshold.
        super().test_guider_cfg(1e-3)


class QwenImage21ImageConditionedModularPipelineTesterConfig(QwenImage21ModularPipelineTesterConfig):
    expected_workflow_blocks = QWENIMAGE21_IMAGE_CONDITIONED_WORKFLOWS
    expected_workflow_defaults = {"image_conditioned": QWENIMAGE21_WORKFLOW_DEFAULTS["image_conditioned"]}

    def get_dummy_inputs(self, seed=0):
        inputs = super().get_dummy_inputs(seed)
        # Keep the condition image at its own size: the resize step would otherwise blow a tiny image up to the
        # default 1024x1024 area before the text encoder and the VAE.
        inputs["output_resolution"] = IMAGE_SIZE
        inputs["image"] = get_dummy_condition_image()
        return inputs


class TestQwenImage21ImageConditionedModularPipelineFast(
    QwenImage21ImageConditionedModularPipelineTesterConfig, ModularPipelineTesterMixin
):
    def test_multiple_condition_images(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["image"] = [get_dummy_condition_image(0), get_dummy_condition_image(1)]
        images = pipe(**inputs, output="images")
        assert images.shape == (1, 4, IMAGE_SIZE, IMAGE_SIZE)

    def test_output_size_follows_the_last_condition_image(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs.pop("height")
        inputs.pop("width")
        inputs["output_resolution"] = 2 * IMAGE_SIZE
        inputs["image"] = [
            get_dummy_condition_image(),
            PIL.Image.new("RGB", (4 * IMAGE_SIZE, IMAGE_SIZE)),
        ]
        state = pipe(**inputs)
        assert state.get("height") == IMAGE_SIZE
        assert state.get("width") == 4 * IMAGE_SIZE
        assert state.get("images").shape == (1, 4, IMAGE_SIZE, 4 * IMAGE_SIZE)

    def test_kv_cache_matches_uncached_denoising(self):
        pipe = self.get_pipeline().to(torch_device)
        cached = pipe(**self.get_dummy_inputs(), use_kv_cache=True, output="images")
        uncached = pipe(**self.get_dummy_inputs(), use_kv_cache=False, output="images")
        assert_tensors_close(cached, uncached, atol=1e-4, rtol=1e-4)


class TestQwenImage21ImageConditionedModularPipelineLoading(
    QwenImage21ImageConditionedModularPipelineTesterConfig, ModularLoadingTesterMixin
):
    pass


class TestQwenImage21ImageConditionedModularPipelineWorkflow(
    QwenImage21ImageConditionedModularPipelineTesterConfig, ModularWorkflowTesterMixin
):
    pass


class TestQwenImage21ImageConditionedModularPipelineMemory(
    QwenImage21ImageConditionedModularPipelineTesterConfig, ModularMemoryTesterMixin
):
    pass


class TestQwenImage21ImageConditionedModularPipelineGuider(
    QwenImage21ImageConditionedModularPipelineTesterConfig, ModularGuiderTesterMixin
):
    def test_guider_cfg(self):
        super().test_guider_cfg(1e-3)
