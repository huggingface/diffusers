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

from diffusers.modular_pipelines import QwenImage21TurboAutoBlocks, QwenImage21TurboModularPipeline

from ...testing_utils import assert_tensors_close, torch_device
from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


IMAGE_SIZE = 32

QWENIMAGE21_TURBO_TEXT2IMAGE_WORKFLOWS = {
    "text2image": [
        ("text_encoder", "QwenImage21TurboTextEncoderStep"),
        ("denoise.input", "QwenImage21TurboTextInputsStep"),
        ("denoise.prepare_latents", "QwenImage21PrepareLatentsStep"),
        ("denoise.set_timesteps", "QwenImage21TurboSetTimestepsStep"),
        ("denoise.prepare_rope_inputs", "QwenImage21RoPEInputsStep"),
        ("denoise.denoise", "QwenImage21TurboDenoiseStep"),
        ("denoise.unpack_latents", "QwenImage21UnpackLatentsStep"),
        ("decode", "QwenImage21DecodeStep"),
    ],
}

QWENIMAGE21_TURBO_IMAGE_CONDITIONED_WORKFLOWS = {
    "image_conditioned": [
        ("text_encoder.resize", "QwenImage21ResizeStep"),
        ("text_encoder.encode", "QwenImage21TurboVLTextEncoderStep"),
        ("vae_encoder.resize", "QwenImage21ResizeStep"),
        ("vae_encoder.preprocess", "QwenImage21ProcessImagesInputStep"),
        ("vae_encoder.encode", "QwenImage21VaeEncoderStep"),
        ("denoise.input", "QwenImage21TurboTextInputsStep"),
        ("denoise.additional_inputs", "QwenImage21AdditionalInputsStep"),
        ("denoise.prepare_latents", "QwenImage21PrepareLatentsStep"),
        ("denoise.set_timesteps", "QwenImage21TurboSetTimestepsStep"),
        ("denoise.prepare_rope_inputs", "QwenImage21ImageConditionedRoPEInputsStep"),
        ("denoise.denoise", "QwenImage21TurboImageConditionedDenoiseStep"),
        ("denoise.unpack_latents", "QwenImage21UnpackLatentsStep"),
        ("decode", "QwenImage21DecodeStep"),
    ],
}


QWENIMAGE21_TURBO_WORKFLOW_DEFAULTS = {
    "text2image": {
        "components": {
            "text_encoder": "Qwen3VLForConditionalGeneration",
            "processor": "Qwen3VLProcessor",
            "scheduler": "FlowMatchEulerDiscreteScheduler",
            "transformer": "QwenImage21Transformer2DModel",
            "vae": "AutoencoderKLQwenImage21",
            "image_processor": "VaeImageProcessor",
        },
        "configs": {"sample_sigmas": None},
        "required_inputs": ["prompt"],
        "inputs": {
            "num_images_per_prompt": 1,
            "latents": None,
            "height": None,
            "width": None,
            "output_resolution": 1024,
            "generator": None,
            "num_inference_steps": 8,
            "sigmas": None,
            "use_kv_cache": True,
            "attention_kwargs": None,
            "output_type": "pil",
        },
        "component_configs": {"image_processor": {"vae_scale_factor": 16}},
    },
    "image_conditioned": {
        "components": {
            "image_processor": "VaeImageProcessor",
            "text_encoder": "Qwen3VLForConditionalGeneration",
            "processor": "Qwen3VLProcessor",
            "vae": "AutoencoderKLQwenImage21",
            "scheduler": "FlowMatchEulerDiscreteScheduler",
            "transformer": "QwenImage21Transformer2DModel",
        },
        "configs": {"sample_sigmas": None},
        "required_inputs": ["image", "prompt"],
        "inputs": {
            "output_resolution": 1024,
            "generator": None,
            "num_images_per_prompt": 1,
            "height": None,
            "width": None,
            "latents": None,
            "num_inference_steps": 8,
            "sigmas": None,
            "use_kv_cache": True,
            "attention_kwargs": None,
            "output_type": "pil",
        },
        "component_configs": {"image_processor": {"vae_scale_factor": 16}},
    },
}


def get_dummy_condition_image(seed=0):
    array = np.random.RandomState(seed).randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return PIL.Image.fromarray(array).convert("RGBA")


class QwenImage21TurboModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = QwenImage21TurboModularPipeline
    pipeline_blocks_class = QwenImage21TurboAutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-qwenimage21-turbo-modular"
    expected_latents_shape = (1, 8, 1, 2, 2)
    params = frozenset(["prompt", "height", "width", "output_resolution", "attention_kwargs", "image"])
    batch_params = frozenset(["prompt"])
    not_params = frozenset(["negative_prompt"])
    expected_workflow_blocks = QWENIMAGE21_TURBO_TEXT2IMAGE_WORKFLOWS
    expected_workflow_defaults = {"text2image": QWENIMAGE21_TURBO_WORKFLOW_DEFAULTS["text2image"]}

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "dance monkey",
            "generator": self.get_generator(seed),
            "num_inference_steps": 2,
            "height": IMAGE_SIZE,
            "width": IMAGE_SIZE,
            "output_type": "pt",
        }


class TestQwenImage21TurboModularPipelineFast(QwenImage21TurboModularPipelineTesterConfig, ModularPipelineTesterMixin):
    def test_checkpoint_grid_sets_the_schedule(self):
        # The tiny repo ships a two-sigma grid, so the schedule follows it rather than `num_inference_steps`, and an
        # explicit `sigmas` list still wins over the configured grid.
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        timesteps = pipe(**inputs, output="timesteps")
        assert len(timesteps) == len(pipe.config.sample_sigmas)
        timesteps = pipe(**self.get_dummy_inputs(), sigmas=[1.0, 0.6, 0.2], output="timesteps")
        assert len(timesteps) == 3

    def test_kv_cache_matches_uncached_denoising(self):
        pipe = self.get_pipeline().to(torch_device)
        cached = pipe(**self.get_dummy_inputs(), use_kv_cache=True, output="images")
        uncached = pipe(**self.get_dummy_inputs(), use_kv_cache=False, output="images")
        assert_tensors_close(cached, uncached, atol=1e-4, rtol=1e-4)


class TestQwenImage21TurboModularPipelineLoading(
    QwenImage21TurboModularPipelineTesterConfig, ModularLoadingTesterMixin
):
    pass


class TestQwenImage21TurboModularPipelineWorkflow(
    QwenImage21TurboModularPipelineTesterConfig, ModularWorkflowTesterMixin
):
    pass


class TestQwenImage21TurboModularPipelineMemory(QwenImage21TurboModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


class QwenImage21TurboImageConditionedModularPipelineTesterConfig(QwenImage21TurboModularPipelineTesterConfig):
    expected_workflow_blocks = QWENIMAGE21_TURBO_IMAGE_CONDITIONED_WORKFLOWS
    expected_workflow_defaults = {"image_conditioned": QWENIMAGE21_TURBO_WORKFLOW_DEFAULTS["image_conditioned"]}

    def get_dummy_inputs(self, seed=0):
        inputs = super().get_dummy_inputs(seed)
        inputs["output_resolution"] = IMAGE_SIZE
        inputs["image"] = get_dummy_condition_image()
        return inputs


class TestQwenImage21TurboImageConditionedModularPipelineFast(
    QwenImage21TurboImageConditionedModularPipelineTesterConfig, ModularPipelineTesterMixin
):
    def test_multiple_condition_images(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["image"] = [get_dummy_condition_image(0), get_dummy_condition_image(1)]
        images = pipe(**inputs, output="images")
        assert images.shape == (1, 4, IMAGE_SIZE, IMAGE_SIZE)


class TestQwenImage21TurboImageConditionedModularPipelineLoading(
    QwenImage21TurboImageConditionedModularPipelineTesterConfig, ModularLoadingTesterMixin
):
    pass


class TestQwenImage21TurboImageConditionedModularPipelineWorkflow(
    QwenImage21TurboImageConditionedModularPipelineTesterConfig, ModularWorkflowTesterMixin
):
    pass


class TestQwenImage21TurboImageConditionedModularPipelineMemory(
    QwenImage21TurboImageConditionedModularPipelineTesterConfig, ModularMemoryTesterMixin
):
    pass
