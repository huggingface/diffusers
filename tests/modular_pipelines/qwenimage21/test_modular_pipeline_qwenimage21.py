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

import numpy as np
import pytest
import torch
from PIL import Image

from diffusers import (
    ClassifierFreeGuidance,
    FlowMatchEulerDiscreteScheduler,
    ModularPipeline,
    QwenImage21AutoBlocks,
    QwenImage21ModularPipeline,
    QwenImage21Pipeline,
)
from diffusers.modular_pipelines.qwenimage21.modular_blocks_qwenimage21 import QwenImage21CoreDenoiseStep

from ...pipelines.qwenimage21.test_qwenimage21 import QwenImage21PipelineTesterConfig
from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularGuiderTesterMixin,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("qwenimage21")
    standard = root / "standard"
    components = QwenImage21PipelineTesterConfig().get_dummy_components()
    components["scheduler"] = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True, max_image_seq_len=8192, base_shift=0.5, max_shift=0.9, shift_terminal=0.02
    )
    QwenImage21Pipeline(**components).save_pretrained(standard)
    pipe = ModularPipeline.from_pretrained(str(standard))
    pipe.load_components(dtype=torch.float32)
    pipe.save_pretrained(str(root / "modular"))
    return root


def condition_image(seed=0, size=(32, 32)):
    return Image.fromarray(np.random.RandomState(seed).randint(0, 256, (*size, 4), dtype=np.uint8))


class QwenImage21ModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = QwenImage21ModularPipeline
    pipeline_blocks_class = QwenImage21AutoBlocks
    pretrained_model_name_or_path = None
    params = frozenset(["prompt", "negative_prompt", "height", "width", "image", "mask_image", "reference_images"])
    batch_params = frozenset(["prompt", "negative_prompt"])
    expected_workflow_blocks = {
        "text2image": [
            ("text_encoder", "QwenImage21TextEncoderStep"),
            ("denoise.input", "QwenImage21TextInputsStep"),
            ("denoise.prepare_latents", "QwenImage21PrepareLatentsStep"),
            ("denoise.set_timesteps", "QwenImage21SetTimestepsStep"),
            ("denoise.denoise", "QwenImage21DenoiseStep"),
            ("decode", "QwenImage21DecodeStep"),
        ],
        "image_conditioned": [
            ("preprocess", "QwenImage21ProcessImagesStep"),
            ("text_encoder", "QwenImage21TextEncoderStep"),
            ("vae_encoder", "QwenImage21VaeEncoderStep"),
            ("denoise.input", "QwenImage21TextInputsStep"),
            ("denoise.prepare_latents", "QwenImage21PrepareLatentsStep"),
            ("denoise.set_timesteps", "QwenImage21SetTimestepsStep"),
            ("denoise.denoise", "QwenImage21DenoiseStep"),
            ("decode", "QwenImage21DecodeStep"),
        ],
        "inpainting": [
            ("preprocess", "QwenImage21ProcessInpaintStep"),
            ("text_encoder", "QwenImage21TextEncoderStep"),
            ("vae_encoder", "QwenImage21VaeEncoderStep"),
            ("source_encoder", "QwenImage21InpaintVaeEncoderStep"),
            ("denoise.input", "QwenImage21TextInputsStep"),
            ("denoise.prepare_latents", "QwenImage21PrepareLatentsStep"),
            ("denoise.set_timesteps", "QwenImage21SetTimestepsStep"),
            ("denoise.prepare_inpaint", "QwenImage21PrepareInpaintStep"),
            ("denoise.denoise", "QwenImage21InpaintDenoiseStep"),
            ("decode", "QwenImage21DecodeStep"),
        ],
    }

    @pytest.fixture(scope="class", autouse=True)
    def checkpoint(self, request, tiny_checkpoint):
        request.cls.pretrained_model_name_or_path = str(tiny_checkpoint / "modular")

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a cat",
            "height": 32,
            "width": 32,
            "output_resolution": 32,
            "num_inference_steps": 2,
            "generator": self.get_generator(seed),
            "output_type": "pt",
        }


class TestQwenImage21ModularPipeline(QwenImage21ModularPipelineTesterConfig, ModularPipelineTesterMixin):
    def test_inference_batch_single_identical(self):
        # The tiny RGBA VAE amplifies the observed 1e-7 CUDA latent difference to 1.6e-4 in decoded pixels.
        super().test_inference_batch_single_identical(expected_max_diff=3e-4)

    @pytest.mark.parametrize("references", [0, 1, 2])
    @pytest.mark.parametrize("use_kv_cache", [False, True])
    @pytest.mark.parametrize("guidance", [1.0, 3.0])
    def test_standard_parity(self, tiny_checkpoint, references, use_kv_cache, guidance):
        reference = QwenImage21Pipeline.from_pretrained(tiny_checkpoint / "standard", dtype=torch.float32)
        pipe = self.get_pipeline()
        pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=guidance))
        inputs = self.get_dummy_inputs()
        inputs.update(use_kv_cache=use_kv_cache, negative_prompt="blurry")
        if references:
            inputs["image"] = [condition_image(i) for i in range(references)]
        expected = reference(**inputs, true_cfg_scale=guidance).images
        inputs["generator"] = self.get_generator()
        actual = pipe(**inputs, output="images")
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("references", [0, 2])
    @pytest.mark.parametrize("strength", [0.5, 1.0])
    def test_inpaint_black_mask_preserves_source_latents(self, references, strength):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs.update(
            image=condition_image(),
            mask_image=Image.new("L", (32, 32), 0),
            reference_images=[condition_image(i + 1) for i in range(references)],
            strength=strength,
        )
        state = pipe(**inputs)
        torch.testing.assert_close(state.get("latents"), state.get("source_latents"), atol=0, rtol=0)
        assert torch.isfinite(state.get("images")).all()

    def test_inpaint_white_mask_matches_conditioned_generation(self):
        pipe = self.get_pipeline()
        source, reference = condition_image(), condition_image(1)
        expected = pipe(**self.get_dummy_inputs(), image=[source, reference], output="images")
        actual = pipe(
            **self.get_dummy_inputs(),
            image=source,
            reference_images=[reference],
            mask_image=Image.new("L", (32, 32), 255),
            output="images",
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_multireference_inpaint_batch_and_partial_mask(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs.update(
            prompt=["a cat", "a dog"],
            height=64,
            width=64,
            output_resolution=64,
            num_images_per_prompt=2,
            image=condition_image(),
            reference_images=[condition_image(1, (32, 64)), condition_image(2)],
            mask_image=Image.fromarray(np.pad(np.full((32, 32), 255, dtype=np.uint8), 16)),
        )
        state = pipe(**inputs)
        assert state.get("images").shape == (4, 4, 64, 64)
        latents, source, mask = state.get("latents"), state.get("source_latents"), state.get("mask")
        torch.testing.assert_close(latents * (1 - mask), source * (1 - mask), atol=0, rtol=0)
        assert (latents * mask - source * mask).abs().max() > 0
        assert state.get("condition_latents").shape[1] > 4

    def test_reference_images_affect_masked_output(self):
        pipe = self.get_pipeline()
        inputs = {"image": condition_image(), "mask_image": Image.new("L", (32, 32), 255), "output": "images"}
        first = pipe(**self.get_dummy_inputs(), **inputs)
        second = pipe(**self.get_dummy_inputs(), reference_images=[condition_image(1)], **inputs)
        assert (first - second).abs().max() > 1e-6

    def test_repeated_calls_reset_kv_cache(self):
        pipe = self.get_pipeline()
        first = pipe(**self.get_dummy_inputs(), output="images")
        pipe(**self.get_dummy_inputs(1), image=[condition_image(), condition_image(1)], output="images")
        second = pipe(**self.get_dummy_inputs(), output="images")
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_guidance_start_with_kv_cache(self):
        pipe = self.get_pipeline()
        pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=3.0, start=0.5))
        inputs = self.get_dummy_inputs()
        inputs.update(num_inference_steps=4, negative_prompt="blurry", image=[condition_image(), condition_image(1)])
        expected = pipe(**inputs, use_kv_cache=False, output="images")
        inputs["generator"] = self.get_generator()
        actual = pipe(**inputs, use_kv_cache=True, output="images")
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_inpaint_dimensions_follow_source(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs.pop("height")
        inputs.pop("width")
        inputs.update(
            output_resolution=64,
            image=condition_image(size=(32, 64)),
            mask_image=Image.new("L", (64, 32)),
            reference_images=[condition_image(size=(64, 32))],
        )
        state = pipe(**inputs)
        assert state.get("height") == 32
        assert state.get("width") == 96
        assert state.get("images").shape == (1, 4, 32, 96)
        assert state.get("condition_shapes") == [(1, 2, 6), (1, 6, 2)]

    def test_reusable_text_encoder(self):
        pipe = self.get_pipeline()
        blocks = pipe.blocks.get_workflow("text2image")
        encoder = blocks.sub_blocks["text_encoder"].init_pipeline()
        encoder.update_components(text_encoder=pipe.text_encoder, processor=pipe.processor, guider=pipe.guider)
        encoded = encoder(prompt="a cat")
        core = QwenImage21CoreDenoiseStep().init_pipeline()
        core.update_components(transformer=pipe.transformer, scheduler=pipe.scheduler, guider=pipe.guider)
        inputs = self.get_dummy_inputs()
        inputs.pop("prompt")
        inputs.pop("output_type")
        state = core(
            **inputs,
            prompt_embeds=encoded.get("prompt_embeds"),
            prompt_embeds_mask=encoded.get("prompt_embeds_mask"),
            image_pad_mask=encoded.get("image_pad_mask"),
            num_images_per_prompt=2,
        )
        assert state.get("latents").shape == (2, 4, 8)
        assert encoded.get("prompt_embeds").shape[0] == 1

    @pytest.mark.parametrize("strength", [0, -0.1, 1.1, 0.1])
    def test_invalid_strength(self, strength):
        pipe = self.get_pipeline()
        with pytest.raises(ValueError, match="strength"):
            pipe(
                **self.get_dummy_inputs(),
                image=condition_image(),
                mask_image=Image.new("L", (32, 32)),
                strength=strength,
            )

    def test_inpaint_rejects_multiple_sources(self):
        pipe = self.get_pipeline()
        with pytest.raises(ValueError, match="one source"):
            pipe(
                **self.get_dummy_inputs(),
                image=[condition_image(), condition_image(1)],
                mask_image=Image.new("L", (32, 32)),
            )

    def test_rejects_references_without_mask(self):
        pipe = self.get_pipeline()
        with pytest.raises(ValueError, match="reference_images"):
            pipe(**self.get_dummy_inputs(), image=condition_image(), reference_images=[condition_image(1)])

    def test_rejects_mask_without_source(self):
        pipe = self.get_pipeline()
        with pytest.raises(ValueError, match="requires a source"):
            pipe(**self.get_dummy_inputs(), mask_image=Image.new("L", (32, 32)))

    def test_load_from_standard_index(self, tiny_checkpoint):
        pipe = ModularPipeline.from_pretrained(str(tiny_checkpoint / "standard"))
        assert isinstance(pipe, QwenImage21ModularPipeline)
        pipe.load_components(dtype=torch.float32)
        assert pipe(**self.get_dummy_inputs(), output="images").shape == (1, 4, 32, 32)


class TestQwenImage21ModularLoading(QwenImage21ModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestQwenImage21ModularWorkflows(QwenImage21ModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestQwenImage21ModularGuiders(QwenImage21ModularPipelineTesterConfig, ModularGuiderTesterMixin):
    pass


class TestQwenImage21ModularMemory(QwenImage21ModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass
