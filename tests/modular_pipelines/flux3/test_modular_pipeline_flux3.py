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
import PIL.Image
import pytest
import torch
from transformers import Qwen3_5ForConditionalGeneration

from diffusers.modular_pipelines import (
    Flux3AutoBlocks,
    Flux3DistilledAutoBlocks,
    Flux3DistilledModularPipeline,
    Flux3ModularPipeline,
)
from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
from diffusers.utils import logging

from ...testing_utils import CaptureLogger, torch_device
from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularGuiderTesterMixin,
    ModularLoadingTesterMixin,
    ModularLoraTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


FLUX3_TINY_REPO = "hf-internal-testing/tiny-flux3-image-modular"
FLUX3_DISTILLED_TINY_REPO = "hf-internal-testing/tiny-flux3-image-distilled-modular"

FLUX3_WORKFLOWS = {
    "text2image": [
        ("prompt_upsampler", "Flux3Text2ImagePromptUpsampleStep"),
        ("text_encoder", "Flux3TextEncoderStep"),
        ("denoise.input", "Flux3TextInputStep"),
        ("denoise.prepare_latents", "Flux3PrepareLatentsStep"),
        ("denoise.set_timesteps", "Flux3SetTimestepsStep"),
        ("denoise.prepare_rope_inputs", "Flux3TextRoPEInputsStep"),
        ("denoise.denoise", "Flux3DenoiseStep"),
        ("denoise.after_denoise", "Flux3UnpackLatentsStep"),
        ("decode", "Flux3DecodeStep"),
    ],
    "image_conditioned": [
        ("prompt_upsampler", "Flux3ImageConditionedPromptUpsampleStep"),
        ("text_encoder", "Flux3TextEncoderStep"),
        ("vae_encoder.preprocess", "Flux3ProcessImagesInputStep"),
        ("vae_encoder.encode", "Flux3VaeEncoderStep"),
        ("denoise.input", "Flux3TextInputStep"),
        ("denoise.cond_input", "Flux3ImageCondInputStep"),
        ("denoise.prepare_latents", "Flux3PrepareLatentsStep"),
        ("denoise.prepare_image_cond_latents", "Flux3PrepareImageCondLatentsStep"),
        ("denoise.set_timesteps", "Flux3SetTimestepsStep"),
        ("denoise.prepare_rope_inputs", "Flux3TextRoPEInputsStep"),
        ("denoise.denoise", "Flux3DenoiseStep"),
        ("denoise.after_denoise", "Flux3UnpackLatentsStep"),
        ("decode", "Flux3DecodeStep"),
    ],
}

# `expected_workflow_defaults`: the components, inputs and component configs each workflow's blocks carry
FLUX3_COMPONENTS = {
    "prompt_upsampler": "Qwen3_5ForConditionalGeneration",
    "prompt_upsampler_processor": "AutoProcessor",
    "text_encoder": "Qwen3VLForConditionalGeneration",
    "tokenizer": "AutoProcessor",
    "transformer": "Flux3Transformer2DModel",
    "scheduler": "FlowMatchEulerDiscreteScheduler",
    "guider": "ClassifierFreeGuidance",
    "vae": "AutoencoderKLFlux2",
    "image_processor": "Flux3ImageProcessor",
}
# the recommended sampling settings the blocks pin on the scheduler / guider they create
FLUX3_SAMPLING = {"scheduler": {"shift": 6.93}, "guider": {"guidance_scale": 3.5}}
# every optional input with its declared default. text2image declares the 1024 x 1024 size; image_conditioned derives the
# size from the first reference image, so its branch declares none
FLUX3_INPUTS = {
    "boxes": None,
    "max_new_tokens": 3584,
    "negative_prompt": None,
    "max_sequence_length": 8192,
    "text_encoder_out_layers": (4, 8, 12, 16, 20, 24, 28, 32),
    "num_images_per_prompt": 1,
    "latents": None,
    "attention_kwargs": None,
    "generator": None,
    "num_inference_steps": 50,
    "timesteps": None,
    "sigmas": None,
    "output_type": "pil",
}
# the decode step's crop-back box is produced by the image branch; text2image exposes it as an optional input
FLUX3_TEXT2IMAGE_INPUTS = {**FLUX3_INPUTS, "height": 1024, "width": 1024, "reference_crop": None}
FLUX3_IMAGE_CONDITIONED_INPUTS = {**FLUX3_INPUTS, "height": None, "width": None}
FLUX3_WORKFLOW_DEFAULTS = {
    "text2image": {
        "components": FLUX3_COMPONENTS,
        "required_inputs": ["prompt"],
        "inputs": FLUX3_TEXT2IMAGE_INPUTS,
        "component_configs": FLUX3_SAMPLING,
    },
    "image_conditioned": {
        "components": FLUX3_COMPONENTS,
        "required_inputs": ["prompt", "image"],
        "inputs": FLUX3_IMAGE_CONDITIONED_INPUTS,
        "component_configs": FLUX3_SAMPLING,
    },
}

# the step-distilled model: the same graph with distilled text encoder / text input / text RoPE / denoise steps (no
# guider, no negative prompt, 16 steps by default)
FLUX3_DISTILLED_WORKFLOWS = {
    "text2image": [
        ("prompt_upsampler", "Flux3Text2ImagePromptUpsampleStep"),
        ("text_encoder", "Flux3DistilledTextEncoderStep"),
        ("denoise.input", "Flux3DistilledTextInputStep"),
        ("denoise.prepare_latents", "Flux3PrepareLatentsStep"),
        ("denoise.set_timesteps", "Flux3SetTimestepsStep"),
        ("denoise.prepare_rope_inputs", "Flux3DistilledTextRoPEInputsStep"),
        ("denoise.denoise", "Flux3DistilledDenoiseStep"),
        ("denoise.after_denoise", "Flux3UnpackLatentsStep"),
        ("decode", "Flux3DecodeStep"),
    ],
    "image_conditioned": [
        ("prompt_upsampler", "Flux3ImageConditionedPromptUpsampleStep"),
        ("text_encoder", "Flux3DistilledTextEncoderStep"),
        ("vae_encoder.preprocess", "Flux3ProcessImagesInputStep"),
        ("vae_encoder.encode", "Flux3VaeEncoderStep"),
        ("denoise.input", "Flux3DistilledTextInputStep"),
        ("denoise.cond_input", "Flux3ImageCondInputStep"),
        ("denoise.prepare_latents", "Flux3PrepareLatentsStep"),
        ("denoise.prepare_image_cond_latents", "Flux3PrepareImageCondLatentsStep"),
        ("denoise.set_timesteps", "Flux3SetTimestepsStep"),
        ("denoise.prepare_rope_inputs", "Flux3DistilledTextRoPEInputsStep"),
        ("denoise.denoise", "Flux3DistilledDenoiseStep"),
        ("denoise.after_denoise", "Flux3UnpackLatentsStep"),
        ("decode", "Flux3DecodeStep"),
    ],
}
FLUX3_DISTILLED_COMPONENTS = {name: cls for name, cls in FLUX3_COMPONENTS.items() if name != "guider"}
FLUX3_DISTILLED_SAMPLING = {"scheduler": {"shift": 6.93}}
# no negative prompt and no attention kwargs, 16 steps by default
FLUX3_DISTILLED_INPUTS = {
    name: value for name, value in FLUX3_INPUTS.items() if name not in ("negative_prompt", "attention_kwargs")
}
FLUX3_DISTILLED_INPUTS["num_inference_steps"] = 16
FLUX3_DISTILLED_WORKFLOW_DEFAULTS = {
    "text2image": {
        "components": FLUX3_DISTILLED_COMPONENTS,
        "required_inputs": ["prompt"],
        "inputs": {**FLUX3_DISTILLED_INPUTS, "height": 1024, "width": 1024, "reference_crop": None},
        "component_configs": FLUX3_DISTILLED_SAMPLING,
    },
    "image_conditioned": {
        "components": FLUX3_DISTILLED_COMPONENTS,
        "required_inputs": ["prompt", "image"],
        "inputs": {**FLUX3_DISTILLED_INPUTS, "height": None, "width": None},
        "component_configs": FLUX3_DISTILLED_SAMPLING,
    },
}


class Flux3ModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = Flux3ModularPipeline
    pipeline_blocks_class = Flux3AutoBlocks
    pretrained_model_name_or_path = FLUX3_TINY_REPO
    # The patchified grid: `AutoencoderKLFlux2` defines its latent statistics over patchified channels.
    expected_latents_shape = (1, 16, 16, 16)
    params = frozenset(["prompt", "height", "width"])
    batch_params = frozenset()
    expected_workflow_blocks = FLUX3_WORKFLOWS
    expected_workflow_defaults = FLUX3_WORKFLOW_DEFAULTS

    @staticmethod
    def get_reference_images(sizes=((64, 64),), seed=0):
        rng = np.random.RandomState(seed)
        return [PIL.Image.fromarray(np.uint8(rng.rand(height, width, 3) * 255)) for width, height in sizes]

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a tiny fox in the snow",
            "max_new_tokens": 16,  # the tiny upsampler's caption; the real default would take minutes on CPU
            "generator": self.get_generator(seed),
            "num_inference_steps": 2,
            "height": 64,
            "width": 64,
            "max_sequence_length": 8,
            "text_encoder_out_layers": (1,),
            "output_type": "pt",
        }


class TestFlux3ModularPipelineFast(Flux3ModularPipelineTesterConfig, ModularPipelineTesterMixin):
    @pytest.mark.skip(reason="FLUX.3 takes one prompt per call; batched prompts are not supported.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="FLUX.3 takes one prompt per call; batched prompts are not supported.")
    def test_inference_batch_single_identical(self):
        pass

    def test_num_images_per_prompt(self):
        # one prompt per call, several images from it
        super().test_num_images_per_prompt(batch_sizes=[1])

    def test_multiple_reference_images(self):
        # references keep their own resolution; several of them go into one conditioning sequence
        pipe = self.get_pipeline().to(torch_device)
        inputs = {**self.get_dummy_inputs(), "num_inference_steps": 1}
        references = self.get_reference_images(sizes=((64, 48), (32, 32)))

        images = pipe(**inputs, image=references, output="images")
        assert images.shape == (1, 3, 64, 64)

        # only using the first reference changes the result
        single = pipe(**inputs, image=references[:1], output="images")
        assert not torch.allclose(images, single)

    def test_prompt_upsampling(self):
        # the prompt upsampler step, first in the default blocks, expands `prompt` into the caption the pipeline then
        # encodes. The tiny repos point the upsampler components at a tiny random Qwen3.5
        pipe = self.get_pipeline().to(torch_device)
        assert isinstance(pipe.prompt_upsampler, Qwen3_5ForConditionalGeneration)
        inputs = self.get_dummy_inputs()
        out = pipe(**inputs, output=["images", "prompt"])
        assert len(out["prompt"]) > len(inputs["prompt"])

        # without the step, the prompt is encoded as given and the image differs
        blocks = self.pipeline_blocks_class()
        blocks.sub_blocks.pop("prompt_upsampler")
        plain = blocks.init_pipeline(self.pretrained_model_name_or_path)
        plain.load_components(dtype=torch.float32)
        plain.to(torch_device)
        assert "prompt_upsampler" not in plain.components
        inputs = self.get_dummy_inputs()
        direct = plain(**inputs, output=["images", "prompt"])
        assert direct["prompt"] == inputs["prompt"]
        assert not torch.allclose(out["images"], direct["images"])

    def test_image_to_image_default_size(self):
        # without a size, image-to-image generates at the first reference's own size; a given side overrides it.
        # A single reference is padded to the image processor's 16-pixel grid and the image is cropped back
        pipe = self.get_pipeline().to(torch_device)
        inputs = {k: v for k, v in self.get_dummy_inputs().items() if k not in ("height", "width")}
        inputs["num_inference_steps"] = 1

        assert pipe(**inputs, image=self.get_reference_images(sizes=((64, 48),)), output="images").shape[-2:] == (
            48,
            64,
        )
        # 62 x 46 is padded to the 64 x 48 canvas, generated there and cropped back to 62 x 46
        assert pipe(**inputs, image=self.get_reference_images(sizes=((62, 46),)), output="images").shape[-2:] == (
            46,
            62,
        )
        # a given size is used as is, without the crop-back
        image = pipe(
            **inputs, image=self.get_reference_images(sizes=((62, 46),)), height=32, width=32, output="images"
        )
        assert image.shape[-2:] == (32, 32)
        # several references: the first one's size, each cropped to the grid
        assert pipe(**inputs, image=self.get_reference_images(sizes=((62, 46), (32, 32))), output="images").shape[
            -2:
        ] == (32, 48)

    def test_text_window_pads_to_multiple(self):
        # the prompt is right-padded to the block's pad multiple (80 for the real model), capped at `max_sequence_length`
        text_node = Flux3TextEncoderStep(pad_multiple=20).init_pipeline(self.pretrained_model_name_or_path)
        text_node.load_components(dtype=torch.float32)
        text_node.to(torch_device)

        short = text_node(prompt="a tiny fox", text_encoder_out_layers=(1,), output="prompt_embeds")
        long = text_node(prompt="a tiny fox in the snow " * 10, text_encoder_out_layers=(1,), output="prompt_embeds")
        assert short.shape[1] % 20 == 0 and long.shape[1] % 20 == 0
        assert long.shape[1] > short.shape[1]
        # the cap wins over the pad multiple
        capped = text_node(
            prompt="a tiny fox", max_sequence_length=8, text_encoder_out_layers=(1,), output="prompt_embeds"
        )
        assert capped.shape[1] == 8

    # TODO: move to `ModularGuiderTesterMixin` -- guider-disabled negative-prompt handling is generic
    def test_guider_disabled_skips_negative_prompt_with_warning(self):
        pipe = self.get_pipeline()
        inputs = {**self.get_dummy_inputs(), "negative_prompt": "blurry", "num_inference_steps": 1}

        assert pipe(**inputs, output="negative_prompt_embeds") is not None

        pipe.guider.disable()
        logger = logging.get_logger("diffusers.modular_pipelines.flux3.encoders")
        with CaptureLogger(logger) as cap:
            negative_prompt_embeds = pipe(**inputs, output="negative_prompt_embeds")
        assert negative_prompt_embeds is None
        assert "guider is disabled" in cap.out

    # TODO: generic too (huggingface/diffusers#14469): a standalone text encoder node has no guider, so it
    # encodes the negative prompt only when given one, and never a "" placeholder
    def test_text_encoder_standalone_encodes_negative_prompt_only_when_given(self):
        text_node = (
            self.pipeline_blocks_class().sub_blocks["text_encoder"].init_pipeline(self.pretrained_model_name_or_path)
        )
        text_node.load_components(dtype=torch.float32)
        text_node.to(torch_device)
        assert "guider" not in text_node.component_names

        inputs = {
            k: v
            for k, v in self.get_dummy_inputs().items()
            if k in ("prompt", "max_sequence_length", "text_encoder_out_layers")
        }
        assert text_node(**inputs, output="negative_prompt_embeds") is None
        assert text_node(**inputs, negative_prompt="blurry", output="negative_prompt_embeds") is not None

    def test_text_input_requires_negative_prompt_embeds_with_cfg(self):
        # node graph as in the docs: embeddings from a standalone text node (no negative prompt) fed to a
        # generation node whose guider needs unconditional embeddings -> a clear error, not a crash in the transformer
        blocks = self.pipeline_blocks_class()
        blocks.sub_blocks.pop("prompt_upsampler")  # the prompt side of the graph stops at the text node
        text_node = blocks.sub_blocks.pop("text_encoder").init_pipeline(self.pretrained_model_name_or_path)
        blocks.sub_blocks.pop("decode")
        generate_node = blocks.init_pipeline(self.pretrained_model_name_or_path)
        for node in (text_node, generate_node):
            node.load_components(dtype=torch.float32)
            node.to(torch_device)

        inputs = self.get_dummy_inputs()
        text_inputs = {k: inputs[k] for k in ("prompt", "max_sequence_length", "text_encoder_out_layers")}
        embeddings = text_node(**text_inputs, output=["prompt_embeds", "negative_prompt_embeds"])
        assert embeddings["negative_prompt_embeds"] is None
        generate_inputs = {k: v for k, v in inputs.items() if k not in text_inputs}
        with pytest.raises(ValueError, match="unconditional embeddings"):
            generate_node(**embeddings, **generate_inputs, output="latents")

        # with the negative prompt encoded (an empty string for plain CFG) the graph runs
        embeddings = text_node(**text_inputs, negative_prompt="", output=["prompt_embeds", "negative_prompt_embeds"])
        assert generate_node(**embeddings, **generate_inputs, output="latents") is not None


class TestFlux3ModularPipelineGuider(Flux3ModularPipelineTesterConfig, ModularGuiderTesterMixin):
    def test_guider_cfg(self):
        # the tiny random model's cond/uncond predictions differ only slightly
        super().test_guider_cfg(expected_max_diff=1e-3)


class TestFlux3ModularPipelineLoading(Flux3ModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestFlux3ModularPipelineMemory(Flux3ModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


class TestFlux3ModularPipelineWorkflow(Flux3ModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class Flux3DistilledModularPipelineTesterConfig(Flux3ModularPipelineTesterConfig):
    pipeline_class = Flux3DistilledModularPipeline
    pipeline_blocks_class = Flux3DistilledAutoBlocks
    pretrained_model_name_or_path = FLUX3_DISTILLED_TINY_REPO
    expected_workflow_blocks = FLUX3_DISTILLED_WORKFLOWS
    expected_workflow_defaults = FLUX3_DISTILLED_WORKFLOW_DEFAULTS


class TestFlux3DistilledModularPipelineFast(Flux3DistilledModularPipelineTesterConfig, ModularPipelineTesterMixin):
    @pytest.mark.skip(reason="FLUX.3 takes one prompt per call; batched prompts are not supported.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="FLUX.3 takes one prompt per call; batched prompts are not supported.")
    def test_inference_batch_single_identical(self):
        pass

    def test_num_images_per_prompt(self):
        super().test_num_images_per_prompt(batch_sizes=[1])

    def test_no_guider_no_negative_prompt(self):
        # the step-distilled model samples without classifier-free guidance: no guider, and the block set neither
        # takes a negative prompt nor produces negative embeddings
        pipe = self.get_pipeline()
        assert "guider" not in pipe.component_names
        blocks = self.pipeline_blocks_class()
        assert {"negative_prompt", "negative_prompt_embeds"}.isdisjoint({p.name for p in blocks.inputs})
        assert {"negative_prompt_embeds", "negative_txt_ids"}.isdisjoint({p.name for p in blocks.intermediate_outputs})

    def test_image_to_image(self):
        pipe = self.get_pipeline()
        inputs = {**self.get_dummy_inputs(), "image": self.get_reference_images(sizes=((64, 48), (32, 32)))}
        images = pipe(**inputs, output="images")
        assert images.shape == (1, 3, 64, 64)


class TestFlux3DistilledModularPipelineLoading(Flux3DistilledModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestFlux3DistilledModularPipelineWorkflow(Flux3DistilledModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestFlux3ModularPipelineLoRA(Flux3ModularPipelineTesterConfig, ModularLoraTesterMixin):
    pass
