# Copyright 2026 The HuggingFace Team.
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

import re

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from diffusers.modular_pipelines import PH3ImageBlocks, PH3ImageModularPipeline
from diffusers.modular_pipelines.p_h3_image.encoders import (
    PH3ImageEditTextEncoderStep,
    PH3ImageEditVaeEncoderStep,
    PH3ImageTextEncoderStep,
)
from diffusers.modular_pipelines.p_h3_image.prompt_upsampler import (
    PH3ImageEditPromptUpsampleStep,
    aspect_ratio_label,
    is_edit_prompt,
)
from diffusers.utils import logging

from ...testing_utils import CaptureLogger
from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


# The blocks each workflow of `PH3ImageBlocks` runs, in order.
P_H3_IMAGE_WORKFLOWS = {
    "text2image": [
        ("prompt_upsampler", "PH3ImageText2ImagePromptUpsampleStep"),
        ("text_encoder", "PH3ImageTextEncoderStep"),
        ("denoise.prepare_layout", "PH3ImagePrepareLayoutStep"),
        ("denoise.prepare_latents", "PH3ImagePrepareLatentsStep"),
        ("denoise.set_timesteps", "PH3ImageSetTimestepsStep"),
        ("denoise.denoise", "PH3ImageDenoiseStep"),
        ("denoise.after_denoise", "PH3ImageAfterDenoiseStep"),
        ("decode", "PH3ImageDecodeStep"),
    ],
    "image_conditioned": [
        ("before_encode", "PH3ImageEditResizeStep"),
        ("prompt_upsampler", "PH3ImageEditPromptUpsampleStep"),
        ("text_encoder", "PH3ImageEditTextEncoderStep"),
        ("vae_encoder", "PH3ImageEditVaeEncoderStep"),
        ("denoise.prepare_layout", "PH3ImageEditPrepareLayoutStep"),
        ("denoise.prepare_condition_latents", "PH3ImagePrepareConditionLatentsStep"),
        ("denoise.prepare_latents", "PH3ImagePrepareLatentsStep"),
        ("denoise.prepare_latents_image_conditioned", "PH3ImageEditPrepareLatentsStep"),
        ("denoise.set_timesteps", "PH3ImageSetTimestepsStep"),
        ("denoise.denoise", "PH3ImageDenoiseStep"),
        ("denoise.after_denoise", "PH3ImageAfterDenoiseStep"),
        ("decode", "PH3ImageDecodeStep"),
    ],
}

_SHARED_COMPONENTS = {
    "prompt_upsampler": "Qwen3_5ForConditionalGeneration",
    "prompt_upsampler_processor": "AutoProcessor",
    "text_encoder": "Qwen3VLForConditionalGeneration",
    "text_connector": "PH3ImageTextConnector",
    "tokenizer": "Qwen2Tokenizer",
    "processor": "Qwen3VLProcessor",
    "scheduler": "MiniMaxH3Scheduler",
    "audio_scheduler": "MiniMaxH3Scheduler",
    "transformer": "MiniMaxH3Transformer3DModel",
    "vae": "AutoencoderKLPH3Image",
    "image_processor": "VaeImageProcessor",
}
_SAMPLE_SIGMAS = [1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0]
_SHARED_INPUTS = {
    "generator": None,
    "latents": None,
    "sigmas": None,
    "attention_kwargs": None,
    "output_type": "pil",
}
P_H3_IMAGE_WORKFLOW_DEFAULTS = {
    "text2image": {
        "components": _SHARED_COMPONENTS,
        "configs": {"sample_sigmas": _SAMPLE_SIGMAS},
        "required_inputs": ["prompt"],
        "inputs": {**_SHARED_INPUTS, "height": 2912, "width": 2912, "max_new_tokens": 1024},
    },
    "image_conditioned": {
        "components": _SHARED_COMPONENTS,
        "configs": {
            "reference_min_pixels": 1024**2,
            "reference_max_pixels": 8 * 1024**2,
            "canvas_min_pixels": 4 * 1024**2,
            "vision_pixels": 1_000_000,
            "sample_sigmas": _SAMPLE_SIGMAS,
        },
        "required_inputs": ["prompt", "image"],
        "inputs": {**_SHARED_INPUTS, "height": None, "width": None, "max_new_tokens": 1536},
    },
}


_EDIT_BRIEF = (
    "subject_definitions:\n<Picture 1> is the image to edit.\n"
    "summary:\n{request} [keyframe completion]\n"
    "retention_analysis:\n<Picture 1>: fully_preserved - the composition\n"
    "detailed_description:\nThe image after the edit: {request}\n"
    "overall_soundscape:\nSilence.\n"
    "non_diegetic_music:\nNone."
)


@pytest.fixture(autouse=True, scope="module")
def tiny_prompt_upsampler():
    r"""
    The tiny prompt upsampler has random weights and cannot write a usable prompt, so its generation is replaced by
    a prompt built from the request: a sentence for text-to-image, a brief for an edit.
    """
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-flux3-prompt-upsampler")

    def generate(self, input_ids, **kwargs):
        request = tokenizer.decode(input_ids[0])
        edit = re.search(r"Edit <Picture 1>\. (.*?)<\|im_end\|>", request, re.S)
        if edit:
            prompt = _EDIT_BRIEF.format(request=edit.group(1))
        else:
            text = re.search(r"Request: (.*?)<\|im_end\|>", request, re.S).group(1)
            prompt = f"A detailed picture of {text}."
        answer = tokenizer(f"Thinking.\n</think>\n\n{prompt}<|im_end|>", add_special_tokens=False, return_tensors="pt")
        return torch.cat([input_ids, answer["input_ids"].to(input_ids.device)], dim=1)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(Qwen3_5ForConditionalGeneration, "generate", generate)
        yield


class PH3ImageModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    """The `text2image` and `image_conditioned` requests of [`PH3ImageBlocks`]."""

    pipeline_class = PH3ImageModularPipeline
    pipeline_blocks_class = PH3ImageBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-p-h3-image-modular-pipe"
    # One 4-channel latent frame for a 32 x 32 image.
    expected_latents_shape = (1, 4, 1, 2, 2)
    params = frozenset(["prompt", "image", "height", "width"])
    # One request is one packed sequence, so prompts are not batched.
    batch_params = frozenset()
    # No guidance, one image per request, fixed sampling grid: no `guidance_scale`, `num_images_per_prompt` or
    # `num_inference_steps`.
    optional_params = frozenset(["latents", "output_type"])
    output_name = "images"
    expected_workflow_blocks = P_H3_IMAGE_WORKFLOWS
    expected_workflow_defaults = P_H3_IMAGE_WORKFLOW_DEFAULTS

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a robot dancing",
            "generator": self.get_generator(seed),
            "height": 32,
            "width": 32,
            "output_type": "pt",
        }

    def get_edit_inputs(self, seed=0):
        image = Image.fromarray((np.random.default_rng(0).random((40, 56, 3)) * 255).astype("uint8"))
        return {
            "prompt": "make it blue",
            "image": image,
            "generator": self.get_generator(seed),
            "height": 32,
            "width": 64,
            "output_type": "pt",
        }


class TestPH3ImageModularPipelineFast(PH3ImageModularPipelineTesterConfig, ModularPipelineTesterMixin):
    @pytest.mark.skip(reason="P-H3-Image packs one request into one sequence, so a batch of prompts is not a thing.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="P-H3-Image packs one request into one sequence, so a batch of prompts is not a thing.")
    def test_inference_batch_single_identical(self):
        pass

    @pytest.mark.skip(reason="One request generates one image, so there is no `num_images_per_prompt`.")
    def test_num_images_per_prompt(self):
        pass

    def test_edit_output(self):
        r"""The edit is generated on the requested canvas, and its conditioning rows are dropped before decoding."""
        pipe = self.get_pipeline()

        state = pipe(**self.get_edit_inputs())

        assert state.get("images").shape == (1, 3, 32, 64)
        # A 2 x 4 latent grid in 2 x 2 patches.
        assert state.get("num_condition_video_rows") == 2
        assert state.get("latents").shape == (1, 4, 1, 2, 4)

    def test_edit_keeps_the_image_size(self):
        r"""Without `height` and `width`, the edit keeps the size of the image, rounded to multiples of 32."""
        pipe = self.get_pipeline()
        pipe.update_components(reference_min_pixels=32 * 32, reference_max_pixels=64 * 128)

        inputs = self.get_edit_inputs()
        inputs.pop("height")
        inputs.pop("width")
        inputs["image"] = Image.new("RGB", (100, 60))
        state = pipe(**inputs)

        assert (state.get("width"), state.get("height")) == (96, 64)
        assert state.get("reference_images")[0].size == (96, 64)

    @pytest.mark.parametrize(
        "size,expected",
        # 56 x 40 has an aspect ratio of 1.4: sqrt(8192 * 1.4) = 107.1 rounds to 96, sqrt(8192 / 1.4) = 76.5 to 64.
        [((56, 40), (96, 64)), ((400, 200), (128, 64))],
        ids=["scaled_up", "scaled_down"],
    )
    def test_edit_scales_an_image_outside_the_area_range(self, size, expected):
        r"""An image below `reference_min_pixels` or above `reference_max_pixels` is scaled to that area."""
        pipe = self.get_pipeline()
        pipe.update_components(reference_min_pixels=64 * 128, reference_max_pixels=64 * 128)

        inputs = self.get_edit_inputs()
        inputs.pop("height")
        inputs.pop("width")
        inputs["image"] = Image.new("RGB", size)
        state = pipe(**inputs)

        assert (state.get("width"), state.get("height")) == expected
        assert state.get("reference_images")[0].size == expected

    def test_edit_scales_up_a_small_first_image(self):
        r"""Without `height` and `width`, the first image is scaled up to `canvas_min_pixels`; the others are not."""
        pipe = self.get_pipeline()
        pipe.update_components(
            reference_min_pixels=32 * 32, reference_max_pixels=128 * 128, canvas_min_pixels=128 * 64
        )

        inputs = self.get_edit_inputs()
        inputs.pop("height")
        inputs.pop("width")
        inputs["image"] = [Image.new("RGB", (64, 32)), Image.new("RGB", (64, 32))]
        state = pipe(**inputs)

        assert (state.get("width"), state.get("height")) == (128, 64)
        assert [image.size for image in state.get("reference_images")] == [(128, 64), (64, 32)]

    def test_edit_canvas_leaves_the_image_size(self):
        r"""`height` and `width` set the canvas of the edit; the image keeps its own size."""
        pipe = self.get_pipeline()
        pipe.update_components(reference_min_pixels=32 * 32, reference_max_pixels=64 * 128)

        inputs = self.get_edit_inputs()
        inputs["image"] = Image.new("RGB", (96, 64))
        state = pipe(**inputs)

        assert (state.get("width"), state.get("height")) == (64, 32)
        assert state.get("reference_images")[0].size == (96, 64)
        # 2 x 3 patches for the image, 1 x 2 for the canvas.
        assert state.get("num_condition_video_rows") == 6
        assert state.get("images").shape == (1, 3, 32, 64)

    def test_multi_image_edit(self):
        r"""Every image becomes a reference block at its own size; the edit is generated at the size of the first."""
        pipe = self.get_pipeline()
        pipe.update_components(reference_min_pixels=32 * 32, reference_max_pixels=64 * 128)

        inputs = self.get_edit_inputs()
        inputs.pop("height")
        inputs.pop("width")
        inputs["image"] = [Image.new("RGB", (96, 64)), Image.new("RGB", (64, 64))]
        state = pipe(**inputs)

        assert (state.get("width"), state.get("height")) == (96, 64)
        assert [image.size for image in state.get("reference_images")] == [(96, 64), (64, 64)]
        # 2 x 3 patches for the image to edit, 2 x 2 for the second image.
        assert state.get("num_condition_video_rows") == 6 + 4
        assert state.get("images").shape == (1, 3, 64, 96)

    @pytest.mark.parametrize("edit", [False, True], ids=["text2image", "image_conditioned"])
    def test_generator_reproducibility(self, edit):
        r"""The same seed reproduces the output and a different seed changes it."""
        pipe = self.get_pipeline()

        def run(seed):
            inputs = self.get_edit_inputs(seed) if edit else self.get_dummy_inputs(seed)
            return pipe(**inputs, output="images")

        assert torch.equal(run(7), run(7))
        assert not torch.equal(run(7), run(8))

    def test_custom_sigmas(self):
        r"""`sigmas` replaces the checkpoint's `sample_sigmas`."""
        pipe = self.get_pipeline()

        state = pipe(**self.get_dummy_inputs(), sigmas=[1.0, 0.5, 0.0])

        assert len(state.get("timesteps")) == 2
        torch.testing.assert_close(state.get("timesteps").cpu(), torch.tensor([0.0, 0.5]))

    @pytest.mark.parametrize("edit", [False, True], ids=["text2image", "image_conditioned"])
    def test_text_encoder_block_standalone(self, edit):
        r"""Both text encoder blocks run on their own. Vision tokens of an edit are tagged as video (`0`)."""
        block = PH3ImageEditTextEncoderStep() if edit else PH3ImageTextEncoderStep()
        pipe = block.init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)

        request = {"vision_images": [Image.new("RGB", (32, 32))]} if edit else {}
        outputs = pipe(prompt="a robot dancing", **request, output=["prompt_embeds", "text_token_tags"])

        assert set(pipe.components) == {"text_encoder", "text_connector", "tokenizer", "processor"}
        assert outputs["prompt_embeds"].shape[1] == outputs["text_token_tags"].shape[0]
        assert outputs["prompt_embeds"].shape[2] == pipe.text_connector.config.out_channels
        assert set(outputs["text_token_tags"].tolist()) == ({0, 1} if edit else {1})

    def test_vae_encoder_block_standalone(self):
        r"""The edit's VAE encoder block runs on its own."""
        pipe = PH3ImageEditVaeEncoderStep().init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)

        condition_latents = pipe(reference_images=[Image.new("RGB", (64, 32))], output="condition_latents")

        assert set(pipe.components) == {"vae"}
        assert len(condition_latents) == 1
        assert condition_latents[0].shape == (1, 4, 1, 2, 4)

    @pytest.mark.parametrize("output_type", ["np", "pil", "pt"])
    def test_output_type(self, output_type):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["output_type"] = output_type
        images = pipe(**inputs, output="images")

        if output_type == "np":
            assert images.shape == (1, 32, 32, 3)
        elif output_type == "pil":
            assert images[0].size == (32, 32)
        else:
            assert images.shape == (1, 3, 32, 32)

    def test_prompt_list(self):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = ["a robot", "a fox"]

        with pytest.raises(ValueError, match="must be a single string"):
            pipe(**inputs)

    @pytest.mark.parametrize("edit", [False, True], ids=["text2image", "image_conditioned"])
    def test_size_rounded_to_multiples_of_32(self, edit):
        r"""A `height` and `width` that are not multiples of 32 are rounded to the nearest ones, with a warning."""
        pipe = self.get_pipeline()

        inputs = self.get_edit_inputs() if edit else self.get_dummy_inputs()
        inputs["height"], inputs["width"] = 30, 70
        logger = logging.get_logger("diffusers.modular_pipelines.p_h3_image.before_encoder")
        with CaptureLogger(logger) as cap:
            images = pipe(**inputs, output="images")

        assert images.shape == (1, 3, 32, 64)
        assert "30x70 becomes 32x64" in cap.out

    def test_edit_height_without_width(self):
        pipe = self.get_pipeline()

        inputs = self.get_edit_inputs()
        inputs["width"] = None

        with pytest.raises(ValueError, match="have to be passed together"):
            pipe(**inputs)

    def get_upsampling_pipeline(self, answer):
        r"""The default blocks, with the prompt upsampler answering `answer` and recording the prompt it was given."""
        pipe = self.pipeline_blocks_class().init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)
        tokenizer = pipe.prompt_upsampler_processor.tokenizer
        answer_ids = tokenizer(answer + "<|im_end|>", add_special_tokens=False, return_tensors="pt")["input_ids"]
        seen = {}

        def generate(input_ids, **kwargs):
            seen["request"] = tokenizer.decode(input_ids[0])
            return torch.cat([input_ids, answer_ids.to(input_ids.device)], dim=1)

        pipe.prompt_upsampler.generate = generate
        return pipe, seen

    def test_prompt_upsampling(self):
        r"""The upsampler writes the prompt for the aspect ratio of the canvas; the text after `</think>` replaces it."""
        pipe, seen = self.get_upsampling_pipeline(
            "It is a fox in snow.\n</think>\n\nA red fox curled up in fresh snow."
        )

        inputs = self.get_dummy_inputs()
        inputs["width"] = 64
        state = pipe(**inputs)

        assert "Aspect ratio: 16:9\nRequest: a robot dancing" in seen["request"]
        assert state.get("prompt") == "A red fox curled up in fresh snow."
        assert state.get("images").shape == (1, 3, 32, 64)

    def test_edit_prompt_upsampling(self):
        r"""For an edit, the upsampler sees every image and its brief replaces the instruction."""
        brief = (
            "subject_definitions:\n<Picture 1> is the image to edit.\n"
            "summary:\nThe same scene, painted blue. [keyframe completion]\n"
            "retention_analysis:\n<Picture 1>: fully_preserved - the composition\n"
            "detailed_description:\nA blue square.\n"
            "overall_soundscape:\nSilence.\n"
            "non_diegetic_music:\nNone."
        )
        pipe, seen = self.get_upsampling_pipeline(f"Paint it blue.\n</think>\n\n{brief}")

        inputs = self.get_edit_inputs()
        inputs["image"] = [inputs["image"], Image.new("RGB", (32, 32))]
        state = pipe(**inputs)

        assert "<Picture 2>: " in seen["request"] and "Edit <Picture 1>. make it blue" in seen["request"]
        assert state.get("prompt") == brief
        assert state.get("images").shape == (1, 3, 32, 64)

    def test_prompt_upsampler_rejects_an_unusable_prompt(self):
        pipe, _ = self.get_upsampling_pipeline("A prompt without the end of the reasoning.")

        with pytest.raises(RuntimeError, match="usable prompt"):
            pipe(**self.get_dummy_inputs())

    def test_prompt_upsampler_image_limit(self):
        r"""The upsampler block run on its own takes the same 8 images as the pipeline."""
        pipe = PH3ImageEditPromptUpsampleStep().init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)

        with pytest.raises(ValueError, match="at most 8 images"):
            pipe(prompt="make it blue", image=[Image.new("RGB", (32, 32))] * 9)

    def test_prompt_upsampler_step_can_be_removed(self):
        blocks = self.pipeline_blocks_class()
        blocks.sub_blocks.pop("prompt_upsampler")
        pipe = blocks.init_pipeline(self.pretrained_model_name_or_path)

        assert "prompt_upsampler" not in pipe.components
        assert "prompt_upsampler_processor" not in pipe.components

    def test_upsampler_helpers(self):
        assert aspect_ratio_label(2912, 2912) == "1:1"
        assert aspect_ratio_label(3872, 2176) == "16:9"
        assert aspect_ratio_label(1664, 2496) == "2:3"
        assert is_edit_prompt("Edit <Picture 1>. Make it night. Keep the boat.")
        assert not is_edit_prompt("Make it night.")
        assert not is_edit_prompt("summary:\nA night scene.")

    def test_edit_prefix(self):
        r"""A short instruction gets the `"Edit <Picture 1>. "` prefix; a prompt that already has it is encoded as given."""
        block = PH3ImageEditTextEncoderStep()
        pipe = block.init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)
        vision_images = [Image.new("RGB", (32, 32))]

        short = pipe(prompt="make it blue", vision_images=vision_images, output="prompt_embeds")
        full = pipe(prompt="Edit <Picture 1>. make it blue", vision_images=vision_images, output="prompt_embeds")

        assert torch.equal(short, full)

    def test_edit_too_many_images(self):
        pipe = self.get_pipeline()

        inputs = self.get_edit_inputs()
        inputs["image"] = [Image.new("RGB", (32, 32))] * 9

        with pytest.raises(ValueError, match="at most 8 images"):
            pipe(**inputs)


class TestPH3ImageModularPipelineLoading(PH3ImageModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestPH3ImageModularPipelineWorkflow(PH3ImageModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestPH3ImageModularPipelineMemory(PH3ImageModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass
