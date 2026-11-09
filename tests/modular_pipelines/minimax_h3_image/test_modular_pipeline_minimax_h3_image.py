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

import numpy as np
import pytest
import torch
from PIL import Image

from diffusers.modular_pipelines import MiniMaxH3ImageBlocks, MiniMaxH3ImageModularPipeline
from diffusers.modular_pipelines.minimax_h3_image.encoders import (
    MiniMaxH3ImageEditTextEncoderStep,
    MiniMaxH3ImageEditVaeEncoderStep,
    MiniMaxH3ImageTextEncoderStep,
)

from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


# The blocks each workflow of `MiniMaxH3ImageBlocks` runs, in order.
MINIMAX_H3_IMAGE_WORKFLOWS = {
    "text2image": [
        ("text_encoder", "MiniMaxH3ImageTextEncoderStep"),
        ("denoise.prepare_canvas", "MiniMaxH3ImagePrepareCanvasStep"),
        ("denoise.prepare_layout", "MiniMaxH3ImagePrepareLayoutStep"),
        ("denoise.prepare_latents", "MiniMaxH3ImagePrepareLatentsStep"),
        ("denoise.set_timesteps", "MiniMaxH3ImageSetTimestepsStep"),
        ("denoise.denoise", "MiniMaxH3ImageDenoiseStep"),
        ("denoise.after_denoise", "MiniMaxH3ImageAfterDenoiseStep"),
        ("decode", "MiniMaxH3ImageDecodeStep"),
    ],
    "image_conditioned": [
        ("before_encode", "MiniMaxH3ImageEditResizeStep"),
        ("text_encoder", "MiniMaxH3ImageEditTextEncoderStep"),
        ("vae_encoder", "MiniMaxH3ImageEditVaeEncoderStep"),
        ("denoise.prepare_canvas", "MiniMaxH3ImagePrepareCanvasStep"),
        ("denoise.prepare_layout", "MiniMaxH3ImageEditPrepareLayoutStep"),
        ("denoise.prepare_latents", "MiniMaxH3ImagePrepareLatentsStep"),
        ("denoise.prepare_condition_latents", "MiniMaxH3ImageEditPrepareLatentsStep"),
        ("denoise.set_timesteps", "MiniMaxH3ImageSetTimestepsStep"),
        ("denoise.denoise", "MiniMaxH3ImageDenoiseStep"),
        ("denoise.after_denoise", "MiniMaxH3ImageAfterDenoiseStep"),
        ("decode", "MiniMaxH3ImageDecodeStep"),
    ],
}

_SHARED_COMPONENTS = {
    "text_encoder": "Qwen3VLForConditionalGeneration",
    "tokenizer": "Qwen2Tokenizer",
    "processor": "Qwen3VLProcessor",
    "scheduler": "MiniMaxH3Scheduler",
    "transformer": "MiniMaxH3ImageTransformer2DModel",
    "vae": "AutoencoderKLMiniMaxH3Image",
    "image_processor": "VaeImageProcessor",
}
_SAMPLE_SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]
# No guidance and a fixed sampling grid with one output head per step: no `guidance_scale`, no `negative_prompt` and no
# `num_inference_steps`.
_SHARED_INPUTS = {
    "height": 2048,
    "width": 2048,
    "min_generation_area": 2048 * 2048,
    "generator": None,
    "latents": None,
    "output_type": "pil",
}
MINIMAX_H3_IMAGE_WORKFLOW_DEFAULTS = {
    "text2image": {
        "components": _SHARED_COMPONENTS,
        "configs": {"sample_sigmas": _SAMPLE_SIGMAS},
        "required_inputs": ["prompt"],
        "inputs": _SHARED_INPUTS,
    },
    "image_conditioned": {
        "components": _SHARED_COMPONENTS,
        "configs": {
            "reference_min_short_edge": 1024,
            "reference_max_short_edge": 2048,
            "vision_min_short_edge": 1024,
            "vision_max_short_edge": 1024,
            "sample_sigmas": _SAMPLE_SIGMAS,
        },
        "required_inputs": ["prompt", "image"],
        # An edit also draws one noise tensor per reference image, which can be passed in like `latents`.
        "inputs": {**_SHARED_INPUTS, "reference_noise": None},
    },
}


class MiniMaxH3ImageModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    """The `text2image` and `image_conditioned` requests of [`MiniMaxH3ImageBlocks`]."""

    pipeline_class = MiniMaxH3ImageModularPipeline
    pipeline_blocks_class = MiniMaxH3ImageBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-minimax-h3-image-modular-pipe"
    # 4 latent channels at a quarter of a 32 x 32 image: the tiny VAE compresses 4x.
    expected_latents_shape = (1, 4, 8, 8)
    params = frozenset(["prompt", "image", "height", "width"])
    # One request is one packed sequence, so prompts are not batched.
    batch_params = frozenset()
    optional_params = frozenset(["latents", "output_type"])
    output_name = "images"
    expected_workflow_blocks = MINIMAX_H3_IMAGE_WORKFLOWS
    expected_workflow_defaults = MINIMAX_H3_IMAGE_WORKFLOW_DEFAULTS

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a robot dancing",
            "generator": self.get_generator(seed),
            "height": 32,
            "width": 32,
            # The smallest area: generate at the requested size.
            "min_generation_area": 1024,
            "output_type": "pt",
        }

    def get_edit_inputs(self, seed=0):
        image = Image.fromarray((np.random.default_rng(0).random((40, 56, 3)) * 255).astype("uint8"))
        return {
            "prompt": "make <Picture 1> blue",
            "image": image,
            "generator": self.get_generator(seed),
            "height": 32,
            "width": 64,
            "min_generation_area": 1024,
            "output_type": "pt",
        }


class TestMiniMaxH3ImageModularPipelineFast(MiniMaxH3ImageModularPipelineTesterConfig, ModularPipelineTesterMixin):
    @pytest.mark.skip(reason="MiniMax-H3-Image packs one request into one sequence, so there is no batch of prompts.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="MiniMax-H3-Image packs one request into one sequence, so there is no batch of prompts.")
    def test_inference_batch_single_identical(self):
        pass

    @pytest.mark.skip(reason="One request generates one image, so there is no `num_images_per_prompt`.")
    def test_num_images_per_prompt(self):
        pass

    def test_edit_output(self):
        r"""The edit is generated on the requested canvas, and its reference rows are dropped before decoding."""
        pipe = self.get_pipeline()

        state = pipe(**self.get_edit_inputs())

        assert state.get("images").shape == (1, 3, 32, 64)
        # The 56 x 40 reference keeps its size: a 14 x 10 latent grid in 2 x 2 patches.
        assert state.get("reference_images")[0].size == (56, 40)
        assert state.get("num_reference_rows") == 7 * 5
        assert state.get("latents").shape == (1, 4, 8, 16)

    def test_multi_image_edit(self):
        r"""Every reference image becomes a block of rows at its own size."""
        pipe = self.get_pipeline()

        inputs = self.get_edit_inputs()
        inputs["image"] = [Image.new("RGB", (48, 32)), Image.new("RGB", (32, 32))]
        inputs["prompt"] = "put <Picture 2> into <Picture 1>"
        state = pipe(**inputs)

        assert [image.size for image in state.get("reference_images")] == [(48, 32), (32, 32)]
        assert state.get("num_reference_rows") == 6 * 4 + 4 * 4
        assert state.get("images").shape == (1, 3, 32, 64)

    @pytest.mark.parametrize(
        "size,expected",
        # Short edge 20 is scaled up to 32: 44 * 1.6 = 70.4 rounds to 72. Short edge 80 is scaled down to 64:
        # 100 * 0.8 = 80.
        [((44, 20), (72, 32)), ((100, 80), (80, 64)), ((60, 50), (60, 50))],
        ids=["scaled_up", "scaled_down", "kept"],
    )
    def test_edit_bounds_the_short_edge_of_a_reference(self, size, expected):
        r"""A reference is resized only when its short edge is outside the bounds; inside, it keeps its pixels."""
        pipe = self.get_pipeline()
        pipe.update_components(reference_min_short_edge=32, reference_max_short_edge=64)

        inputs = self.get_edit_inputs()
        inputs["image"] = Image.new("RGB", size)
        state = pipe(**inputs)

        assert state.get("reference_images")[0].size == expected
        # The text encoder gets its own copy, on its own bounds.
        assert min(state.get("vision_images")[0].size) == 32

    def test_edit_crops_a_reference_to_whole_patches(self):
        r"""A reference that is not a whole number of patches is cropped at the top left, not resized."""
        pipe = self.get_pipeline()

        inputs = self.get_edit_inputs()
        # 54 x 42 pixels: 13 x 10 latents after the crop to whole VAE cells, 12 x 10 after the crop to whole patches.
        inputs["image"] = Image.new("RGB", (54, 42))
        state = pipe(**inputs)

        assert state.get("reference_images")[0].size == (54, 42)
        assert state.get("reference_latents")[0].shape == (1, 4, 10, 12)
        assert state.get("num_reference_rows") == 6 * 5

    def test_edit_reference_noise(self):
        r"""`reference_noise` replaces the draws for the reference images, the way `latents` does for the image."""
        pipe = self.get_pipeline()

        def run(reference_seed):
            inputs = self.get_edit_inputs()
            # A 14 x 10 latent grid for the 56 x 40 reference.
            noise = torch.randn(1, 4, 10, 14, generator=torch.Generator().manual_seed(reference_seed))
            return pipe(**inputs, reference_noise=[noise], output="images")

        assert torch.equal(run(1), run(1))
        assert not torch.equal(run(1), run(2))

    def test_edit_reference_noise_count(self):
        pipe = self.get_pipeline()

        with pytest.raises(ValueError, match="one tensor per reference image"):
            pipe(**self.get_edit_inputs(), reference_noise=[])

    @pytest.mark.parametrize("edit", [False, True], ids=["text2image", "image_conditioned"])
    def test_min_generation_area(self, edit):
        r"""An image below `min_generation_area` is generated on a larger canvas and resized back."""
        pipe = self.get_pipeline()

        inputs = self.get_edit_inputs() if edit else self.get_dummy_inputs()
        inputs.update(height=20, width=28, min_generation_area=64 * 64)
        state = pipe(**inputs)

        # sqrt(4096 / 560) = 2.70: 54.1 x 75.7 rounds to 56 x 72 on the tiny canvas grid of 8.
        assert (state.get("height"), state.get("width")) == (56, 72)
        assert state.get("latents").shape == (1, 4, 14, 18)
        assert state.get("images").shape == (1, 3, 20, 28)

    def test_min_generation_area_out_of_range(self):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["min_generation_area"] = 1023

        with pytest.raises(ValueError, match="min_generation_area"):
            pipe(**inputs)

    @pytest.mark.parametrize("edit", [False, True], ids=["text2image", "image_conditioned"])
    def test_generator_reproducibility(self, edit):
        r"""The same seed reproduces the output and a different seed changes it."""
        pipe = self.get_pipeline()

        def run(seed):
            inputs = self.get_edit_inputs(seed) if edit else self.get_dummy_inputs(seed)
            return pipe(**inputs, output="images")

        assert torch.equal(run(7), run(7))
        assert not torch.equal(run(7), run(8))

    def test_one_step_per_output_head(self):
        r"""The schedule has one step per output head of the transformer, on the checkpoint's sampling grid."""
        pipe = self.get_pipeline()

        state = pipe(**self.get_dummy_inputs())

        assert len(state.get("timesteps")) == pipe.transformer.config.num_pdd_heads == len(_SAMPLE_SIGMAS) - 1
        expected = 1.0 - torch.tensor(_SAMPLE_SIGMAS[:-1])
        torch.testing.assert_close(state.get("timesteps").cpu(), expected)

    def test_reference_rows_are_held_at_the_conditioning_level(self):
        r"""The reference rows of an edit read a fixed timestep on every step; the generated rows step down."""
        pipe = self.get_pipeline()

        state = pipe(**self.get_edit_inputs())

        num_reference_rows = state.get("num_reference_rows")
        image_indices = state.get("image_indices").cpu()
        for step, (timesteps, indices) in enumerate(state.get("row_timestep_plan")):
            row_timesteps = timesteps.cpu()[indices.cpu()]
            assert torch.all(row_timesteps[image_indices[:num_reference_rows]] == pipe.reference_noise_aug)
            assert torch.all(row_timesteps[image_indices[num_reference_rows:]] == state.get("timesteps")[step].cpu())

    @pytest.mark.parametrize("edit", [False, True], ids=["text2image", "image_conditioned"])
    def test_text_encoder_block_standalone(self, edit):
        r"""Both text encoder blocks run on their own. Vision tokens of an edit are tagged as image (`0`)."""
        block = MiniMaxH3ImageEditTextEncoderStep() if edit else MiniMaxH3ImageTextEncoderStep()
        pipe = block.init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)

        request = {"vision_images": [Image.new("RGB", (32, 32))]} if edit else {}
        outputs = pipe(prompt="a robot dancing", **request, output=["prompt_embeds", "text_token_tags"])

        assert set(pipe.components) == {"text_encoder", "tokenizer", "processor"}
        assert outputs["prompt_embeds"].shape[1] == outputs["text_token_tags"].shape[0]
        assert set(outputs["text_token_tags"].tolist()) == ({0, 1} if edit else {1})

    def test_vae_encoder_block_standalone(self):
        r"""The edit's VAE encoder block runs on its own, and its posterior sample follows the generator."""
        pipe = MiniMaxH3ImageEditVaeEncoderStep().init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)

        def run(seed):
            return pipe(
                reference_images=[Image.new("RGB", (64, 32))],
                generator=self.get_generator(seed),
                output="reference_latents",
            )

        reference_latents = run(0)

        assert set(pipe.components) == {"vae"}
        assert len(reference_latents) == 1
        assert reference_latents[0].shape == (1, 4, 8, 16)
        assert torch.equal(reference_latents[0], run(0)[0])
        assert not torch.equal(reference_latents[0], run(1)[0])

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

        with pytest.raises(ValueError, match="must be a single non-empty string"):
            pipe(**inputs)


class TestMiniMaxH3ImageModularPipelineLoading(MiniMaxH3ImageModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestMiniMaxH3ImageModularPipelineWorkflow(MiniMaxH3ImageModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestMiniMaxH3ImageModularPipelineMemory(MiniMaxH3ImageModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass
