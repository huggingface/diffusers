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

import pytest
import torch
from PIL import Image

from diffusers.modular_pipelines import (
    MiniMaxH3Blocks,
    MiniMaxH3ModularPipeline,
    MiniMaxH3Ref2VABlocks,
    MiniMaxH3Ref2VAModularPipeline,
)
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep
from diffusers.pipelines.minimax_h3 import MiniMaxH3Reference, packing, packing_ref2va

from ..test_modular_pipelines_common import ModularPipelineTesterMixin, _get_specified_components


# MiniMax-H3 generates 5 to 15 seconds at a fixed 24 fps, so the shortest admissible request is 124 frames
# (`17 * 7 + 5`, the next length the video VAE can decode). It is affordable on CPU because the canvas is tiny: 32
# pixels is a single `(1, 2, 2)` patch row per latent frame.
NUM_FRAMES = 124
RESOLUTION = 32
NUM_AUDIO_LATENTS = 207
# `prod(encoder_rates)` of the tiny audio VAE, standing in for the released 800 samples per latent.
AUDIO_HOP_LENGTH = 4

# The released short edge of an image reference is 2048 pixels, which packs thousands of conditioning rows.
TEST_REFERENCE_IMAGE_SHORT_EDGE = 64

# Both blocksets read the same repository, as the released one does: `transformer/` serves the `t2va` / `fl2va` half
# and `transformer_ref/` the `ref2va` one, every other component is shared.
TINY_MODULAR_REPO_ID = "diffusers-internal-dev/tiny-minimax-h3-modular-pipe"


T2VA_WORKFLOW = [
    ("setup", "MiniMaxH3SetupStep"),
    ("text_encoder", "MiniMaxH3TextEncoderStep"),
    ("prepare_layout", "MiniMaxH3PrepareLayoutStep"),
    ("prepare_latents", "MiniMaxH3PrepareLatentsStep"),
    ("set_timesteps", "MiniMaxH3SetTimestepsStep"),
    ("denoise", "MiniMaxH3DenoiseStep"),
    ("decode.video", "MiniMaxH3VideoDecodeStep"),
    ("decode.audio", "MiniMaxH3AudioDecodeStep"),
]
# A keyframe adds the one block that encodes it; whether it anchors the first or the last frame is a matter of the
# packed layout, not of which blocks run.
FL2VA_WORKFLOW = [
    *T2VA_WORKFLOW[:2],
    ("vae_encoder", "MiniMaxH3KeyframeVaeEncoderStep"),
    *T2VA_WORKFLOW[2:],
]
MINIMAX_H3_WORKFLOWS = {
    "t2va": T2VA_WORKFLOW,
    "fl2va": FL2VA_WORKFLOW,
    "fl2va_last_frame": FL2VA_WORKFLOW,
}
MINIMAX_H3_REF2VA_WORKFLOWS = {
    "ref2va": [
        ("setup", "MiniMaxH3Ref2VASetupStep"),
        ("text_encoder", "MiniMaxH3Ref2VATextEncoderStep"),
        ("reference_encoder", "MiniMaxH3Ref2VAReferenceEncoderStep"),
        ("prepare_layout", "MiniMaxH3Ref2VAPrepareLayoutStep"),
        ("prepare_latents", "MiniMaxH3PrepareLatentsStep"),
        ("set_timesteps", "MiniMaxH3SetTimestepsStep"),
        ("denoise", "MiniMaxH3Ref2VADenoiseStep"),
        ("decode.video", "MiniMaxH3VideoDecodeStep"),
        ("decode.audio", "MiniMaxH3AudioDecodeStep"),
    ]
}


@pytest.fixture(autouse=True, scope="module")
def small_references():
    """
    Encode references at `TEST_REFERENCE_IMAGE_SHORT_EDGE` instead of the released 2048 pixels.

    A 2048 pixel short edge packs thousands of conditioning rows, which is minutes per pipeline call on CPU. Every
    request of this module passes `height` and `width`, so the canvas of a generated video is never derived from
    these.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(packing_ref2va, "MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE", TEST_REFERENCE_IMAGE_SHORT_EDGE)
        patch.setattr(packing, "MINIMAX_H3_SHORT_EDGE", TEST_REFERENCE_IMAGE_SHORT_EDGE)
        patch.setattr(packing, "MINIMAX_H3_MAX_PIXELS", TEST_REFERENCE_IMAGE_SHORT_EDGE**2 * 2)
        yield


class MiniMaxH3ModularTesterBase(ModularPipelineTesterMixin):
    """What the two blocksets share: the repository, the output surface and the tests that do not apply."""

    pretrained_model_name_or_path = TINY_MODULAR_REPO_ID
    # MiniMax-H3 packs one request into one sequence and rejects a list of prompts, so nothing is batched.
    batch_params = frozenset()
    # The checkpoint is guidance-distilled and generates a video, so neither `guidance_scale` nor
    # `num_images_per_prompt` exists.
    optional_params = frozenset(["num_inference_steps", "latents", "output_type"])
    output_name = "videos"

    @pytest.mark.skip(reason="MiniMax-H3 packs one request into one sequence, so a batch of prompts is not a thing.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="MiniMax-H3 packs one request into one sequence, so a batch of prompts is not a thing.")
    def test_inference_batch_single_identical(self):
        pass

    @pytest.mark.skip(reason="One request generates one video, so there is no `num_images_per_prompt`.")
    def test_num_images_per_prompt(self):
        pass

    def test_load_expected_components_from_pretrained(self, tmp_path):
        r"""
        Every component the repository holds *for this half* is loaded from it.

        The repository holds both checkpoint partitions, as the released one does, so its index also names the
        transformer of the other half — which these blocks never declare and must therefore never load.
        """
        pipe = self.get_pipeline()
        specified = _get_specified_components(self.pretrained_model_name_or_path, cache_dir=tmp_path)

        expected = {name for name in specified if name in pipe.components}
        actual = {
            name
            for name in pipe.components
            if getattr(pipe, name, None) is not None
            and getattr(getattr(pipe, name), "_diffusers_load_id", None) not in (None, "null")
        }

        assert expected == actual, f"Component mismatch: missing={expected - actual}, unexpected={actual - expected}"
        assert "transformer_ref" in specified and "transformer" in specified


class TestMiniMaxH3ModularPipelineFast(MiniMaxH3ModularTesterBase):
    pipeline_class = MiniMaxH3ModularPipeline
    pipeline_blocks_class = MiniMaxH3Blocks

    params = frozenset(["prompt", "image", "last_image", "height", "width", "num_frames"])
    expected_workflow_blocks = MINIMAX_H3_WORKFLOWS

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a robot dancing",
            "generator": self.get_generator(seed),
            "num_inference_steps": 2,
            "height": RESOLUTION,
            "width": RESOLUTION,
            "num_frames": NUM_FRAMES,
            "output_type": "pt",
        }

    def test_video_and_audio_outputs(self):
        r"""One call denoises both modalities out of the one packed sequence, and returns both."""
        pipe = self.get_pipeline()

        state = pipe(**self.get_dummy_inputs())
        video, audio = state.get("videos"), state.get("audio")

        assert video.shape == (1, NUM_FRAMES, 3, RESOLUTION, RESOLUTION)
        # The audio VAE is mono and takes the two stereo channels as two batch items, which the decoder block stacks
        # back into `(batch, 2, samples)`.
        assert audio.shape == (1, 2, NUM_AUDIO_LATENTS * AUDIO_HOP_LENGTH)
        assert video.min() >= 0.0 and video.max() <= 1.0
        assert state.get("sampling_rate") == pipe.audio_vae.config.sampling_rate

    def test_fl2va_keyframe(self):
        r"""A keyframe goes through the `fl2va` workflow, which encodes it into conditioning rows."""
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["image"] = Image.new("RGB", (48, 80))
        video = pipe(**inputs, output="videos")

        assert video.shape == (1, NUM_FRAMES, 3, RESOLUTION, RESOLUTION)

    def test_generator_reproducibility(self):
        r"""
        Two runs from the same generator state are identical, two seeds differ.

        The blocks draw from the one generator the request carries, in the order they run — the keyframe conditioning
        noise in the VAE encoder step, then the video and the audio noise in the prepare-latents step — which is what
        keeps a modular run reproducible, and equal to the standard pipeline's.
        """
        pipe = self.get_pipeline()

        def run(seed):
            inputs = self.get_dummy_inputs()
            inputs["generator"] = torch.Generator("cpu").manual_seed(seed)
            inputs["image"] = Image.new("RGB", (48, 80))
            state = pipe(**inputs)
            return state.get("videos"), state.get("audio")

        video, audio = run(7)
        same_video, same_audio = run(7)
        other_video, other_audio = run(8)

        assert torch.equal(video, same_video)
        assert torch.equal(audio, same_audio)
        assert not torch.equal(video, other_video)
        assert not torch.equal(audio, other_audio)

    def test_text_encoder_block_standalone(self):
        r"""
        The text encoder block runs on its own, without the denoiser or the VAEs.

        Encoding a presentation once and reusing it across requests is the modular way of passing `prompt_embeds`,
        so the block may only touch the components it declares.
        """
        pipe = MiniMaxH3TextEncoderStep().init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)

        outputs = pipe(prompt="a robot dancing", output=["prompt_embeds", "text_token_tags"])

        assert set(pipe.components) == {"text_encoder", "tokenizer", "processor"}
        assert outputs["prompt_embeds"].shape[0] == 1
        assert outputs["prompt_embeds"].shape[1] == outputs["text_token_tags"].shape[0]
        # `1` tags a text row, and a text-only presentation is the verbatim prompt.
        assert set(outputs["text_token_tags"].tolist()) == {1}


class TestMiniMaxH3Ref2VAModularPipelineFast(MiniMaxH3ModularTesterBase):
    pipeline_class = MiniMaxH3Ref2VAModularPipeline
    pipeline_blocks_class = MiniMaxH3Ref2VABlocks

    params = frozenset(["prompt", "references", "height", "width", "num_frames"])
    expected_workflow_blocks = MINIMAX_H3_REF2VA_WORKFLOWS

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a robot dancing",
            "references": [MiniMaxH3Reference(image=Image.new("RGB", (48, 80)))],
            "generator": self.get_generator(seed),
            "num_inference_steps": 2,
            "height": RESOLUTION,
            "width": RESOLUTION,
            "num_frames": NUM_FRAMES,
            "output_type": "pt",
        }

    def test_video_and_audio_outputs(self):
        r"""A reference conditions the request without binding the generated geometry."""
        pipe = self.get_pipeline()

        state = pipe(**self.get_dummy_inputs())
        video, audio = state.get("videos"), state.get("audio")

        assert video.shape == (1, NUM_FRAMES, 3, RESOLUTION, RESOLUTION)
        assert audio.shape == (1, 2, NUM_AUDIO_LATENTS * AUDIO_HOP_LENGTH)
        assert video.min() >= 0.0 and video.max() <= 1.0

    def test_audio_reference_alone_is_rejected(self):
        r"""An audio reference is conditioning for a subject, so it cannot be the only reference of a request."""
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        sample_rate = pipe.audio_vae.config.sampling_rate
        inputs["references"] = [MiniMaxH3Reference(audio=torch.zeros(2, 6 * sample_rate), sample_rate=sample_rate)]

        with pytest.raises(ValueError, match="has to be paired with at least one image or video reference"):
            pipe(**inputs)
