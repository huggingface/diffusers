# coding=utf-8
# Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

import functools
import math
import struct
import tempfile
import wave
from pathlib import Path

import pytest
import torch
from PIL import Image

from diffusers import ClassifierFreeGuidance, ModularPipeline
from diffusers.modular_pipelines import (
    KandinskyS2VAutoBlocks,
    KandinskyS2VDistilledAutoBlocks,
    KandinskyS2VDistilledModularPipeline,
    KandinskyS2VModularPipeline,
)

from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
)


# One tiny repository per variant, laid out like the real checkpoints.
TINY_REPO = "kandinskylab/tiny-kandinsky-s2v-modular-pipe"
TINY_DISTILLED_REPO = "kandinskylab/tiny-kandinsky-s2v-distilled-modular-pipe"
AUDIO_SAMPLE_RATE = 16000
# One chunk of NUM_LATENT_FRAMES covers ((8 - 1) * 4 + 1) / 25 = 1.16 s at 25 fps; the speech is shorter, so the
# video is trimmed back to it. NUM_LATENT_FRAMES has to reach the FramePack memory window, sum(frame_buckets) = 7.
NUM_LATENT_FRAMES = 8


@functools.lru_cache(maxsize=1)
def dummy_audio() -> str:
    """A second of a plain tone; the speech is read from a file, not a tensor."""
    path = Path(tempfile.mkdtemp()) / "speech.wav"
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(AUDIO_SAMPLE_RATE)
        handle.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 24))) for i in range(AUDIO_SAMPLE_RATE)))
    return str(path)


class KandinskyS2VModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = KandinskyS2VModularPipeline
    pipeline_blocks_class = KandinskyS2VAutoBlocks
    pretrained_model_name_or_path = TINY_REPO
    params = frozenset(["image", "audio", "prompt", "height", "width", "num_latent_frames"])
    batch_params = frozenset(["prompt"])
    optional_params = frozenset(["num_inference_steps", "output_type"])
    output_name = "videos"

    def get_dummy_inputs(self, seed=0):
        return {
            "image": Image.new("RGB", (64, 64), color="red"),
            "audio": dummy_audio(),
            # Short prompts: the tiny transformer's text RoPE table is shorter than the default prompt.
            "prompt": "a person speaking",
            "negative_prompt": "static, motionless",
            "height": 64,
            "width": 64,
            "num_latent_frames": NUM_LATENT_FRAMES,
            "num_inference_steps": 2,
            "generator": self.get_generator(seed),
            "output_type": "pt",
        }


class TestKandinskyS2VModularPipelineFast(KandinskyS2VModularPipelineTesterConfig, ModularPipelineTesterMixin):
    @pytest.mark.skip(reason="The pipeline takes one prompt at a time; batching is not supported")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="The pipeline takes one prompt at a time; batching is not supported")
    def test_inference_batch_single_identical(self):
        pass

    @pytest.mark.skip(reason="One video per call; there is no num_images_per_prompt")
    def test_num_images_per_prompt(self):
        pass

    @pytest.mark.skip(reason="The normalisations do their arithmetic in fp32, which casting the whole model bypasses")
    def test_float16_inference(self):
        pass

    def test_two_chunks_carry_memory(self):
        """Two chunks end to end: the second is conditioned on the FramePack memory rolled from the first."""
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["num_chunks"] = 2
        latents = pipe(**{**inputs, "output_type": "latent"}, output="videos")
        assert latents.shape[1] == 2 * NUM_LATENT_FRAMES
        assert torch.isfinite(latents).all()

    def test_rejects_speech_past_the_encoder_limit(self):
        pipe = self.get_pipeline()
        extractor = pipe.tokenizer.feature_extractor
        limit = extractor.n_samples / extractor.sampling_rate
        chunk_seconds = ((NUM_LATENT_FRAMES - 1) * 4 + 1) / 25
        inputs = self.get_dummy_inputs()
        inputs["num_chunks"] = math.ceil(limit / chunk_seconds) + 1
        with pytest.raises(ValueError, match="truncates"):
            pipe(**inputs, output="videos")


class TestKandinskyS2VModularPipelineLoading(KandinskyS2VModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestKandinskyS2VModularPipelineMemory(KandinskyS2VModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


FAST_SKIPS = {
    "test_inference_batch_consistent": "The pipeline takes one prompt at a time; batching is not supported",
    "test_inference_batch_single_identical": "The pipeline takes one prompt at a time; batching is not supported",
    "test_num_images_per_prompt": "One video per call; there is no num_images_per_prompt",
    "test_float16_inference": "The normalisations do their arithmetic in fp32, which casting the whole model bypasses",
}


class KandinskyS2VDistilledModularPipelineTesterConfig(KandinskyS2VModularPipelineTesterConfig):
    """The pi-Flow checkpoints: a head widened to its grid, `KandinskyS2VPiflowScheduler`, guidance off."""

    pipeline_class = KandinskyS2VDistilledModularPipeline
    pipeline_blocks_class = KandinskyS2VDistilledAutoBlocks
    pretrained_model_name_or_path = TINY_DISTILLED_REPO


class TestKandinskyS2VDistilledModularPipelineFast(
    KandinskyS2VDistilledModularPipelineTesterConfig, ModularPipelineTesterMixin
):
    @pytest.mark.skip(reason=FAST_SKIPS["test_inference_batch_consistent"])
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason=FAST_SKIPS["test_inference_batch_single_identical"])
    def test_inference_batch_single_identical(self):
        pass

    @pytest.mark.skip(reason=FAST_SKIPS["test_num_images_per_prompt"])
    def test_num_images_per_prompt(self):
        pass

    @pytest.mark.skip(reason=FAST_SKIPS["test_float16_inference"])
    def test_float16_inference(self):
        pass

    def test_guidance_is_off(self):
        assert self.get_pipeline().guider.guidance_scale == 1.0

    def test_rejects_guidance(self):
        """Guidance on top of the student would over-guide it, quietly and at twice the cost."""
        pipe = self.get_pipeline()
        pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=4.0))
        with pytest.raises(ValueError, match="distilled into the student"):
            pipe(**self.get_dummy_inputs(), output="videos")


class TestKandinskyS2VDistilledModularPipelineLoading(
    KandinskyS2VDistilledModularPipelineTesterConfig, ModularLoadingTesterMixin
):
    pass


class TestKandinskyS2VDistilledModularPipelineMemory(
    KandinskyS2VDistilledModularPipelineTesterConfig, ModularMemoryTesterMixin
):
    pass


@pytest.mark.parametrize(
    ("repo", "pipeline_class", "blocks_class"),
    [
        (TINY_REPO, KandinskyS2VModularPipeline, KandinskyS2VAutoBlocks),
        (TINY_DISTILLED_REPO, KandinskyS2VDistilledModularPipeline, KandinskyS2VDistilledAutoBlocks),
    ],
)
def test_from_pretrained_picks_the_variant(repo, pipeline_class, blocks_class):
    """What a user does: the checkpoint alone decides the pipeline and blocks, base or distilled."""
    pipe = ModularPipeline.from_pretrained(repo)
    assert type(pipe) is pipeline_class
    assert type(pipe.blocks) is blocks_class
