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

from diffusers import ClassifierFreeGuidance
from diffusers.modular_pipelines import (
    KandinskyS2VAutoBlocks,
    KandinskyS2VDistilledAutoBlocks,
    KandinskyS2VDistilledModularPipeline,
    KandinskyS2VModularPipeline,
)
from diffusers.modular_pipelines.kandinsky_s2v.before_denoise import KandinskyS2VPrepareStep

from ...testing_utils import numpy_cosine_similarity_distance, require_accelerator, torch_device
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


def tone(seconds: int) -> list[int]:
    """A plain tone as 16-bit samples."""
    return [int(8000 * math.sin(i / 24)) for i in range(seconds * AUDIO_SAMPLE_RATE)]


@functools.lru_cache(maxsize=2)
def dummy_audio(seconds: int = 1) -> str:
    """The tone as a wav file."""
    path = Path(tempfile.mkdtemp()) / "speech.wav"
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(AUDIO_SAMPLE_RATE)
        handle.writeframes(b"".join(struct.pack("<h", sample) for sample in tone(seconds)))
    return str(path)


def dummy_waveform(seconds: int = 1) -> torch.Tensor:
    """The same tone as a mono waveform, scaled as an audio loader reads 16-bit samples."""
    return torch.tensor(tone(seconds), dtype=torch.float32) / 32768


def check_float16_inference(tester, expected_max_diff=5e-2):
    """The mixin's fp16 check, with the pipeline loaded in fp16 rather than cast to it after loading.

    Loading in a dtype keeps `_keep_in_fp32_modules` in fp32; `.to(torch.float16)` would cast them too.
    """
    outputs = []
    for dtype in (torch.float32, torch.float16):
        pipe = tester.get_pipeline(dtype=dtype).to(torch_device)
        outputs.append(pipe(**tester.get_dummy_inputs(), output=tester.output_name).float().cpu())
    max_diff = numpy_cosine_similarity_distance(outputs[0].flatten().numpy(), outputs[1].flatten().numpy())
    assert max_diff < expected_max_diff, f"FP16 inference is different from FP32 inference (max_diff: {max_diff})"


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

    @require_accelerator
    def test_float16_inference(self):
        check_float16_inference(self)

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

    @pytest.mark.parametrize("seconds", [1, 2])
    def test_audio_as_a_waveform(self, seconds):
        """A mono waveform gives the same video as a file holding the same samples, padded or cut to the chunks."""
        pipe = self.get_pipeline()
        inputs = {**self.get_dummy_inputs(), "num_chunks": 1}
        from_file = pipe(**{**inputs, "audio": dummy_audio(seconds)}, output="videos")
        inputs = {**self.get_dummy_inputs(), "num_chunks": 1}
        from_waveform = pipe(**{**inputs, "audio": dummy_waveform(seconds)}, output="videos")
        assert torch.equal(from_file, from_waveform)

    @pytest.mark.parametrize(
        "waveform",
        [
            dummy_waveform().expand(2, -1),
            torch.zeros(0),
            (dummy_waveform() * 32768).to(torch.int16),
        ],
        ids=["multichannel", "empty", "int16"],
    )
    def test_rejects_a_bad_waveform(self, waveform):
        pipe = self.get_pipeline()
        with pytest.raises(ValueError, match="mono float waveform"):
            pipe(**{**self.get_dummy_inputs(), "audio": waveform}, output="videos")

    def test_rejects_a_single_step(self):
        """One step leaves an empty Euler schedule, which would decode the initial noise."""
        pipe = self.get_pipeline()
        with pytest.raises(ValueError, match="at least 2"):
            pipe(**{**self.get_dummy_inputs(), "num_inference_steps": 1}, output="videos")

    def test_rejects_negative_embeds_without_pooled(self):
        prepare_pipe = KandinskyS2VPrepareStep().init_pipeline(self.pretrained_model_name_or_path)
        prepare_pipe.load_components()
        with pytest.raises(ValueError, match="must be passed together"):
            prepare_pipe(
                reference_latents=torch.randn(1, 8, 8, 4),
                audio_embeds=torch.randn(16, 16),
                num_chunks=1,
                prompt_embeds=torch.randn(3, 16),
                pooled_prompt_embeds=torch.randn(16),
                negative_prompt_embeds=torch.randn(3, 16),
            )


class TestKandinskyS2VModularPipelineLoading(KandinskyS2VModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestKandinskyS2VModularPipelineMemory(KandinskyS2VModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


FAST_SKIPS = {
    "test_inference_batch_consistent": "The pipeline takes one prompt at a time; batching is not supported",
    "test_inference_batch_single_identical": "The pipeline takes one prompt at a time; batching is not supported",
    "test_num_images_per_prompt": "One video per call; there is no num_images_per_prompt",
}


class KandinskyS2VDistilledModularPipelineTesterConfig(KandinskyS2VModularPipelineTesterConfig):
    """The pi-Flow checkpoints: a head widened to its grid, `PiflowScheduler`, guidance off."""

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

    @require_accelerator
    def test_float16_inference(self):
        check_float16_inference(self)

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
