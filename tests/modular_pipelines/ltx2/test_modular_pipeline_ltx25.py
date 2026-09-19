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

from diffusers.modular_pipelines import LTX25AutoBlocks, LTX25ModularPipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition

from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


LTX25_REPO_ID = "hf-internal-testing/tiny-ltx2-5-modular-pipe"

LTX25_WORKFLOWS = {
    "text2video": [
        ("text_encoder.text_encoder", "LTX2TextEncoderStep"),
        ("text_encoder.connectors", "LTX2TextConnectorStep"),
        ("duration", "LTX2DurationStep"),
        ("input", "LTX2TextInputStep"),
        ("denoise.set_timesteps", "LTX2SetTimestepsStep"),
        ("denoise.prepare_latents", "LTX2PrepareLatentsStep"),
        ("denoise.prepare_audio_latents", "LTX2PrepareAudioLatentsStep"),
        ("denoise.prepare_coords", "LTX2PrepareCoordsStep"),
        ("denoise.denoise", "LTX2DenoiseStep"),
        ("denoise.unpack", "LTX2UnpackLatentsStep"),
        ("decode.video_decode", "LTX2DiffusionVaeDecoderStep"),
        ("decode.audio_decode", "LTX2AudioDecoderStep"),
    ],
    "image2video": [
        ("text_encoder.text_encoder", "LTX2TextEncoderStep"),
        ("text_encoder.connectors", "LTX2TextConnectorStep"),
        ("duration", "LTX2DurationStep"),
        ("vae_encoder", "LTX2VaeEncoderStep"),
        ("input", "LTX2TextInputStep"),
        ("denoise.set_timesteps", "LTX2SetTimestepsStep"),
        ("denoise.prepare_latents", "LTX2PrepareLatentsStep"),
        ("denoise.prepare_i2v_latents", "LTX2Image2VideoPrepareLatentsStep"),
        ("denoise.prepare_audio_latents", "LTX2PrepareAudioLatentsStep"),
        ("denoise.prepare_coords", "LTX2PrepareCoordsStep"),
        ("denoise.denoise", "LTX2Image2VideoDenoiseStep"),
        ("denoise.unpack", "LTX2UnpackLatentsStep"),
        ("decode.video_decode", "LTX2DiffusionVaeDecoderStep"),
        ("decode.audio_decode", "LTX2AudioDecoderStep"),
    ],
    "condition": [
        ("text_encoder.text_encoder", "LTX2TextEncoderStep"),
        ("text_encoder.connectors", "LTX2TextConnectorStep"),
        ("duration", "LTX2DurationStep"),
        ("condition_encoder", "LTX2ConditionEncoderStep"),
        ("input", "LTX2TextInputStep"),
        ("denoise.prepare_latents", "LTX2ConditionPrepareLatentsStep"),
        ("denoise.set_timesteps", "LTX2ConditionSetTimestepsStep"),
        ("denoise.prepare_audio_latents", "LTX2ConditionPrepareAudioLatentsStep"),
        ("denoise.prepare_coords", "LTX2ConditionPrepareCoordsStep"),
        ("denoise.denoise", "LTX2ConditionDenoiseStep"),
        ("denoise.trim_condition_tokens", "LTX2TrimConditionTokensStep"),
        ("denoise.unpack", "LTX2UnpackLatentsStep"),
        ("decode.video_decode", "LTX2DiffusionVaeDecoderStep"),
        ("decode.audio_decode", "LTX2AudioDecoderStep"),
    ],
    "in_context": [
        ("text_encoder.text_encoder", "LTX2TextEncoderStep"),
        ("text_encoder.connectors", "LTX2TextConnectorStep"),
        ("condition_encoder", "LTX2ConditionEncoderStep"),
        ("reference_encoder", "LTX2ReferenceEncoderStep"),
        ("input", "LTX2TextInputStep"),
        ("denoise.prepare_latents", "LTX2InContextPrepareLatentsStep"),
        ("denoise.set_timesteps", "LTX2ConditionSetTimestepsStep"),
        ("denoise.prepare_audio_latents", "LTX2ConditionPrepareAudioLatentsStep"),
        ("denoise.prepare_coords", "LTX2ConditionPrepareCoordsStep"),
        ("denoise.denoise", "LTX2ConditionDenoiseStep"),
        ("denoise.trim_condition_tokens", "LTX2TrimConditionTokensStep"),
        ("denoise.unpack", "LTX2UnpackLatentsStep"),
        ("decode.video_decode", "LTX2DiffusionVaeDecoderStep"),
        ("decode.audio_decode", "LTX2AudioDecoderStep"),
    ],
}


class LTX25ModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    """Shared configuration for every LTX-2.5 workflow; a variant config adds its own `params` and dummy inputs."""

    pipeline_class = LTX25ModularPipeline
    pipeline_blocks_class = LTX25AutoBlocks
    pretrained_model_name_or_path = LTX25_REPO_ID
    batch_params = frozenset(["prompt"])
    optional_params = frozenset(["num_videos_per_prompt"])
    expected_workflow_blocks = LTX25_WORKFLOWS
    output_name = "videos"

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a robot dancing",
            "negative_prompt": "",
            "generator": self.get_generator(seed),
            "sigmas": [1.0, 0.5],
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "frame_rate": 25.0,
            "max_sequence_length": 16,
            "output_type": "pt",
        }


class LTX25ModularPipelineFastTesterMixin(ModularPipelineTesterMixin):
    """`ModularPipelineTesterMixin` with the two adjustments every LTX-2.5 workflow needs."""

    @pytest.mark.skip(reason="num_videos_per_prompt")
    def test_num_images_per_prompt(self):
        pass

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=1e-3)


class LTX25Text2VideoModularPipelineTesterConfig(LTX25ModularPipelineTesterConfig):
    params = frozenset(["prompt", "height", "width", "num_frames"])


class TestLTX25Text2VideoModularPipelineFast(
    LTX25Text2VideoModularPipelineTesterConfig, LTX25ModularPipelineFastTesterMixin
):
    def test_diffusion_decoder_output(self):
        pipe = self.get_pipeline().to("cpu")

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs, output=["videos", "audio"])
        videos, audio = output["videos"], output["audio"]

        assert videos.shape == (1, 5, 3, 32, 32)
        assert audio.shape[0] == 1
        assert audio.shape[1] == pipe.vocoder.config.out_channels
        assert torch.isnan(audio).sum() == 0

    def test_auto_duration_predicts_a_grid_valid_frame_count(self):
        pipe = self.get_pipeline().to("cpu")

        inputs = self.get_dummy_inputs()
        inputs.pop("num_frames")
        inputs["min_seconds"] = 0.5
        inputs["max_seconds"] = 2.0
        videos = pipe(**inputs, output="videos")

        num_frames = videos.shape[1]
        assert (num_frames - 1) % pipe.vae_temporal_compression_ratio == 0
        assert 0 < num_frames <= round(2.0 * inputs["frame_rate"])


class TestLTX25Text2VideoModularPipelineLoading(LTX25Text2VideoModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestLTX25Text2VideoModularPipelineMemory(LTX25Text2VideoModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


# The four workflows share `LTX25AutoBlocks` and the same repo, so one workflow test class covers all of them.
class TestLTX25ModularPipelineWorkflow(LTX25Text2VideoModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class LTX25Image2VideoModularPipelineTesterConfig(LTX25ModularPipelineTesterConfig):
    params = frozenset(["prompt", "image", "height", "width", "num_frames"])

    def get_dummy_inputs(self, seed=0):
        inputs = super().get_dummy_inputs(seed)
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        inputs["image"] = PIL.Image.fromarray(image)
        # Skip H.264 CRF re-compression so the test does not depend on PyAV.
        inputs["image_crf"] = 0
        return inputs


class TestLTX25Image2VideoModularPipelineFast(
    LTX25Image2VideoModularPipelineTesterConfig, LTX25ModularPipelineFastTesterMixin
):
    pass


class TestLTX25Image2VideoModularPipelineLoading(
    LTX25Image2VideoModularPipelineTesterConfig, ModularLoadingTesterMixin
):
    pass


class TestLTX25Image2VideoModularPipelineMemory(LTX25Image2VideoModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


class LTX25ConditionModularPipelineTesterConfig(LTX25ModularPipelineTesterConfig):
    params = frozenset(["prompt", "conditions", "height", "width", "num_frames"])

    def get_dummy_inputs(self, seed=0):
        inputs = super().get_dummy_inputs(seed)
        image = torch.rand((1, 3, 32, 32), generator=torch.Generator("cpu").manual_seed(seed))
        # Synthetic float tensors skip H.264 CRF re-compression (training path uses PIL/uint8).
        inputs["conditions"] = LTX2VideoCondition(frames=image, index=0, strength=1.0, crf=0)
        return inputs


class TestLTX25ConditionModularPipelineFast(
    LTX25ConditionModularPipelineTesterConfig, LTX25ModularPipelineFastTesterMixin
):
    pass


class TestLTX25ConditionModularPipelineLoading(LTX25ConditionModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestLTX25ConditionModularPipelineMemory(LTX25ConditionModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


class LTX25InContextModularPipelineTesterConfig(LTX25ModularPipelineTesterConfig):
    params = frozenset(["prompt", "reference_conditions", "height", "width", "num_frames"])

    def get_dummy_inputs(self, seed=0):
        inputs = super().get_dummy_inputs(seed)
        video = torch.rand((1, 5, 3, 32, 32), generator=torch.Generator("cpu").manual_seed(seed))
        inputs["reference_conditions"] = LTX2ReferenceCondition(frames=video, strength=1.0)
        return inputs


class TestLTX25InContextModularPipelineFast(
    LTX25InContextModularPipelineTesterConfig, LTX25ModularPipelineFastTesterMixin
):
    pass


class TestLTX25InContextModularPipelineLoading(LTX25InContextModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestLTX25InContextModularPipelineMemory(LTX25InContextModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass
