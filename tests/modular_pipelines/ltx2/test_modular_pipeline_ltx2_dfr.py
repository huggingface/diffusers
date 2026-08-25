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

import pytest
import torch
import torch.nn.functional as F

from diffusers.modular_pipelines import LTX2DFRBlocks, LTX2DFRModularPipeline
from diffusers.modular_pipelines.ltx2.utils import resolve_canvas
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


# Differs from `hf-internal-testing/tiny-ltx2-5-modular-pipe` in one place: its transformer sets
# `use_keyframes_abs_pos_embedding`, which DFR requires and the LTX-2.5 fixture leaves off.
LTX2DFR_REPO_ID = "hf-internal-testing/tiny-ltx2-5-dfr-modular-pipe"

LTX2DFR_WORKFLOWS = {
    "text2video": [
        ("text_encoder.text_encoder", "LTX2TextEncoderStep"),
        ("text_encoder.connectors", "LTX2TextConnectorStep"),
        ("duration", "LTX2DurationStep"),
        ("plan", "LTX2DFRPlanStep"),
        ("denoise.input", "LTX2TextInputStep"),
        ("denoise.prepare_latents", "LTX2DFRPrepareLatentsStep"),
        ("denoise.set_timesteps", "LTX2ConditionSetTimestepsStep"),
        ("denoise.prepare_audio_latents", "LTX2ConditionPrepareAudioLatentsStep"),
        ("denoise.prepare_coords", "LTX2ConditionPrepareCoordsStep"),
        ("denoise.denoise", "LTX2ConditionDenoiseStep"),
        ("decode.split_keyframes", "LTX2DFRSplitKeyframesStep"),
        ("decode.video_decode", "LTX2DiffusionVaeDecoderStep"),
        ("decode.audio_decode", "LTX2AudioDecoderStep"),
    ],
    "condition": [
        ("text_encoder.text_encoder", "LTX2TextEncoderStep"),
        ("text_encoder.connectors", "LTX2TextConnectorStep"),
        ("duration", "LTX2DurationStep"),
        ("plan", "LTX2DFRPlanStep"),
        ("condition_encoder", "LTX2ConditionEncoderStep"),
        ("denoise.input", "LTX2TextInputStep"),
        ("denoise.prepare_latents", "LTX2DFRPrepareLatentsStep"),
        ("denoise.set_timesteps", "LTX2ConditionSetTimestepsStep"),
        ("denoise.prepare_audio_latents", "LTX2ConditionPrepareAudioLatentsStep"),
        ("denoise.prepare_coords", "LTX2ConditionPrepareCoordsStep"),
        ("denoise.denoise", "LTX2ConditionDenoiseStep"),
        ("decode.split_keyframes", "LTX2DFRSplitKeyframesStep"),
        ("decode.video_decode", "LTX2DiffusionVaeDecoderStep"),
        ("decode.audio_decode", "LTX2AudioDecoderStep"),
    ],
}

# The fixture VAE compresses time by 2, so the segment grid offers 6 and 8 pixel frames. 13 frames land on the
# 6-frame grid unpadded and buy two keyframe slots, at pixel frames 6 and 12.
DUMMY_NUM_FRAMES = 13
DUMMY_NUM_SLOTS = 2


class LTX2DFRModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    """Shared configuration for the DFR workflows; a variant config adds its own `params` and dummy inputs."""

    pipeline_class = LTX2DFRModularPipeline
    pipeline_blocks_class = LTX2DFRBlocks
    pretrained_model_name_or_path = LTX2DFR_REPO_ID
    batch_params = frozenset(["prompt"])
    optional_params = frozenset(["num_inference_steps", "num_videos_per_prompt", "latents"])
    expected_workflow_blocks = LTX2DFR_WORKFLOWS
    output_name = "videos"

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a robot dancing",
            "negative_prompt": "",
            "generator": self.get_generator(seed),
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "num_frames": DUMMY_NUM_FRAMES,
            "frame_rate": 25.0,
            "max_sequence_length": 16,
            "output_type": "pt",
        }


class LTX2DFRModularPipelineFastTesterMixin(ModularPipelineTesterMixin):
    """`ModularPipelineTesterMixin` with the two adjustments every LTX-2 workflow needs."""

    @pytest.mark.skip(reason="num_videos_per_prompt")
    def test_num_images_per_prompt(self):
        pass

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=1e-3)


class LTX2DFRText2VideoModularPipelineTesterConfig(LTX2DFRModularPipelineTesterConfig):
    params = frozenset(["prompt", "height", "width", "num_frames"])


class TestLTX2DFRText2VideoModularPipelineFast(
    LTX2DFRText2VideoModularPipelineTesterConfig, LTX2DFRModularPipelineFastTesterMixin
):
    def test_generates_a_keyframe_slot_per_segment_border(self):
        pipe = self.get_pipeline().to("cpu")

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs, output=["videos", "keyframes_latents"])

        assert output["videos"].shape == (1, DUMMY_NUM_FRAMES, 3, 32, 32)
        # One latent frame of content per slot, at the target's latent resolution.
        keyframes = output["keyframes_latents"]
        assert keyframes.shape[2] == DUMMY_NUM_SLOTS
        assert keyframes.shape[3:] == (32 // pipe.vae_spatial_compression_ratio,) * 2
        assert torch.isnan(keyframes).sum() == 0

    def test_canvas_padding_is_trimmed_back_to_the_requested_length(self):
        pipe = self.get_pipeline().to("cpu")
        ratio = pipe.vae_temporal_compression_ratio

        # 11 frames divide neither segment length, so the canvas pads to 13 and is trimmed back.
        requested = 11
        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = requested
        canvas_frames, _, _ = resolve_canvas(requested, ratio)
        assert canvas_frames > requested, "pick a frame count that actually pads, or this asserts nothing"

        videos = pipe(**inputs, output="videos")
        assert videos.shape[1] == requested

    def test_keyframe_marker_reaches_the_transformer(self):
        # `video_keyframes_mask` is plumbed to the denoiser purely by its `denoiser_input_fields` tag, with no
        # DFR-specific denoise block. Zeroing the learned embedding it gates is the behavioural check that the
        # tag actually arrives: if it does not, the two runs are identical.
        pipe = self.get_pipeline().to("cpu")

        with_marker = pipe(**self.get_dummy_inputs(), output="videos")
        with torch.no_grad():
            pipe.transformer.keyframes_abs_pos_embedding.zero_()
        without_marker = pipe(**self.get_dummy_inputs(), output="videos")

        assert not torch.allclose(with_marker, without_marker)

    def test_detailing_pass_consumes_the_first_pass_output(self):
        # The full DFR recipe is two passes of these blocks, with the spatial detailing IC-LoRA loaded in
        # between. Here only the plumbing is exercised: the second pass has to accept the first pass's video
        # latents, its keyframe slots and an in-context reference, and return the same shapes one scale up.
        pipe = self.get_pipeline().to("cpu")

        base_inputs = self.get_dummy_inputs()
        base_inputs["output_type"] = "latent"
        first = pipe(**base_inputs, output=["videos", "keyframes_latents"])
        video_latents, keyframes_latents = first["videos"], first["keyframes_latents"]

        # Stands in for `LTX2LatentUpsamplePipeline`, which is a separate pipeline in the real recipe.
        upsample = lambda latents: F.interpolate(latents, scale_factor=(1, 2, 2), mode="nearest")  # noqa: E731

        detail_inputs = self.get_dummy_inputs()
        detail_inputs.update(
            output_type="latent",
            height=64,
            width=64,
            latents=upsample(video_latents),
            keyframes_latents=upsample(keyframes_latents),
            detailing_reference_latents=video_latents,
            detailing_reference_downscale_factor=2,
            noise_scale=0.4,
        )
        second = pipe(**detail_inputs, output=["videos", "keyframes_latents"])

        assert second["videos"].shape[-2:] == (2 * video_latents.shape[-2], 2 * video_latents.shape[-1])
        assert second["keyframes_latents"].shape[2] == DUMMY_NUM_SLOTS
        assert torch.isnan(second["videos"]).sum() == 0

    def test_auto_duration_lands_on_the_segment_grid(self):
        # The plan step pads a duration-head prediction onto the segment grid, so the two have to agree: the
        # predicted length must itself be a length `resolve_canvas` accepts.
        pipe = self.get_pipeline().to("cpu")

        inputs = self.get_dummy_inputs()
        inputs.pop("num_frames")
        inputs["min_seconds"] = 0.5
        inputs["max_seconds"] = 2.0
        videos = pipe(**inputs, output="videos")

        num_frames = videos.shape[1]
        assert (num_frames - 1) % pipe.vae_temporal_compression_ratio == 0
        assert 0 < num_frames <= round(2.0 * inputs["frame_rate"])

    def test_rejects_a_frame_count_off_the_latent_grid(self):
        pipe = self.get_pipeline().to("cpu")

        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = DUMMY_NUM_FRAMES + 1
        with pytest.raises(ValueError, match="num_frames"):
            pipe(**inputs, output="videos")


class TestLTX2DFRText2VideoModularPipelineLoading(
    LTX2DFRText2VideoModularPipelineTesterConfig, ModularLoadingTesterMixin
):
    pass


class TestLTX2DFRText2VideoModularPipelineMemory(
    LTX2DFRText2VideoModularPipelineTesterConfig, ModularMemoryTesterMixin
):
    pass


# Both workflows share `LTX2DFRBlocks` and the same repo, so one workflow test class covers them.
class TestLTX2DFRModularPipelineWorkflow(LTX2DFRText2VideoModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class LTX2DFRConditionModularPipelineTesterConfig(LTX2DFRModularPipelineTesterConfig):
    params = frozenset(["prompt", "conditions", "height", "width", "num_frames"])

    def get_dummy_inputs(self, seed=0):
        inputs = super().get_dummy_inputs(seed)
        image = torch.rand((1, 3, 32, 32), generator=torch.Generator("cpu").manual_seed(seed))
        # Synthetic float tensors skip H.264 CRF re-compression (training path uses PIL/uint8).
        inputs["conditions"] = LTX2VideoCondition(frames=image, index=0, strength=1.0, crf=0)
        return inputs


class TestLTX2DFRConditionModularPipelineFast(
    LTX2DFRConditionModularPipelineTesterConfig, LTX2DFRModularPipelineFastTesterMixin
):
    pass


class TestLTX2DFRConditionModularPipelineLoading(
    LTX2DFRConditionModularPipelineTesterConfig, ModularLoadingTesterMixin
):
    pass


class TestLTX2DFRConditionModularPipelineMemory(LTX2DFRConditionModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass
