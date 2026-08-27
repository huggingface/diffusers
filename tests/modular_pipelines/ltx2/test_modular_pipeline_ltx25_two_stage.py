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

from diffusers.modular_pipelines import LTX25TwoStageBlocks, LTX25TwoStageModularPipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


LTX25_REPO_ID = "hf-internal-testing/tiny-ltx2-5-modular-pipe"

LTX25_TWO_STAGE_WORKFLOWS = {
    "text2video": [
        ("text_encoder.text_encoder", "LTX2TextEncoderStep"),
        ("text_encoder.connectors", "LTX2TextConnectorStep"),
        ("duration", "LTX2DurationStep"),
        ("input", "LTX2TextInputStep"),
        ("stage_1.set_timesteps", "LTX2SetTimestepsStep"),
        ("stage_1.prepare_latents", "LTX2PrepareLatentsStep"),
        ("stage_1.prepare_audio_latents", "LTX2PrepareAudioLatentsStep"),
        ("stage_1.prepare_coords", "LTX2PrepareCoordsStep"),
        ("stage_1.denoise", "LTX2DenoiseStep"),
        ("stage_1.unpack", "LTX2UnpackLatentsStep"),
        ("upsample.latent_upsample", "LTX2LatentUpsampleStep"),
        ("stage_2.prepare_latents", "LTX2Stage2PrepareLatentsStep"),
        ("stage_2.set_timesteps", "LTX2SetTimestepsStep"),
        ("stage_2.prepare_audio_latents", "LTX2Stage2PrepareAudioLatentsStep"),
        ("stage_2.prepare_coords", "LTX2PrepareCoordsStep"),
        ("stage_2.denoise", "LTX2DenoiseStep"),
        ("stage_2.unpack", "LTX2UnpackLatentsStep"),
        ("decode.video_decode", "LTX2DiffusionVaeDecoderStep"),
        ("decode.audio_decode", "LTX2AudioDecoderStep"),
    ],
}


class LTX25TwoStageModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = LTX25TwoStageModularPipeline
    pipeline_blocks_class = LTX25TwoStageBlocks
    pretrained_model_name_or_path = LTX25_REPO_ID
    params = frozenset(["prompt", "height", "width", "num_frames"])
    batch_params = frozenset(["prompt"])
    optional_params = frozenset(["num_videos_per_prompt"])
    # Each pass runs its own fixed sigma schedule, so there is no step count to set; and the first pass always
    # starts from noise, so there are no pre-generated `latents` to pass -- the upsample step produces them.
    not_params = frozenset(["num_inference_steps", "latents"])
    expected_workflow_blocks = LTX25_TWO_STAGE_WORKFLOWS
    output_name = "videos"

    def get_pipeline(self, components_manager=None, dtype=torch.float32):
        # The LTX-2.5 fixture ships no `latent_upsampler`, so the stage bridge is a seeded tiny one built here. Its
        # weights are fixed by the seed, which keeps outputs comparable across the pipelines a test class builds.
        pipe = super().get_pipeline(components_manager=components_manager, dtype=dtype)
        torch.manual_seed(0)
        latent_upsampler = LTX2LatentUpsamplerModel(
            in_channels=pipe.transformer.config.in_channels, mid_channels=32, num_blocks_per_stage=1
        )
        pipe.update_components(latent_upsampler=latent_upsampler.to(dtype))
        return pipe

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a robot dancing",
            "negative_prompt": "",
            "generator": self.get_generator(seed),
            "sigmas": [1.0, 0.5],
            "stage_2_sigmas": [0.5, 0.25],
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "frame_rate": 25.0,
            "max_sequence_length": 16,
            "output_type": "pt",
        }


class TestLTX25TwoStageModularPipelineFast(LTX25TwoStageModularPipelineTesterConfig, ModularPipelineTesterMixin):
    @pytest.mark.skip(reason="num_videos_per_prompt")
    def test_num_images_per_prompt(self):
        pass

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=1e-3)

    def test_output_is_twice_the_requested_resolution(self):
        # As with the standard pipelines, `height` / `width` are the first pass's resolution and the upsample
        # doubles them.
        pipe = self.get_pipeline().to("cpu")

        output = pipe(**self.get_dummy_inputs(), output=["videos", "audio"])
        videos, audio = output["videos"], output["audio"]

        assert videos.shape == (1, 5, 3, 64, 64)
        assert audio.shape[0] == 1
        assert torch.isnan(audio).sum() == 0

    def test_first_pass_decodes_on_its_own(self):
        # Popping `upsample` and `stage_2` leaves `stage_1 -> decode`: a preview of the first pass at the requested
        # resolution, since the pass leaves packed latents in state like any other core denoise.
        pipe = self.get_pipeline().to("cpu")
        blocks = LTX25TwoStageBlocks()
        blocks.sub_blocks.pop("upsample")
        blocks.sub_blocks.pop("stage_2")
        preview_pipe = blocks.init_pipeline(LTX25_REPO_ID)
        preview_pipe.update_components(**{name: getattr(pipe, name) for name in pipe.pretrained_component_names})

        videos = preview_pipe(**self.get_dummy_inputs(), output="videos")

        assert videos.shape == (1, 5, 3, 32, 32)

    def test_stage_2_follows_the_workflow(self):
        # `stage_2` picks the workflow's second-pass group, and the image / frame conditions are re-encoded at the
        # upsampled resolution ahead of it, so an image-to-video or condition run refines under its conditioning.
        blocks = LTX25TwoStageBlocks()

        def selected(**inputs):
            names = {
                name: type(block).__name__ for name, block in blocks.get_execution_blocks(**inputs).sub_blocks.items()
            }
            return (
                names["stage_2.prepare_latents"],
                "stage_2.prepare_i2v_latents" in names,
                "stage_2_vae_encoder" in names,
                "stage_2_condition_encoder" in names,
            )

        assert selected(prompt=True) == ("LTX2Stage2PrepareLatentsStep", False, False, False)
        assert selected(prompt=True, image=True, image_latents=True) == (
            "LTX2Stage2PrepareLatentsStep",
            True,
            True,
            False,
        )
        assert selected(prompt=True, conditions=True, condition_latents=True) == (
            "LTX2ConditionStage2PrepareLatentsStep",
            False,
            False,
            True,
        )

    def test_split_stages_match_the_single_call(self):
        # The point of the blockset: `stage_1` / `upsample` / `stage_2` are each usable as their own pipeline, and
        # chaining them by hand -- one generator threaded through, as in the standard two-stage recipe -- is the
        # same computation as the one call.
        pipe = self.get_pipeline().to("cpu")
        reference = pipe(**self.get_dummy_inputs(), output="videos")

        blocks = LTX25TwoStageBlocks()
        stage_2 = blocks.sub_blocks.pop("stage_2")
        upsample = blocks.sub_blocks.pop("upsample")
        decode = blocks.sub_blocks.pop("decode")
        components = {name: getattr(pipe, name) for name in pipe.pretrained_component_names}
        # The stage blocks declare no autoencoder, so a standalone stage pipeline resolves the latent geometry and
        # statistics to the `LTX2ModularPipeline` fallbacks -- the production values, which the tiny fixture VAEs do
        # not have. Pin the fixture's values on those pipelines; a real run has the checkpoint's.
        vae_values = {
            name: getattr(pipe, name)
            for name in (
                "vae_spatial_compression_ratio",
                "vae_temporal_compression_ratio",
                "vae_scaling_factor",
                "latents_mean",
                "latents_std",
                "audio_latent_channels",
                "audio_latent_mel_bins",
                "audio_vae_mel_compression_ratio",
                "audio_vae_temporal_compression_ratio",
                "audio_sampling_rate",
                "audio_hop_length",
                "audio_latents_mean",
                "audio_latents_std",
            )
        }
        pipes = []
        for stage_blocks in (blocks, upsample, stage_2, decode):
            stage_pipe = stage_blocks.init_pipeline(LTX25_REPO_ID)
            stage_pipe.update_components(**{k: v for k, v in components.items() if k in stage_pipe.components})
            if "diffusion_decoder" not in stage_pipe.components:
                stage_pipe.__class__ = type(
                    type(stage_pipe).__name__,
                    (type(stage_pipe),),
                    {name: property(lambda self, value=value: value) for name, value in vae_values.items()},
                )
            pipes.append(stage_pipe)
        stage_1_pipe, upsample_pipe, stage_2_pipe, decode_pipe = pipes

        inputs = self.get_dummy_inputs()

        def carry(to_pipe, *from_states):
            # Hand a downstream pipeline whatever it declares as an input, from the upstream states (later wins).
            carried = {}
            for from_state in from_states:
                carried.update(
                    {
                        name: from_state.get(name)
                        for name in to_pipe.blocks.input_names
                        if from_state.get(name) is not None
                    }
                )
            return carried

        stage_1_state = stage_1_pipe(**inputs)
        upsample_state = upsample_pipe(**carry(upsample_pipe, stage_1_state))
        stage_2_inputs = carry(stage_2_pipe, stage_1_state, upsample_state)
        stage_2_inputs.update(stage_2_sigmas=inputs["stage_2_sigmas"], generator=inputs["generator"])
        stage_2_state = stage_2_pipe(**stage_2_inputs)
        decode_inputs = carry(decode_pipe, stage_2_state)
        decode_inputs.update(generator=inputs["generator"], output_type=inputs["output_type"])
        videos = decode_pipe(**decode_inputs, output="videos")

        assert torch.allclose(videos, reference, atol=1e-4)


class TestLTX25TwoStageModularPipelineLoading(LTX25TwoStageModularPipelineTesterConfig, ModularLoadingTesterMixin):
    @pytest.mark.skip(reason="the fixture ships no `latent_upsampler`, so a reloaded pipeline cannot run")
    def test_save_from_pretrained(self):
        pass


class TestLTX25TwoStageModularPipelineWorkflow(LTX25TwoStageModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    # `ModularPipeline.from_pretrained(repo, workflow=...)` resolves the blocks from the repo's
    # `modular_model_index.json`, which for the LTX-2.5 fixture is `LTX25AutoBlocks`. These two tests need a fixture
    # whose index names `LTX25TwoStageBlocks` and ships a `latent_upsampler`.
    @pytest.mark.skip(reason="the fixture routes to `LTX25AutoBlocks`")
    def test_from_pretrained_workflow(self):
        pass

    @pytest.mark.skip(reason="the fixture routes to `LTX25AutoBlocks`")
    def test_load_components_workflow(self):
        pass
