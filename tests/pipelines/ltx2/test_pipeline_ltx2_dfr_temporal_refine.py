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
import PIL.Image
import pytest
import torch

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    LTX2DFRPipeline,
    LTX2DFRTemporalRefinePipeline,
    LTXEulerAncestralRFScheduler,
)
from diffusers.pipelines.ltx2.dfr_core import ANCHOR_KEYFRAME_STRENGTH
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BasePipelineTesterConfig, PipelineTesterMixin
from .dfr_dummies import get_dfr_dummy_components, get_dfr_dummy_inputs, get_temporal_dummy_components


enable_full_determinism()


class LTX2DFRTemporalRefinePipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = LTX2DFRTemporalRefinePipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "num_frames", "frame_rate", "prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    output_shape = (17, 3, 32, 32)
    optional_input_params = BasePipelineTesterConfig.optional_input_params - {
        "num_inference_steps",
        "num_images_per_prompt",
        "latents",
    }

    def get_dummy_components(self):
        return get_temporal_dummy_components()

    def get_dummy_inputs(self):
        generator = torch.Generator("cpu").manual_seed(1)
        return {
            "prompt": "a robot dancing",
            "generator": self.get_generator(0),
            "latents": torch.randn(1, 4, 5, 16, 16, generator=generator),
            "keyframes_latents": torch.randn(1, 4, 1, 16, 16, generator=generator),
            "keyframe_positions": [8],
            "audio_latents": torch.randn(1, 2, 8, 2, generator=generator),
            "height": 32,
            "width": 32,
            "num_frames": 9,
            "frame_rate": 25.0,
            "sigmas": [0.75, 0.25],
            "use_cross_timestep": False,
            "max_sequence_length": 16,
            "output_type": "pt",
        }


class TestLTX2DFRTemporalRefinePipeline(LTX2DFRTemporalRefinePipelineTesterConfig, PipelineTesterMixin):
    @pytest.mark.skip("Temporal refine takes a 5D latent canvas that is not prompt-batched.")
    def test_inference_batch_consistent(self, *args, **kwargs):
        pass

    @pytest.mark.skip("Temporal refine takes a 5D latent canvas that is not prompt-batched.")
    def test_inference_batch_single_identical(self, *args, **kwargs):
        pass

    def test_temporal_upsample_round_doubles_the_frame_count(self):
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        output = pipe(**self.get_dummy_inputs())
        assert output.frames.shape[1] == (9 - 1) * 2 + 1
        assert pipe.num_timesteps == 2

    @pytest.mark.parametrize(
        ("scheduler", "message"),
        [
            (FlowMatchEulerDiscreteScheduler(), "LTXEulerAncestralRFScheduler"),
            (LTXEulerAncestralRFScheduler(eta=0.0), "stochastic step"),
        ],
    )
    def test_a_non_ancestral_scheduler_is_refused(self, scheduler, message):
        # `denoise` falls back to a deterministic Euler step for anything else, so an unchecked scheduler here
        # would run to completion and just return a softer canvas.
        components = self.get_dummy_components()
        components["scheduler"] = scheduler
        pipe = self.get_pipeline(**components).to(torch_device)
        with pytest.raises(ValueError, match=message):
            pipe(**self.get_dummy_inputs())

    def test_temporal_round_tiles_get_distinct_ancestral_noise(self):
        dfr = LTX2DFRPipeline(**get_dfr_dummy_components()).to(torch_device)
        stage = dfr(**get_dfr_dummy_inputs(generator=self.get_generator(0), num_frames=17, output_type="latent"))
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        seeds = []
        original = pipe.denoise

        def capture(**kwargs):
            generator = kwargs.get("generator")
            seeds.append(generator.initial_seed() if generator is not None else None)
            return original(**kwargs)

        pipe.denoise = capture
        try:
            pipe(
                prompt="a robot dancing",
                latents=stage.frames,
                keyframes_latents=stage.keyframes,
                keyframe_positions=stage.keyframe_positions,
                audio_latents=stage.audio,
                height=32,
                width=32,
                num_frames=17,
                frame_rate=25.0,
                sigmas=[0.75, 0.25],
                use_cross_timestep=False,
                max_sequence_length=16,
                output_type="latent",
                generator=self.get_generator(0),
            )
        finally:
            del pipe.denoise

        assert seeds == [1000 * 1 + 0, 1000 * 1 + 1]

    def test_a_condition_keeps_its_moment_through_the_refine_round(self):
        dfr = LTX2DFRPipeline(**get_dfr_dummy_components()).to(torch_device)
        image = PIL.Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        stage = dfr(
            **get_dfr_dummy_inputs(
                generator=self.get_generator(0),
                num_frames=9,
                output_type="latent",
                conditions=[LTX2VideoCondition(frames=image, index=4, strength=1.0, crf=0)],
            )
        )
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        coords = []
        original = pipe.prepare_latents

        def capture(**kwargs):
            result = original(**kwargs)
            coords.append((kwargs["frame_rate"], result[3], kwargs.get("num_frames")))
            return result

        pipe.prepare_latents = capture
        try:
            pipe(
                prompt="a robot dancing",
                latents=stage.frames,
                keyframes_latents=stage.keyframes,
                keyframe_positions=stage.keyframe_positions,
                audio_latents=stage.audio,
                conditions=[LTX2VideoCondition(frames=image, index=4, strength=1.0, crf=0)],
                height=32,
                width=32,
                num_frames=9,
                condition_num_frames=9,
                frame_rate=25.0,
                sigmas=[0.75, 0.25],
                use_cross_timestep=False,
                max_sequence_length=16,
                output_type="latent",
                generator=self.get_generator(0),
            )
        finally:
            del pipe.prepare_latents

        frame_rate, video_coords, num_frames = coords[-1]
        latent_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1
        tokens_per_frame = (32 // pipe.vae_spatial_compression_ratio) ** 2
        condition_token = latent_frames * tokens_per_frame
        start = video_coords[0, 0, condition_token, 0].item() * frame_rate
        assert round(start) == 14, f"condition landed at pixel {start}, expected 14"
        end = video_coords[0, 0, condition_token, 1].item() * frame_rate
        assert round(end - start) == 1

    def test_last_frame_condition_stays_at_the_end_across_a_temporal_round(self):
        dfr = LTX2DFRPipeline(**get_dfr_dummy_components()).to(torch_device)
        frame = np.full((32, 32, 3), 200, dtype=np.uint8)
        last_stage = dfr(
            **get_dfr_dummy_inputs(
                generator=self.get_generator(0),
                num_frames=17,
                output_type="latent",
                conditions=LTX2VideoCondition(frames=frame, index=-1, strength=1.0, crf=0),
            )
        )
        first_stage = dfr(
            **get_dfr_dummy_inputs(
                generator=self.get_generator(0),
                num_frames=17,
                output_type="latent",
                conditions=LTX2VideoCondition(frames=frame, index=0, strength=1.0, crf=0),
            )
        )
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        shared = {
            "prompt": "a robot dancing",
            "height": 32,
            "width": 32,
            "num_frames": 17,
            "frame_rate": 25.0,
            "sigmas": [0.75, 0.25],
            "use_cross_timestep": False,
            "max_sequence_length": 16,
            "output_type": "pt",
            "generator": self.get_generator(1),
        }
        last_out = pipe(
            **shared,
            latents=last_stage.frames,
            keyframes_latents=last_stage.keyframes,
            keyframe_positions=last_stage.keyframe_positions,
            audio_latents=last_stage.audio,
            conditions=LTX2VideoCondition(frames=frame, index=-1, strength=1.0, crf=0),
            condition_num_frames=17,
        ).frames
        first_out = pipe(
            **shared,
            latents=first_stage.frames,
            keyframes_latents=first_stage.keyframes,
            keyframe_positions=first_stage.keyframe_positions,
            audio_latents=first_stage.audio,
            conditions=LTX2VideoCondition(frames=frame, index=0, strength=1.0, crf=0),
            condition_num_frames=17,
        ).frames
        assert last_out.shape[1] == (17 - 1) * 2 + 1
        assert not torch.allclose(last_out, first_out)

    def test_a_carried_slot_is_the_copy_the_stitched_canvas_kept(self):
        dfr = LTX2DFRPipeline(**get_dfr_dummy_components()).to(torch_device)
        stage = dfr(**get_dfr_dummy_inputs(generator=self.get_generator(0), num_frames=17, output_type="latent"))
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        slot_slices, anchors, denoised = [], [], []
        original_prepare, original_denoise = pipe.prepare_latents, pipe.denoise

        def capture_prepare(**kwargs):
            result = original_prepare(**kwargs)
            slot_slices.append(result[-1])
            anchors.append(kwargs.get("keyframe_latents"))
            return result

        def capture_denoise(**kwargs):
            result = original_denoise(**kwargs)
            denoised.append(result[0])
            return result

        pipe.prepare_latents, pipe.denoise = capture_prepare, capture_denoise
        try:
            round1 = pipe(
                prompt="a robot dancing",
                latents=stage.frames,
                keyframes_latents=stage.keyframes,
                keyframe_positions=stage.keyframe_positions,
                audio_latents=stage.audio,
                height=32,
                width=32,
                num_frames=17,
                frame_rate=25.0,
                round_index=1,
                sigmas=[0.75, 0.25],
                use_cross_timestep=False,
                max_sequence_length=16,
                output_type="latent",
                generator=self.get_generator(0),
            )
            pipe(
                prompt="a robot dancing",
                latents=round1.frames,
                keyframes_latents=round1.keyframes,
                keyframe_positions=round1.keyframe_positions,
                audio_latents=stage.audio,
                height=32,
                width=32,
                num_frames=(round1.frames.shape[2] - 1) * pipe.vae_temporal_compression_ratio + 1,
                frame_rate=50.0,
                source_seconds=17 / 25.0,
                round_index=2,
                sigmas=[0.75, 0.25],
                use_cross_timestep=False,
                max_sequence_length=16,
                output_type="latent",
                generator=self.get_generator(0),
            )
        finally:
            del pipe.prepare_latents, pipe.denoise

        assert len(denoised) == 2 + 4
        first_tile_slot = denoised[0][:, slot_slices[0]]
        second_tile_slot = denoised[1][:, slot_slices[1]][:, : first_tile_slot.shape[1]]
        assert not torch.allclose(first_tile_slot, second_tile_slot)
        round_2_anchors = anchors[2]
        assert [position for position, _, _ in round_2_anchors] == [16]
        _, anchor_latent, strength = round_2_anchors[0]
        assert strength == ANCHOR_KEYFRAME_STRENGTH
        assert torch.equal(pipe._pack_latents(anchor_latent), first_tile_slot)
