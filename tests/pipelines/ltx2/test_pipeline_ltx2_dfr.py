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

from diffusers import LTX2DFRPipeline, LTX2LatentUpsamplePipeline, LTXEulerAncestralRFScheduler
from diffusers.pipelines.ltx2.dfr_core import (
    EPILOGUE_KEYFRAME_STRENGTH,
    MAX_CONDITIONING_FPS,
    _audio_window_for_tile,
    trim_canvas,
)
from diffusers.pipelines.ltx2.dfr_layout import LTX2DFREpilogueTile, epilogue_tiles, video_tile_plan

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin
from .dfr_dummies import get_dfr_dummy_components, get_dfr_dummy_inputs


enable_full_determinism()


class LTX2DFRPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = LTX2DFRPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "num_frames", "frame_rate", "prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    output_shape = (9, 3, 32, 32)
    optional_input_params = BasePipelineTesterConfig.optional_input_params - {
        "num_inference_steps",
        "num_images_per_prompt",
        "latents",
    }

    def get_dummy_components(self):
        return get_dfr_dummy_components()

    def get_dummy_inputs(self):
        inputs = get_dfr_dummy_inputs()
        inputs["generator"] = self.get_generator(0)
        return inputs


class TestLTX2DFRPipeline(LTX2DFRPipelineTesterConfig, PipelineTesterMixin):
    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-3):
        super().test_inference_batch_single_identical(
            batch_size=batch_size,
            expected_max_diff=expected_max_diff,
            additional_params_copy_to_batched_inputs=[],
        )

    def test_padded_canvas_is_trimmed_back_to_the_request(self):
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = 11
        assert pipe(**inputs).frames.shape[1] == 11

    def test_latent_output_keeps_the_padded_canvas(self):
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = 11
        inputs["output_type"] = "latent"
        output = pipe(**inputs)
        trimmed = trim_canvas(output.frames, 11, pipe.vae_temporal_compression_ratio)
        assert trimmed.shape[2] < output.frames.shape[2]

    def test_the_pass_conditions_at_the_snapped_fps(self):
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        captured = []
        original = pipe.prepare_latents

        def capture(**kwargs):
            result = original(**kwargs)
            captured.append((result[3], kwargs["num_frames"], kwargs["height"], kwargs["width"]))
            return result

        inputs = self.get_dummy_inputs()
        inputs["frame_rate"] = 48.0
        pipe.prepare_latents = capture
        try:
            pipe(**inputs)
        finally:
            del pipe.prepare_latents

        assert len(captured) == 1
        coords, num_frames, height, width = captured[0]
        expected = pipe.transformer.rope.prepare_video_coords(
            batch_size=1,
            num_frames=(num_frames - 1) // pipe.vae_temporal_compression_ratio + 1,
            height=height // pipe.vae_spatial_compression_ratio,
            width=width // pipe.vae_spatial_compression_ratio,
            device=coords.device,
            fps=MAX_CONDITIONING_FPS,
        )
        assert torch.allclose(coords[:1, :, : expected.shape[2]], expected)

    def test_partially_conditioned_keyframe_starts_from_its_clean_content(self):
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        keyframe = torch.randn(1, 4, 1, 16, 16, device=torch_device)
        latents, conditioning_mask, clean_latents, _, _, _ = pipe.prepare_latents(
            keyframe_latents=[(8, keyframe, 0.95)],
            num_channels_latents=4,
            height=32,
            width=32,
            num_frames=9,
            noise_scale=0.0,
            dtype=torch.float32,
            device=torch_device,
        )
        packed_keyframe = pipe._pack_latents(keyframe)
        block = slice(latents.shape[1] - packed_keyframe.shape[1], latents.shape[1])
        assert torch.allclose(conditioning_mask[:, block], torch.full_like(conditioning_mask[:, block], 0.95))
        assert torch.allclose(clean_latents[:, block], packed_keyframe)
        assert torch.allclose(latents[:, block], packed_keyframe * 0.95, atol=1e-6)

    def test_public_latents_are_normalized_on_the_way_in(self):
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        raw = torch.randn(1, 4, 5, 16, 16, device=torch_device)
        packed_raw = pipe._pack_latents(raw)
        _, _, clean, _, _, _ = pipe.prepare_latents(
            keyframe_latents=[(8, raw[:, :, :1], 1.0)],
            num_channels_latents=4,
            height=32,
            width=32,
            num_frames=9,
            noise_scale=0.0,
            dtype=torch.float32,
            device=torch_device,
            latents=raw,
            latents_normalized=False,
        )
        assert not torch.allclose(clean[:, : packed_raw.shape[1]], packed_raw)

    def test_keyframe_marker_reaches_the_transformer(self):
        components = self.get_dummy_components()
        pipe = self.get_pipeline(**components).to(torch_device)
        marked = pipe(**self.get_dummy_inputs()).frames
        with torch.no_grad():
            components["transformer"].keyframes_abs_pos_embedding.zero_()
        unmarked = pipe(**self.get_dummy_inputs()).frames
        assert not torch.allclose(marked, unmarked)

    def test_stage1_then_spatial_upsample_then_stage2(self):
        components = get_dfr_dummy_components(spatial_upsampler=True)
        pipe = self.get_pipeline(**{k: v for k, v in components.items() if k != "latent_upsampler"}).to(torch_device)
        upsample_pipe = LTX2LatentUpsamplePipeline(
            vae=components["vae"], latent_upsampler=components["latent_upsampler"]
        ).to(torch_device)

        stage1 = pipe(**{**self.get_dummy_inputs(), "output_type": "latent"})
        up_video = upsample_pipe(latents=stage1.frames, height=32, width=32, output_type="latent", return_dict=False)[
            0
        ]
        up_keyframes = upsample_pipe(
            latents=stage1.keyframes, height=32, width=32, output_type="latent", return_dict=False
        )[0]
        stage2_height = up_video.shape[-2] * pipe.vae_spatial_compression_ratio
        stage2_width = up_video.shape[-1] * pipe.vae_spatial_compression_ratio
        stage2_inputs = self.get_dummy_inputs()
        stage2_inputs.update(
            latents=up_video,
            audio_latents=stage1.audio,
            keyframes_latents=up_keyframes,
            keyframe_positions=stage1.keyframe_positions,
            reference_latents=stage1.frames,
            height=stage2_height,
            width=stage2_width,
            sigmas=[0.75, 0.25],
            noise_scale=0.75,
            output_type="latent",
        )
        stage2 = pipe(**stage2_inputs)
        assert stage2.frames.shape[-2] == up_video.shape[-2]
        assert stage2.keyframes.shape[2] == up_keyframes.shape[2]
        assert stage2.keyframe_positions == stage1.keyframe_positions

    def _epilogue_plan(self, pipe, keyframe_positions=(8, 16, 24, 32, 40, 48, 56, 64)):
        latent_frames = 33
        latent_height = latent_width = 32 // pipe.vae_spatial_compression_ratio
        seams = [position // pipe.vae_temporal_compression_ratio for position in keyframe_positions]
        tiles = epilogue_tiles(
            latent_shape=(latent_frames, latent_height, latent_width), frame_tiles=2, frame_seams=seams
        )
        keyframe = torch.randn(1, 4, 1, latent_height, latent_width, device=torch_device)
        _, _, _, video_coords, _, _ = pipe.prepare_latents(
            keyframe_latents=[(position, keyframe, EPILOGUE_KEYFRAME_STRENGTH) for position in keyframe_positions],
            num_channels_latents=4,
            height=32,
            width=32,
            num_frames=(latent_frames - 1) * pipe.vae_temporal_compression_ratio + 1,
            dtype=torch.float32,
            device=torch_device,
            generator=self.get_generator(0),
        )
        plan = video_tile_plan(tiles, video_coords, latent_frames, latent_height, latent_width)
        return tiles, video_coords, plan, len(keyframe_positions), latent_frames * latent_height * latent_width

    def test_a_tiled_epilogue_pass_routes_every_token_with_unit_total_weight(self):
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        _, video_coords, plan, _, _ = self._epilogue_plan(pipe)
        totals = torch.zeros(video_coords.shape[2], device=video_coords.device)
        for tile in plan:
            totals.index_add_(0, tile.keep, tile.weights.to(totals.dtype))
        assert torch.allclose(totals, torch.ones_like(totals), atol=1e-6)

    def test_a_keyframe_two_epilogue_windows_share_is_a_single_token(self):
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        tiles, _, plan, num_keyframes, first_keyframe_token = self._epilogue_plan(pipe)
        assert len({(frames.start, frames.stop) for frames, _, _, _ in tiles}) > 1
        tokens_per_keyframe = (32 // pipe.vae_spatial_compression_ratio) ** 2
        shared = 0
        for index in range(num_keyframes):
            token = first_keyframe_token + index * tokens_per_keyframe
            windows = {
                (frames.start, frames.stop)
                for (frames, _, _, _), tile in zip(tiles, plan)
                if bool((tile.keep == token).any())
            }
            assert windows, f"keyframe token {token} reaches no window"
            shared += len(windows) > 1
        assert shared, "no keyframe token is shared across windows"

    def test_the_epilogue_is_given_its_keyframes_rather_than_regenerating_them(self):
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        keyframe = torch.randn(1, 4, 1, 16, 16, device=torch_device)
        positions = [8, 16, 24, 32, 40, 48, 56, 64]
        guidance = torch.cat([keyframe] * len(positions), dim=2)
        _, conditioning_mask, _, _, keyframes_mask, slot_token_slice = pipe.prepare_latents(
            keyframe_latents=[
                (position, guidance[:, :, index : index + 1], EPILOGUE_KEYFRAME_STRENGTH)
                for index, position in enumerate(positions)
            ],
            num_channels_latents=4,
            height=32,
            width=32,
            num_frames=65,
            dtype=torch.float32,
            device=torch_device,
            generator=self.get_generator(0),
        )
        assert slot_token_slice is None
        tokens_per_frame = (32 // pipe.vae_spatial_compression_ratio) ** 2
        base_tokens = (65 - 1) // pipe.vae_temporal_compression_ratio + 1
        appended = conditioning_mask[:, base_tokens * tokens_per_frame :]
        assert appended.shape[1] == 8 * tokens_per_frame
        assert torch.allclose(appended, torch.full_like(appended, EPILOGUE_KEYFRAME_STRENGTH))
        assert torch.count_nonzero(keyframes_mask[:, base_tokens * tokens_per_frame :]) == 0

    def test_composed_epilogue_pins_guidance_and_tiles(self):
        # `__call__` takes the tiling, not a resolved token plan: the plan needs the RoPE coordinates
        # `prepare_latents` builds inside the call, so a caller cannot produce one that is guaranteed to match.
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        keyframe = torch.randn(1, 4, 1, 16, 16, device=torch_device)
        output = pipe(
            **self.get_dummy_inputs(),
            generate_slots=False,
            guidance_keyframe_latents=keyframe,
            guidance_keyframe_positions=[8],
            freeze_audio=True,
            video_tiles=[
                LTX2DFREpilogueTile(
                    frames=slice(0, 5),
                    heights=slice(0, 16),
                    widths=slice(0, 16),
                    blend_weight=torch.ones(5, 16, 16),
                )
            ],
        )
        assert output.frames.shape[1] == 9
        assert pipe.num_timesteps == 2

    def test_a_single_call_tile_reproduces_the_untiled_call(self):
        # The whole canvas as one tile with unit weights must be a no-op, which is what pins the token plan
        # `__call__` resolves against its own coordinates.
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        keyframe = torch.randn(1, 4, 1, 16, 16, device=torch_device)
        shared = {
            "generate_slots": False,
            "guidance_keyframe_latents": keyframe,
            "guidance_keyframe_positions": [8],
            "freeze_audio": True,
        }
        tiled = pipe(
            **self.get_dummy_inputs(),
            **shared,
            video_tiles=[
                LTX2DFREpilogueTile(
                    frames=slice(0, 5),
                    heights=slice(0, 16),
                    widths=slice(0, 16),
                    blend_weight=torch.ones(5, 16, 16),
                )
            ],
        )
        untiled = pipe(**self.get_dummy_inputs(), **shared)
        assert torch.allclose(tiled.frames, untiled.frames, atol=1e-4)

    @pytest.mark.parametrize(("pixel_start", "first_latent"), [(0, 0), (48, 20)])
    def test_tile_audio_is_the_stage_1_window_on_the_playback_clock(self, pixel_start, first_latent):
        source = torch.arange(40, dtype=torch.float32).reshape(1, 40, 1)
        window = _audio_window_for_tile(
            source,
            pixel_start=pixel_start,
            tile_frames=48,
            playback_fps=48.0,
            source_seconds=2.0,
            conditioning_fps=48.0,
            audio_latents_per_second=20.0,
        )
        assert window.shape == (1, 20, 1)
        expected = torch.arange(first_latent, first_latent + 20, dtype=torch.float32)
        assert torch.allclose(window.flatten(), expected)

    def test_a_single_tile_plan_reproduces_the_untiled_pass(self):
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        pipe._interrupt = False
        pipe._current_timestep = None
        pipe._attention_kwargs = None
        keyframe = torch.randn(1, 4, 1, 16, 16, device=torch_device)
        reference = torch.randn(1, 4, 2, 8, 8, device=torch_device)
        latents, conditioning_mask, clean_latents, video_coords, keyframes_mask, _ = pipe.prepare_latents(
            keyframe_latents=[(8, keyframe, 0.95)],
            slot_frame_indices=[4],
            reference_latents=reference,
            reference_downscale_factor=2,
            num_channels_latents=4,
            height=32,
            width=32,
            num_frames=9,
            noise_scale=0.9,
            dtype=torch.float32,
            device=torch_device,
            generator=self.get_generator(0),
        )
        text_embeds, text_mask = pipe.encode_prompt(
            prompt="a robot dancing", max_sequence_length=16, device=torch_device
        )
        video_embeds, audio_embeds, connector_mask = pipe.connectors(
            text_embeds, text_mask, padding_side=pipe.tokenizer_padding_side
        )
        plan = video_tile_plan(
            [
                LTX2DFREpilogueTile(
                    frames=slice(0, 5),
                    heights=slice(0, 16),
                    widths=slice(0, 16),
                    blend_weight=torch.ones(5, 16, 16),
                )
            ],
            video_coords,
            5,
            16,
            16,
        )
        assert len(plan) == 1
        assert plan[0].keep.numel() == latents.shape[1]
        assert torch.equal(plan[0].coords, video_coords)
        audio_latents = pipe.prepare_audio_latents(
            num_channels_latents=pipe.audio_latent_channels,
            audio_latent_length=8,
            num_mel_bins=pipe.audio_mel_bins,
            dtype=torch.float32,
            device=torch_device,
            generator=self.get_generator(0),
        )

        def run(video_tile_plan):
            out, _ = pipe.denoise(
                latents=latents,
                conditioning_mask=conditioning_mask,
                clean_latents=clean_latents,
                video_coords=video_coords,
                keyframes_mask=keyframes_mask,
                prompt_embeds=video_embeds,
                audio_prompt_embeds=audio_embeds,
                prompt_attention_mask=connector_mask,
                sigmas=[0.9, 0.7],
                frame_rate=25.0,
                audio_latents=audio_latents,
                video_tile_plan=video_tile_plan,
                generator=self.get_generator(0),
            )
            return out

        assert torch.allclose(run(plan), run(None), atol=1e-5)

    def test_distilled_euler_keeps_the_scheduler_step(self):
        # Stage 1 / 2 stay on FlowMatch Euler. Re-pinning after every step is ancestral-only; doing it here
        # would snap IC-LoRA reference tokens and first-frame anchors every step and change the canvas.
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        pipe._interrupt = False
        pipe._current_timestep = None
        pipe._attention_kwargs = None
        keyframe = torch.randn(1, 4, 1, 16, 16, device=torch_device)
        latents, conditioning_mask, clean_latents, video_coords, keyframes_mask, _ = pipe.prepare_latents(
            keyframe_latents=[(8, keyframe, 0.95)],
            num_channels_latents=4,
            height=32,
            width=32,
            num_frames=9,
            noise_scale=0.975,
            dtype=torch.float32,
            device=torch_device,
            generator=self.get_generator(0),
        )
        text_embeds, text_mask = pipe.encode_prompt(
            prompt="a robot dancing", max_sequence_length=16, device=torch_device
        )
        video_embeds, audio_embeds, connector_mask = pipe.connectors(
            text_embeds, text_mask, padding_side=pipe.tokenizer_padding_side
        )

        def zeros_step(model_output, timestep, sample, **kwargs):
            return (torch.zeros_like(sample),)

        pipe.scheduler.step = zeros_step
        out, _ = pipe.denoise(
            latents=latents,
            conditioning_mask=conditioning_mask,
            clean_latents=clean_latents,
            video_coords=video_coords,
            keyframes_mask=keyframes_mask,
            prompt_embeds=video_embeds,
            audio_prompt_embeds=audio_embeds,
            prompt_attention_mask=connector_mask,
            sigmas=[0.975, 0.9],
            frame_rate=25.0,
            audio_latents=pipe.prepare_audio_latents(
                num_channels_latents=pipe.audio_latent_channels,
                audio_latent_length=8,
                num_mel_bins=pipe.audio_mel_bins,
                dtype=torch.float32,
                device=torch_device,
                generator=self.get_generator(0),
            ),
        )
        assert torch.allclose(out, torch.zeros_like(out))

    def test_ancestral_step_does_not_erode_conditioning(self):
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        pipe.scheduler = LTXEulerAncestralRFScheduler(eta=0.5)
        pipe._interrupt = False
        pipe._current_timestep = None
        pipe._attention_kwargs = None
        keyframe = torch.randn(1, 4, 1, 16, 16, device=torch_device)
        latents, conditioning_mask, clean_latents, video_coords, keyframes_mask, _ = pipe.prepare_latents(
            keyframe_latents=[(8, keyframe, 0.95)],
            num_channels_latents=4,
            height=32,
            width=32,
            num_frames=9,
            noise_scale=0.975,
            dtype=torch.float32,
            device=torch_device,
            generator=self.get_generator(0),
        )
        block = slice(latents.shape[1] - pipe._pack_latents(keyframe).shape[1], latents.shape[1])
        packed_keyframe = pipe._pack_latents(keyframe)
        text_embeds, text_mask = pipe.encode_prompt(
            prompt="a robot dancing", max_sequence_length=16, device=torch_device
        )
        video_embeds, audio_embeds, connector_mask = pipe.connectors(
            text_embeds, text_mask, padding_side=pipe.tokenizer_padding_side
        )
        out, _ = pipe.denoise(
            latents=latents,
            conditioning_mask=conditioning_mask,
            clean_latents=clean_latents,
            video_coords=video_coords,
            keyframes_mask=keyframes_mask,
            prompt_embeds=video_embeds,
            audio_prompt_embeds=audio_embeds,
            prompt_attention_mask=connector_mask,
            sigmas=[0.975, 0.9, 0.7],
            frame_rate=25.0,
            audio_latents=pipe.prepare_audio_latents(
                num_channels_latents=pipe.audio_latent_channels,
                audio_latent_length=8,
                num_mel_bins=pipe.audio_mel_bins,
                dtype=torch.float32,
                device=torch_device,
                generator=self.get_generator(0),
            ),
            generator=torch.Generator(device=torch_device).manual_seed(1),
        )
        cos = torch.nn.functional.cosine_similarity(
            out[:, block].float().flatten(), packed_keyframe.float().flatten(), dim=0
        )
        assert cos > 0.9, f"anchor tokens drifted from their conditioned content (cos={cos:.3f})"

    def test_the_epilogue_keeps_every_batch_element_distinct(self):
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        rebuilt = pipe.rebuild_epilogue_keyframes(
            torch.randn(2, 4, 2, 8, 8, device=torch_device),
            decode_timestep=0.0,
            decode_noise_scale=0.0,
            seed=0,
            device=torch.device(torch_device),
            dtype=torch.float32,
        )
        assert rebuilt.shape[0] == 2
        assert not torch.allclose(rebuilt[0], rebuilt[1])


class TestLTX2DFRPipelineMemory(LTX2DFRPipelineTesterConfig, MemoryTesterMixin):
    @pytest.mark.skip(
        "Pre-existing for the whole LTX-2 family, not DFR-specific: the shared harness group-offloads only "
        "`text_encoder` / `transformer` and moves `vae`, leaving the LTX-2-specific `connectors` on the CPU while it "
        "receives accelerator tensors from the offloaded text encoder. Verified to fail identically on the stock "
        "`LTX2Pipeline`. `test_pipeline_level_group_offloading_inference`, which offloads every component, passes."
    )
    def test_group_offloading_inference(self):
        pass
