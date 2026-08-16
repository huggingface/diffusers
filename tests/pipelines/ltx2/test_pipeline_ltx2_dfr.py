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
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
    LTX2DFRPipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplerModel, LTX2TextConnectors
from diffusers.pipelines.ltx2.dfr_layout import LTX2DFREpilogueTile, epilogue_tiles, video_tile_plan
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.pipeline_ltx2_dfr import (
    ANCHOR_KEYFRAME_STRENGTH,
    EPILOGUE_KEYFRAME_STRENGTH,
    MAX_CONDITIONING_FPS,
    _audio_window_for_tile,
)
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class LTX2DFRPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = LTX2DFRPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "num_frames", "frame_rate", "prompt_embeds", "temporal_upscalings"]
    )
    batch_input_params = frozenset(["prompt"])
    output_shape = (9, 3, 32, 32)
    # DFR stages use fixed distilled schedules (`stage_*_sigmas`); `latents` cannot seed stage 1, which always
    # generates at half resolution. Videos are counted with `num_videos_per_prompt`.
    optional_input_params = BasePipelineTesterConfig.optional_input_params - {
        "num_inference_steps",
        "num_images_per_prompt",
        "latents",
    }

    base_text_encoder_ckpt_id = "hf-internal-testing/tiny-gemma3"

    def get_dummy_components(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_text_encoder_ckpt_id)
        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(self.base_text_encoder_ckpt_id)

        torch.manual_seed(0)
        transformer = LTX2VideoTransformer3DModel(
            in_channels=4,
            out_channels=4,
            patch_size=1,
            patch_size_t=1,
            num_attention_heads=2,
            attention_head_dim=8,
            cross_attention_dim=16,
            audio_in_channels=4,
            audio_out_channels=4,
            audio_num_attention_heads=2,
            audio_attention_head_dim=4,
            audio_cross_attention_dim=8,
            num_layers=2,
            qk_norm="rms_norm_across_heads",
            caption_channels=text_encoder.config.text_config.hidden_size,
            rope_double_precision=False,
            rope_type="split",
            vae_scale_factors=(2, 2, 2),
            use_keyframes_abs_pos_embedding=True,
        )
        # Zero-init would make a broken `video_keyframes_mask` an exact no-op.
        torch.nn.init.normal_(transformer.keyframes_abs_pos_embedding, std=0.1)

        torch.manual_seed(0)
        connectors = LTX2TextConnectors(
            caption_channels=text_encoder.config.text_config.hidden_size,
            text_proj_in_factor=text_encoder.config.text_config.num_hidden_layers + 1,
            video_connector_num_attention_heads=4,
            video_connector_attention_head_dim=8,
            video_connector_num_layers=1,
            video_connector_num_learnable_registers=None,
            audio_connector_num_attention_heads=4,
            audio_connector_attention_head_dim=8,
            audio_connector_num_layers=1,
            audio_connector_num_learnable_registers=None,
            connector_rope_base_seq_len=32,
            rope_theta=10000.0,
            rope_double_precision=False,
            causal_temporal_positioning=False,
            rope_type="split",
        )

        torch.manual_seed(0)
        vae = AutoencoderKLLTX2Video(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(8,),
            decoder_block_out_channels=(8,),
            layers_per_block=(1,),
            decoder_layers_per_block=(1, 1),
            spatio_temporal_scaling=(True,),
            decoder_spatio_temporal_scaling=(True,),
            decoder_inject_noise=(False, False),
            downsample_type=("spatial",),
            upsample_residual=(False,),
            upsample_factor=(1,),
            timestep_conditioning=False,
            patch_size=1,
            patch_size_t=1,
            encoder_causal=True,
            decoder_causal=False,
        )
        vae.use_framewise_encoding = False
        vae.use_framewise_decoding = False

        torch.manual_seed(0)
        audio_vae = AutoencoderKLLTX2Audio(
            base_channels=4,
            output_channels=2,
            ch_mult=(1,),
            num_res_blocks=1,
            attn_resolutions=None,
            in_channels=2,
            resolution=32,
            latent_channels=2,
            norm_type="pixel",
            causality_axis="height",
            dropout=0.0,
            mid_block_add_attention=False,
            sample_rate=16000,
            mel_hop_length=160,
            is_causal=True,
            mel_bins=8,
        )

        torch.manual_seed(0)
        vocoder = LTX2Vocoder(
            in_channels=audio_vae.config.output_channels * audio_vae.config.mel_bins,
            hidden_channels=32,
            out_channels=2,
            upsample_kernel_sizes=[4, 4],
            upsample_factors=[2, 2],
            resnet_kernel_sizes=[3],
            resnet_dilations=[[1, 3, 5]],
            leaky_relu_negative_slope=0.1,
            output_sampling_rate=16000,
        )

        # `mid_channels` feeds a GroupNorm with 32 groups, so it cannot be shrunk below 32.
        torch.manual_seed(0)
        latent_upsampler = LTX2LatentUpsamplerModel(
            in_channels=4,
            mid_channels=32,
            num_blocks_per_stage=1,
            dims=3,
            spatial_upsample=True,
            temporal_upsample=False,
            use_rational_resampler=False,
        )

        torch.manual_seed(0)
        temporal_latent_upsampler = LTX2LatentUpsamplerModel(
            in_channels=4,
            mid_channels=32,
            num_blocks_per_stage=1,
            dims=3,
            spatial_upsample=False,
            temporal_upsample=True,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "transformer": transformer,
            "vae": vae,
            "audio_vae": audio_vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "connectors": connectors,
            "vocoder": vocoder,
            "latent_upsampler": latent_upsampler,
            "temporal_latent_upsampler": temporal_latent_upsampler,
            "processor": None,
            "prompt_enhancer": None,
            "duration_head": None,
        }

    def get_dummy_inputs(self):
        # Dummy VAE temporal ratio is 2, so the segment grid is 6 / 8 pixel frames (not 24 / 32).
        return {
            "prompt": "a robot dancing",
            "generator": self.get_generator(0),
            "height": 32,
            "width": 32,
            "num_frames": 9,
            "frame_rate": 25.0,
            "stage_1_sigmas": [1.0, 0.5],
            "stage_2_sigmas": [0.75, 0.25],
            "temporal_round_sigmas": [0.75, 0.25],
            "use_cross_timestep": False,
            "max_sequence_length": 16,
            "output_type": "pt",
        }


class TestLTX2DFRPipeline(LTX2DFRPipelineTesterConfig, PipelineTesterMixin):
    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-3):
        # Mixin default copies `num_inference_steps` into batched inputs (DFR has none). Tolerance is 1e-3 because
        # batched vs single matmuls on accelerator reduce in a different order (max diff ~6e-4 on CUDA).
        super().test_inference_batch_single_identical(
            batch_size=batch_size,
            expected_max_diff=expected_max_diff,
            additional_params_copy_to_batched_inputs=[],
        )

    def test_padded_canvas_is_trimmed_back_to_the_request(self):
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)

        inputs = self.get_dummy_inputs()
        # 10 content frames: segment 6 pads by 2, so the canvas is 13 and must be trimmed to 11.
        inputs["num_frames"] = 11
        output = pipe(**inputs)

        assert output.frames.shape[1] == 11

    def test_temporal_upsample_round_doubles_the_frame_count(self):
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)

        inputs = self.get_dummy_inputs()
        inputs["temporal_upscalings"] = 1
        output = pipe(**inputs)

        assert output.frames.shape[1] == (self.output_shape[0] - 1) * 2 + 1
        # One 8-frame segment -> one tile: 2 + 2 stage steps plus 2 tile steps. `callback_on_step_end` indexes this.
        assert pipe.num_timesteps == 2 + 2 + 2

    def test_temporal_round_tiles_get_distinct_ancestral_noise(self):
        # Tiles are positionally identical; each must draw from `seed + 1000 * round + tile`, not a shared stream.
        import diffusers.pipelines.ltx2.pipeline_ltx2_dfr as dfr_module

        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)

        captured = []
        original_step = dfr_module.ancestral_euler_step

        def capture(sample, denoised, sigma, sigma_next, eta, noise):
            captured.append(noise.clone())
            return original_step(sample, denoised, sigma, sigma_next, eta, noise)

        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = 17
        inputs["temporal_upscalings"] = 1
        dfr_module.ancestral_euler_step = capture
        try:
            pipe(**inputs)
        finally:
            dfr_module.ancestral_euler_step = original_step

        assert len(captured) >= 2, f"expected ancestral draws from at least 2 tiles, got {len(captured)}"

        first_tile = captured[0]
        expected = randn_tensor(
            first_tile.shape,
            generator=torch.Generator(device=torch_device).manual_seed(0 + 1000 * 1 + 0),
            device=torch.device(torch_device),
            dtype=first_tile.dtype,
        )
        assert torch.equal(first_tile, expected)

    @pytest.mark.parametrize(
        ("num_frames", "temporal_upscalings", "output_frames", "num_timesteps"),
        [
            # No rounds: the epilogue is one 2-step loop, whatever its tiling, because the tiles live inside the
            # transformer call rather than around the schedule.
            (9, 0, 9, 2 + 2 + 2),
            # One round leaves a 33-frame canvas the epilogue then cuts 2 ways in time as well; still one loop.
            (17, 1, 33, 2 + 2 + 2 * 2 + 2),
        ],
    )
    def test_the_epilogue_runs_on_the_canvas_the_earlier_stages_left_behind(
        self, num_frames, temporal_upscalings, output_frames, num_timesteps
    ):
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)

        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = num_frames
        inputs["temporal_upscalings"] = temporal_upscalings
        inputs["spatial_upscalings"] = 2
        output = pipe(**inputs)

        # `height` / `width` are the output either way; only the stages below them move.
        assert output.frames.shape[1:] == (output_frames, *self.output_shape[1:])
        assert pipe.num_timesteps == num_timesteps

    @pytest.mark.parametrize(("temporal_upscalings", "expected_pixel"), [(0, 7), (1, 14), (2, 28)])
    def test_a_condition_keeps_its_moment_through_the_refine_rounds(self, temporal_upscalings, expected_pixel):
        # `index` is a latent index on the canvas the caller asked for. Each round doubles the canvas and the frame
        # rate, so a fixed moment moves to twice its pixel position, and the epilogue runs on the refined canvas. The
        # scaled position does not generally land on a latent boundary, so it has to travel as a pixel index -- the
        # dummy VAE ratio is 2, latent 4 is pixel 7, and one round puts it at 14, which is not on the latent grid.
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)

        coords = []
        original = pipe.prepare_latents

        def capture(**kwargs):
            result = original(**kwargs)
            coords.append((kwargs["frame_rate"], result[3], kwargs.get("num_frames")))
            return result

        image = PIL.Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = 33
        inputs["temporal_upscalings"] = temporal_upscalings
        inputs["spatial_upscalings"] = 2
        inputs["conditions"] = [LTX2VideoCondition(frames=image, index=4, strength=1.0)]
        pipe.prepare_latents = capture
        try:
            pipe(**inputs)
        finally:
            del pipe.prepare_latents

        # The epilogue is the last pass and covers the whole canvas, so its condition sits where the caller asked.
        # Read the position back off the RoPE coordinates the pass will actually attend with, in seconds / fps.
        frame_rate, video_coords, num_frames = coords[-1]
        latent_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1
        tokens_per_frame = (32 // pipe.vae_spatial_compression_ratio) ** 2
        condition_token = latent_frames * tokens_per_frame
        start = video_coords[0, 0, condition_token, 0].item() * frame_rate
        assert round(start) == expected_pixel, f"condition landed at pixel {start}, expected {expected_pixel}"
        # A single-pixel-frame keyframe spans exactly one frame.
        end = video_coords[0, 0, condition_token, 1].item() * frame_rate
        assert round(end - start) == 1

    def test_the_epilogue_resolution_rule_is_reported_before_the_generic_one(self):
        # `spatial_upscalings=2` needs multiples of `2 ** upscalings * vae_spatial_compression_ratio` -- 128 on the
        # shipped VAE, which is why UHD is 3840x2176 rather than 3840x2160. A caller who misses it must be told that
        # divisor, not the looser one every LTX-2 pipeline applies.
        pipe = self.get_pipeline(**self.get_dummy_components())
        divisor = 4 * pipe.vae_spatial_compression_ratio

        inputs = self.get_dummy_inputs()
        inputs["spatial_upscalings"] = 2
        inputs["height"] = inputs["width"] = 32 + divisor // 2
        with pytest.raises(ValueError, match=f"divisible by {divisor}"):
            pipe(**inputs)

    def test_the_epilogue_keeps_every_batch_element_distinct(self):
        # The carry keyframes are rebuilt through RGB one plane at a time, and every batch element has its own picture
        # to stretch. Collapsing the batch there would hand one element's frames to all of them.
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)

        inputs = self.get_dummy_inputs()
        inputs["spatial_upscalings"] = 2
        inputs["num_videos_per_prompt"] = 2
        output = pipe(**inputs)

        assert output.frames.shape[0] == 2

    def _epilogue_inputs(self, **overrides):
        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = 33
        inputs["temporal_upscalings"] = 1
        inputs["spatial_upscalings"] = 2
        inputs.update(overrides)
        return inputs

    def _epilogue_plan(self, pipe, keyframe_positions=(8, 16, 24, 32, 40, 48, 56, 64)):
        """The epilogue's real tiling and token plan for a 33-frame request with one refine round."""
        # One round turns the 33-frame canvas into 65 frames; at the dummy ratio of 2 that is a 33-frame latent grid,
        # cut on the round's seams and split two ways on each spatial axis.
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
        # Conditionings are attached once on the whole canvas and filtered per tile at the token level, so the routing
        # is what decides which window sees which keyframe. Every token -- base grid and appended alike -- must come
        # out with a total weight of exactly one, or the blended canvas is dimmed, doubled, or missing tokens.
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)

        _, video_coords, plan, _, _ = self._epilogue_plan(pipe)

        totals = torch.zeros(video_coords.shape[2], device=video_coords.device)
        for tile in plan:
            totals.index_add_(0, tile.keep, tile.weights.to(totals.dtype))
        assert torch.allclose(totals, torch.ones_like(totals), atol=1e-6)

    def test_a_keyframe_two_epilogue_windows_share_is_a_single_token(self):
        # A later window's dropped run-up is content the window before it owns. There is one token per keyframe and
        # every window covering it reads that same token, so the two cannot disagree about a frame they both hold.
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)

        tiles, _, plan, num_keyframes, first_keyframe_token = self._epilogue_plan(pipe)
        assert len({(frames.start, frames.stop) for frames, _, _, _ in tiles}) > 1, "expected 2 time windows"

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
        # The carry keyframes are frames the refine rounds already settled. Re-attaching them as generated slots asked
        # the pass to invent content it already had; they arrive rebuilt at the output resolution and pinned clean.
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)

        calls = []
        original = pipe.prepare_latents

        def capture(**kwargs):
            result = original(**kwargs)
            calls.append((kwargs, result))
            return result

        pipe.prepare_latents = capture
        try:
            pipe(**self._epilogue_inputs())
        finally:
            del pipe.prepare_latents

        # The epilogue is the last pass, so its call is the one left standing. Read the answer off the state it built
        # rather than the arguments it was handed.
        _, (_, conditioning_mask, _, _, keyframes_mask, slot_token_slice) = calls[-1]
        assert slot_token_slice is None, "the epilogue must not allocate generated slot tokens"

        # Everything appended past the base grid is the carry keyframes, at the output resolution.
        tokens_per_frame = (32 // pipe.vae_spatial_compression_ratio) ** 2
        base_tokens = (65 - 1) // pipe.vae_temporal_compression_ratio + 1
        appended = conditioning_mask[:, base_tokens * tokens_per_frame :]
        assert appended.shape[1] == 8 * tokens_per_frame, "expected the 8 carry keyframes of a one-round canvas"
        # Pinned fully clean...
        assert torch.allclose(appended, torch.full_like(appended, EPILOGUE_KEYFRAME_STRENGTH))
        # ...and unmarked, which is what distinguishes given content from a generated slot.
        assert torch.count_nonzero(keyframes_mask[:, base_tokens * tokens_per_frame :]) == 0

    @pytest.mark.parametrize(("pixel_start", "first_latent"), [(0, 0), (48, 20)])
    def test_tile_audio_is_the_stage_1_window_on_the_playback_clock(self, pixel_start, first_latent):
        # 40 stage-1 audio frames spanning 2 s. A refine round doubles the frame rate, so a 48-frame tile at 48 fps
        # is 1 s of sound however long the canvas got -- the tile starting at 48 hears the second second, and only it.
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
        # `_video_tile_plan` is the whole correctness surface of the tiled path: which tokens a tile sees, how its
        # positions are rebased, and what weight its prediction carries. Degenerate it to one tile covering the whole
        # canvas and the two paths must agree exactly -- anything else is a bug in the plan rather than in the tiling.
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        pipe._interrupt = False
        pipe._current_timestep = None
        pipe._attention_kwargs = None

        keyframe = torch.randn(1, 4, 1, 16, 16, device=torch_device)
        reference = torch.randn(1, 4, 2, 8, 8, device=torch_device)
        prepared = pipe.prepare_latents(
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
        latents, conditioning_mask, clean_latents, video_coords, keyframes_mask, _ = prepared

        text_embeds, text_mask = pipe.encode_prompt(
            prompt="a robot dancing", max_sequence_length=16, device=torch_device
        )
        video_embeds, audio_embeds, connector_mask = pipe.connectors(
            text_embeds, text_mask, padding_side=pipe.tokenizer_padding_side
        )
        # 9 pixel frames at the dummy VAE's temporal ratio of 2 is a 5-frame latent grid.
        whole_canvas = [
            LTX2DFREpilogueTile(
                frames=slice(0, 5),
                heights=slice(0, 16),
                widths=slice(0, 16),
                blend_weight=torch.ones(5, 16, 16),
            )
        ]
        plan = video_tile_plan(whole_canvas, video_coords, 5, 16, 16)
        assert len(plan) == 1
        assert plan[0].keep.numel() == latents.shape[1], "one tile must keep every token"
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

    def test_ancestral_step_does_not_erode_conditioning(self):
        # Ancestral Euler noises every token; the strength-0.95 blend has to be re-applied or tile seams drift.
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        pipe._interrupt = False
        pipe._current_timestep = None
        pipe._attention_kwargs = None

        keyframe = torch.randn(1, 4, 1, 16, 16, device=torch_device)
        strength = 0.95
        latents, conditioning_mask, clean_latents, video_coords, keyframes_mask, _ = pipe.prepare_latents(
            keyframe_latents=[(8, keyframe, strength)],
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
            ancestral_eta=0.5,
            generator=self.get_generator(0),
            ancestral_generator=torch.Generator(device=torch_device).manual_seed(1),
        )

        anchor_out = out[:, block].float()
        cos = torch.nn.functional.cosine_similarity(anchor_out.flatten(), packed_keyframe.float().flatten(), dim=0)
        assert cos > 0.9, f"anchor tokens drifted from their conditioned content (cos={cos:.3f})"

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

    def test_keyframe_marker_reaches_the_transformer(self):
        components = self.get_dummy_components()
        pipe = self.get_pipeline(**components).to(torch_device)

        marked = pipe(**self.get_dummy_inputs()).frames

        with torch.no_grad():
            components["transformer"].keyframes_abs_pos_embedding.zero_()
        unmarked = pipe(**self.get_dummy_inputs()).frames

        assert not torch.allclose(marked, unmarked)

    def test_last_frame_condition_stays_at_the_end_across_a_temporal_round(self):
        # 17 frames -> 2 segments, so the temporal round runs 2 tiles. `index=-1` must land on the last original
        # latent (scaled into the 2x canvas), not wrap to the first tile's last frame or to frame 0.
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)
        frame = np.full((32, 32, 3), 200, dtype=np.uint8)

        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = 17
        inputs["temporal_upscalings"] = 1
        last = dict(inputs)
        last["conditions"] = LTX2VideoCondition(frames=frame, index=-1, strength=1.0, crf=0)
        first = dict(inputs)
        first["conditions"] = LTX2VideoCondition(frames=frame, index=0, strength=1.0, crf=0)

        last_out = pipe(**last).frames
        first_out = pipe(**first).frames

        assert last_out.shape[1] == (17 - 1) * 2 + 1
        assert not torch.allclose(last_out, first_out)

    def test_stages_1_and_2_condition_at_the_snapped_fps(self):
        # RoPE time is `pixel_frame / fps`, and the transformer never saw 48. Every stage must lay its tokens out at
        # the snapped rate; only the returned frame count and the audio trim use the playback rate.
        pipe = self.get_pipeline(**self.get_dummy_components()).to(torch_device)

        base_coords = []
        original = pipe.prepare_latents

        def capture(**kwargs):
            result = original(**kwargs)
            base_coords.append((result[3], kwargs["num_frames"], kwargs["height"], kwargs["width"]))
            return result

        inputs = self.get_dummy_inputs()
        inputs["frame_rate"] = 48.0
        pipe.prepare_latents = capture
        try:
            pipe(**inputs)
        finally:
            del pipe.prepare_latents

        assert len(base_coords) == 2, "expected stage 1 and stage 2"
        for coords, num_frames, height, width in base_coords:
            # Compare against what the real RoPE produces at 60, rather than trusting the argument that was plumbed in.
            expected = pipe.transformer.rope.prepare_video_coords(
                batch_size=1,
                num_frames=(num_frames - 1) // pipe.vae_temporal_compression_ratio + 1,
                height=height // pipe.vae_spatial_compression_ratio,
                width=width // pipe.vae_spatial_compression_ratio,
                device=coords.device,
                fps=MAX_CONDITIONING_FPS,
            )
            assert torch.allclose(coords[:1, :, : expected.shape[2]], expected)

    def test_a_carried_slot_is_the_copy_the_stitched_canvas_kept(self):
        # Two tiles invent the slot that falls in the later one's dropped lead-in. The stitch keeps the earlier tile's
        # frames there, so the earlier tile's slot is the one the next round must anchor on; the later tile's describes
        # frames the canvas does not hold.
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

        inputs = self.get_dummy_inputs()
        # 17 frames -> segment 8, so round 1 runs 2 tiles whose slot at pixel 8 falls in tile 1's dropped lead-in,
        # and round 2 tile 0 anchors on it at pixel 16.
        inputs["num_frames"] = 17
        inputs["temporal_upscalings"] = 2
        pipe.prepare_latents, pipe.denoise = capture_prepare, capture_denoise
        try:
            pipe(**inputs)
        finally:
            del pipe.prepare_latents, pipe.denoise

        # Passes are stage 1, stage 2, then round 1's 2 tiles, then round 2's 4 tiles.
        assert len(denoised) == 2 + 2 + 4
        first_tile_slot = denoised[2][:, slot_slices[2]]
        second_tile_slot = denoised[3][:, slot_slices[3]][:, : first_tile_slot.shape[1]]
        assert not torch.allclose(first_tile_slot, second_tile_slot), "the two tiles must invent different content"

        round_2_anchors = anchors[4]
        assert [position for position, _, _ in round_2_anchors] == [16]
        _, anchor_latent, strength = round_2_anchors[0]
        assert strength == ANCHOR_KEYFRAME_STRENGTH
        assert torch.equal(pipe._pack_latents(anchor_latent), first_tile_slot)


class TestLTX2DFRPipelineMemory(LTX2DFRPipelineTesterConfig, MemoryTesterMixin):
    @pytest.mark.skip(
        "Pre-existing for the whole LTX-2 family, not DFR-specific: the shared harness group-offloads only "
        "`text_encoder` / `transformer` and moves `vae`, leaving the LTX-2-specific `connectors` on the CPU while it "
        "receives accelerator tensors from the offloaded text encoder. Verified to fail identically on the stock "
        "`LTX2Pipeline`. `test_pipeline_level_group_offloading_inference`, which offloads every component, passes."
    )
    def test_group_offloading_inference(self):
        pass
