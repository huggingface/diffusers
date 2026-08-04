# Copyright 2025 The HuggingFace Team.
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

import unittest

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    AutoencoderKLLTX2VideoDiffusionDecoder,
    FlowMatchEulerDiscreteScheduler,
    LTX2ImageToVideoPipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2 import (
    LTX2AutoDuration,
    LTX2DurationHead,
    LTX2LatentUpsamplePipeline,
    LTX2TextConnectors,
)
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder

from ...testing_utils import enable_full_determinism, torch_device
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class LTX2ImageToVideoPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = LTX2ImageToVideoPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS.union({"image"})
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "audio_latents",
            "return_dict",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        ]
    )
    test_attention_slicing = False
    test_xformers_attention = False

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
        )

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

        scheduler = FlowMatchEulerDiscreteScheduler()

        components = {
            "transformer": transformer,
            "vae": vae,
            "audio_vae": audio_vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "connectors": connectors,
            "vocoder": vocoder,
            "processor": None,
            "prompt_enhancer": None,
            "duration_head": None,
        }

        return components

    def get_dummy_upsample_component(self, in_channels=4, mid_channels=32, num_blocks_per_stage=1):
        upsampler = LTX2LatentUpsamplerModel(
            in_channels=in_channels,
            mid_channels=mid_channels,
            num_blocks_per_stage=num_blocks_per_stage,
        )

        return upsampler

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        image = torch.rand((1, 3, 32, 32), generator=generator, device=device)

        inputs = {
            "image": image,
            "prompt": "a robot dancing",
            "negative_prompt": "",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "frame_rate": 25.0,
            "max_sequence_length": 16,
            "output_type": "pt",
        }

        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = pipe(**inputs)
        video = output.frames
        audio = output.audio

        self.assertEqual(video.shape, (1, 5, 3, 32, 32))
        self.assertEqual(audio.shape[0], 1)
        self.assertEqual(audio.shape[1], components["vocoder"].config.out_channels)

        # fmt: off
        expected_video_slice = torch.tensor(
            [
                0.3573, 0.8382, 0.3581, 0.6114, 0.3682, 0.7969, 0.2552, 0.6399, 0.3113, 0.1497, 0.3249, 0.5395, 0.3498, 0.4526, 0.4536, 0.4555
            ]
        )
        expected_audio_slice = torch.tensor(
            [
                0.0294, 0.0498, 0.1269, 0.1135, 0.1639, 0.1116, 0.1730, 0.0931, 0.0672, -0.0069, 0.0688, 0.0097, 0.0808, 0.1231, 0.0986, 0.0739
            ]
        )
        # fmt: on

        video = video.flatten()
        audio = audio.flatten()
        generated_video_slice = torch.cat([video[:8], video[-8:]])
        generated_audio_slice = torch.cat([audio[:8], audio[-8:]])

        assert torch.allclose(expected_video_slice, generated_video_slice, atol=1e-4, rtol=1e-4)
        assert torch.allclose(expected_audio_slice, generated_audio_slice, atol=1e-4, rtol=1e-4)

    def test_two_stages_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["output_type"] = "latent"
        first_stage_output = pipe(**inputs)
        video_latent = first_stage_output.frames
        audio_latent = first_stage_output.audio

        self.assertEqual(video_latent.shape, (1, 4, 3, 16, 16))
        self.assertEqual(audio_latent.shape, (1, 2, 5, 2))
        self.assertEqual(audio_latent.shape[1], components["vocoder"].config.out_channels)

        inputs["latents"] = video_latent
        inputs["audio_latents"] = audio_latent
        inputs["output_type"] = "pt"
        second_stage_output = pipe(**inputs)
        video = second_stage_output.frames
        audio = second_stage_output.audio

        self.assertEqual(video.shape, (1, 5, 3, 32, 32))
        self.assertEqual(audio.shape[0], 1)
        self.assertEqual(audio.shape[1], components["vocoder"].config.out_channels)

        # fmt: off
        expected_video_slice = torch.tensor(
            [
                0.2665, 0.6915, 0.2939, 0.6767, 0.2552, 0.6215, 0.1765, 0.6248, 0.2800, 0.2356, 0.3480, 0.5395, 0.3190, 0.4128, 0.4784, 0.4086
            ]
        )
        expected_audio_slice = torch.tensor(
            [
                0.0273, 0.0490, 0.1253, 0.1129, 0.1655, 0.1057, 0.1707, 0.0943, 0.0672, -0.0069, 0.0688, 0.0097, 0.0808, 0.1231, 0.0986, 0.0739
            ]
        )
        # fmt: on

        video = video.flatten()
        audio = audio.flatten()
        generated_video_slice = torch.cat([video[:8], video[-8:]])
        generated_audio_slice = torch.cat([audio[:8], audio[-8:]])

        assert torch.allclose(expected_video_slice, generated_video_slice, atol=1e-4, rtol=1e-4)
        assert torch.allclose(expected_audio_slice, generated_audio_slice, atol=1e-4, rtol=1e-4)

    def test_two_stages_inference_with_upsampler(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["output_type"] = "latent"
        first_stage_output = pipe(**inputs)
        video_latent = first_stage_output.frames
        audio_latent = first_stage_output.audio

        self.assertEqual(video_latent.shape, (1, 4, 3, 16, 16))
        self.assertEqual(audio_latent.shape, (1, 2, 5, 2))
        self.assertEqual(audio_latent.shape[1], components["vocoder"].config.out_channels)

        upsampler = self.get_dummy_upsample_component(in_channels=video_latent.shape[1])
        upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=upsampler)
        upscaled_video_latent = upsample_pipe(latents=video_latent, output_type="latent", return_dict=False)[0]
        self.assertEqual(upscaled_video_latent.shape, (1, 4, 3, 32, 32))

        inputs["latents"] = upscaled_video_latent
        inputs["audio_latents"] = audio_latent
        inputs["output_type"] = "pt"
        second_stage_output = pipe(**inputs)
        video = second_stage_output.frames
        audio = second_stage_output.audio

        self.assertEqual(video.shape, (1, 5, 3, 64, 64))
        self.assertEqual(audio.shape[0], 1)
        self.assertEqual(audio.shape[1], components["vocoder"].config.out_channels)

        # fmt: off
        expected_video_slice = torch.tensor(
            [
                0.4497, 0.6757, 0.4219, 0.7686, 0.4525, 0.6483, 0.3969, 0.7404, 0.3541, 0.3039, 0.4592, 0.3521, 0.3665, 0.2785, 0.3336, 0.3079
            ]
        )
        expected_audio_slice = torch.tensor(
            [
                0.0271, 0.0492, 0.1249, 0.1126, 0.1661, 0.1060, 0.1717, 0.0944, 0.0672, -0.0069, 0.0688, 0.0097, 0.0808, 0.1231, 0.0986, 0.0739
            ]
        )
        # fmt: on

        video = video.flatten()
        audio = audio.flatten()
        generated_video_slice = torch.cat([video[:8], video[-8:]])
        generated_audio_slice = torch.cat([audio[:8], audio[-8:]])

        assert torch.allclose(expected_video_slice, generated_video_slice, atol=1e-4, rtol=1e-4)
        assert torch.allclose(expected_audio_slice, generated_audio_slice, atol=1e-4, rtol=1e-4)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(batch_size=2, expected_max_diff=2e-2)

    def get_dummy_diffusion_decoder_vae(self):
        """The LTX-2.4 diffusion-decoder VAE at the same 2x spatial / 1x temporal compression as the dummy VAE.

        The decoder's neighborhood attention needs every stage to hold at least its kernel, so with a kernel of
        3 all four upsamples but the last are identity strides -- that is what keeps a 32x32x5 dummy video
        decodable while still exercising all five stages.
        """
        torch.manual_seed(0)
        return AutoencoderKLLTX2VideoDiffusionDecoder(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(8,),
            down_block_types=("LTX2VideoDownBlock3D",),
            layers_per_block=(1,),
            spatio_temporal_scaling=(True,),
            downsample_type=("spatial",),
            patch_size=1,
            patch_size_t=1,
            encoder_causal=True,
            decoder_head_dim=16,
            decoder_stage_channels=(32, 16, 16, 16, 16),
            decoder_stage_depths=(1, 1, 1, 1, 2),
            decoder_stage_kernels=((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
            decoder_upsample_strides=((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 2, 2)),
            decoder_upsample_channel_reductions=(2, 1, 1, 1),
            decoder_stage5_kernel=(3, 3, 3),
            decoder_t_emb_dim=32,
            spatial_compression_ratio=2,
            temporal_compression_ratio=1,
        )

    def test_inference_with_diffusion_decoder_vae(self):
        """LTX-2.4's diffusion decoder decodes through the I2V pipeline, and its noise follows the generator.

        The conv-decoder path noises the latents itself at `decode_timestep` and passes that timestep to
        `vae.decode`; this decoder instead samples and denoises its own noise, so it takes the generator.
        Running twice from the same seed is what shows the generator actually reaches it -- if it did not, the
        decoder's noise would be fresh on every call.
        """
        device = "cpu"

        components = self.get_dummy_components()
        components["vae"] = self.get_dummy_diffusion_decoder_vae()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        first = pipe(**self.get_dummy_inputs(device)).frames
        second = pipe(**self.get_dummy_inputs(device)).frames

        self.assertEqual(first.shape, (1, 5, 3, 32, 32))
        assert torch.equal(first, second), "Same-seed runs should match: the decoder's noise must be seeded."

    def get_dummy_duration_head(self):
        torch.manual_seed(0)
        # The dummy connectors emit 4 heads * 8 head_dim = 32 wide output for both streams.
        return LTX2DurationHead(
            video_cross_attention_dim=32,
            audio_cross_attention_dim=32,
            pooler_hidden_dim=8,
            num_queries=1,
            num_pooler_heads=2,
            mlp_hidden_dim=8,
        )

    def test_auto_duration_produces_a_grid_valid_frame_count(self):
        components = self.get_dummy_components()
        components["duration_head"] = self.get_dummy_duration_head()
        pipe = self.pipeline_class(**components).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_frames"] = LTX2AutoDuration(min_seconds=0.5, max_seconds=2.0)
        frames = pipe(**inputs).frames[0]

        ratio = pipe.vae_temporal_compression_ratio
        assert (len(frames) - 1) % ratio == 0
        assert 0 < len(frames) <= round(2.0 * inputs["frame_rate"])

    def test_omitting_num_frames_auto_predicts_when_a_head_is_present(self):
        components = self.get_dummy_components()
        components["duration_head"] = self.get_dummy_duration_head()
        pipe = self.pipeline_class(**components).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs.pop("num_frames")
        inputs["output_type"] = "latent"
        latents = pipe(**inputs).frames

        legacy_latent_frames = (121 - 1) // pipe.vae_temporal_compression_ratio + 1
        assert latents.shape[2] != legacy_latent_frames, (
            "omitting num_frames with a head present must not use the legacy default"
        )

    def test_omitting_num_frames_uses_the_legacy_default_without_a_head(self):
        # Guards backwards compatibility: a pre-2.4 pipeline has no duration_head and must keep 121.
        components = self.get_dummy_components()
        assert components.get("duration_head") is None
        pipe = self.pipeline_class(**components).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs.pop("num_frames")
        # Decoding 121 frames is needlessly slow here; the latent frame count already pins num_frames down.
        inputs["output_type"] = "latent"
        latents = pipe(**inputs).frames

        # Latents come back unpacked as [batch, channels, latent_frames, height, width].
        expected_latent_frames = (121 - 1) // pipe.vae_temporal_compression_ratio + 1
        assert latents.shape[2] == expected_latent_frames

    def test_explicit_num_frames_wins_over_a_present_head(self):
        components = self.get_dummy_components()
        components["duration_head"] = self.get_dummy_duration_head()
        pipe = self.pipeline_class(**components).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_frames"] = 9
        frames = pipe(**inputs).frames[0]

        assert len(frames) == 9

    def test_auto_duration_with_multiple_prompts_raises(self):
        # The head predicts one duration, so it cannot serve prompts with different natural lengths.
        # Without this guard the pipeline silently applied the first prompt's length to all of them.
        components = self.get_dummy_components()
        components["duration_head"] = self.get_dummy_duration_head()
        pipe = self.pipeline_class(**components).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt"] = ["a robot dancing", "a much longer and quite different scene"]
        inputs["negative_prompt"] = ["", ""]
        inputs["num_frames"] = LTX2AutoDuration()

        with self.assertRaises(ValueError) as ctx:
            pipe(**inputs)
        assert "2 prompts were supplied" in str(ctx.exception)

    def test_multiple_prompts_still_work_with_an_explicit_num_frames(self):
        # The guard must be scoped to the auto path -- batched prompts with an integer are unaffected.
        components = self.get_dummy_components()
        components["duration_head"] = self.get_dummy_duration_head()
        pipe = self.pipeline_class(**components).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt"] = ["a robot dancing", "a much longer and quite different scene"]
        inputs["negative_prompt"] = ["", ""]
        inputs["num_frames"] = 5
        inputs["output_type"] = "latent"

        latents = pipe(**inputs).frames

        assert latents.shape[0] == 2

    def test_auto_duration_without_a_head_raises(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_frames"] = LTX2AutoDuration()

        with self.assertRaises(ValueError) as ctx:
            pipe(**inputs)
        assert "duration_head" in str(ctx.exception)
