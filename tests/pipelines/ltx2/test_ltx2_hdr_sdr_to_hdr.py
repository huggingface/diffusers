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

"""`LTX2HDRPipeline` with `hdr_transform="acescct"`: the LTX-2.5 SDR-To-HDR IC-LoRA path."""

from unittest import mock

import pytest
import torch

from diffusers import LTX2HDRPipeline, LTX2VideoDiffusionDecoderModel
from diffusers.pipelines.ltx2 import LTX2HDRReferenceCondition
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
from diffusers.utils.import_utils import is_peft_available

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils.common import BasePipelineOutputMixin
from .test_ltx2_hdr import LTX2HDRPipelineTesterConfig


enable_full_determinism()


# The dummy VAE compresses by 2 in space and time, so 31 x 29 is padded to 32 x 30 and 5 frames is "2k + 1".
HEIGHT, WIDTH, NUM_FRAMES = 31, 29, 5
PADDED_HEIGHT, PADDED_WIDTH = 32, 30
# The dummy transformer uses LTX-2.0-style caption projections, whose input width is the tiny Gemma's hidden size.
CONTEXT_DIM = 32


def get_dummy_diffusion_decoder():
    """A tiny `LTX2VideoDiffusionDecoderModel` matching the dummy VAE: 4 latent channels, x2 in space and time."""
    torch.manual_seed(0)
    return LTX2VideoDiffusionDecoderModel(
        out_channels=3,
        latent_channels=4,
        patch_size=2,
        decoder_head_dim=16,
        decoder_stage_channels=(32, 16, 16, 16, 16),
        decoder_stage_depths=(1, 1, 1, 1, 1),
        decoder_stage_kernels=((3, 3, 3),) * 4,
        decoder_upsample_strides=((1, 1, 1), (2, 1, 1), (1, 1, 1), (1, 1, 1)),
        decoder_upsample_channel_reductions=(2, 1, 1, 1),
        decoder_stage5_kernel=(3, 3, 3),
        decoder_t_emb_dim=32,
        spatial_compression_ratio=2,
        temporal_compression_ratio=2,
    )


class TestLTX2HDRPipelineSDRToHDR(LTX2HDRPipelineTesterConfig, BasePipelineOutputMixin):
    def get_sdr_to_hdr_pipeline(self, text_components: bool = False, diffusion_decoder: bool = False):
        components = self.get_dummy_components()
        if not text_components:
            # The reference runs no text encoder, so the pipeline must work without one.
            components.update(text_encoder=None, tokenizer=None, connectors=None)
        if diffusion_decoder:
            components["diffusion_decoder"] = get_dummy_diffusion_decoder()
        pipe = LTX2HDRPipeline(**components, hdr_transform="acescct")
        pipe.set_progress_bar_config(disable=True)
        return pipe.to(torch_device)

    def get_sdr_to_hdr_inputs(self, **overrides):
        generator = self.get_generator(0)
        # An 8-bit sRGB clip whose size is not a multiple of the VAE's spatial compression ratio.
        frames = torch.randint(0, 256, (NUM_FRAMES, HEIGHT, WIDTH, 3), generator=generator, dtype=torch.uint8)
        inputs = {
            "reference_conditions": LTX2HDRReferenceCondition(frames=frames.numpy()),
            # Stored without a batch dimension, like the IC-LoRA's `video_context`.
            "connector_video_embeds": torch.randn(7, CONTEXT_DIM, generator=generator),
            "height": HEIGHT,
            "width": WIDTH,
            "num_frames": NUM_FRAMES,
            "frame_rate": 24.0,
            "sigmas": [1.0, 0.5],
            "generator": generator,
            "output_type": "pt",
        }
        inputs.update(overrides)
        return inputs

    def test_end_to_end(self):
        pipe = self.get_sdr_to_hdr_pipeline()
        assert pipe.text_encoder is None and pipe.tokenizer is None and pipe.connectors is None

        video = pipe(**self.get_sdr_to_hdr_inputs()).frames
        assert video.shape == (1, NUM_FRAMES, HEIGHT, WIDTH, 3)
        assert video.dtype == torch.float32
        assert torch.isfinite(video).all()
        assert video.min() >= 0.0

        codes = pipe(**self.get_sdr_to_hdr_inputs(output_colorspace="acescct")).frames
        assert codes.min() >= 0.0 and codes.max() <= 1.0

    def test_end_to_end_diffusion_decoder(self):
        pipe = self.get_sdr_to_hdr_pipeline(diffusion_decoder=True)
        with mock.patch.object(pipe.vae, "decode", side_effect=AssertionError("the VAE must not decode")):
            video = pipe(**self.get_sdr_to_hdr_inputs()).frames
        assert video.shape == (1, NUM_FRAMES, HEIGHT, WIDTH, 3)
        assert torch.isfinite(video).all()

    def test_no_text_encoder_call(self):
        pipe = self.get_sdr_to_hdr_pipeline(text_components=True)
        forbidden = AssertionError("the SDR-To-HDR path must not encode a prompt")
        with (
            mock.patch.object(pipe, "encode_prompt", side_effect=forbidden),
            mock.patch.object(pipe.text_encoder, "forward", side_effect=forbidden),
            mock.patch.object(pipe.connectors, "forward", side_effect=forbidden),
        ):
            pipe(**self.get_sdr_to_hdr_inputs())

    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"prompt": "a robot dancing"}, "prompt"),
            ({"connector_video_embeds": None}, "connector_video_embeds"),
            ({"guidance_scale": 3.0}, "guidance"),
            ({"stg_scale": 1.0}, "guidance"),
            ({"modality_scale": 2.0}, "guidance"),
            ({"reference_conditions": None}, "reference"),
            ({"num_frames": 4}, "num_frames"),
            ({"num_frames": 7}, "fewer than"),
        ],
    )
    def test_invalid_inputs(self, overrides, match):
        pipe = self.get_sdr_to_hdr_pipeline()
        with pytest.raises(ValueError, match=match):
            pipe(**self.get_sdr_to_hdr_inputs(**overrides))

    def test_batched_scene_embedding_matches_unbatched(self):
        pipe = self.get_sdr_to_hdr_pipeline()
        inputs = self.get_sdr_to_hdr_inputs(output_type="latent")
        unbatched = pipe(**inputs).frames
        inputs = self.get_sdr_to_hdr_inputs(output_type="latent")
        inputs["connector_video_embeds"] = inputs["connector_video_embeds"][None]
        batched = pipe(**inputs).frames
        assert torch.equal(unbatched, batched)

    @staticmethod
    def _perturb_audio(pipe, seed):
        """Feed the transformer random audio latents and audio context instead of what the pipeline built."""
        forward = pipe.transformer.forward

        def perturbed_forward(*args, **kwargs):
            generator = torch.Generator().manual_seed(seed)
            for name in ("audio_hidden_states", "audio_encoder_hidden_states"):
                value = kwargs[name]
                kwargs[name] = torch.randn(value.shape, generator=generator).to(value.device, value.dtype) * 3.0
            return forward(*args, **kwargs)

        return mock.patch.object(pipe.transformer, "forward", side_effect=perturbed_forward)

    def test_audio_does_not_influence_video(self):
        pipe = self.get_sdr_to_hdr_pipeline()
        outputs = []
        for seed in (0, 1):
            with self._perturb_audio(pipe, seed):
                outputs.append(pipe(**self.get_sdr_to_hdr_inputs(output_type="latent")).frames)
        assert torch.equal(outputs[0], outputs[1])

        # Passing the IC-LoRA's `audio_context` (the reference ignores it) does not change the video either.
        with_audio_context = pipe(
            **self.get_sdr_to_hdr_inputs(output_type="latent", connector_audio_embeds=torch.randn(7, CONTEXT_DIM))
        ).frames
        without_audio_context = pipe(**self.get_sdr_to_hdr_inputs(output_type="latent")).frames
        assert torch.equal(with_audio_context, without_audio_context)

    def test_audio_influences_video_with_logc3(self):
        # Sanity check of `_perturb_audio`: the LogC3 path keeps audio-to-video cross-attention.
        pipe = self.get_pipeline().to(torch_device)
        outputs = []
        for seed in (0, 1):
            with self._perturb_audio(pipe, seed):
                inputs = self.get_dummy_inputs()
                inputs["output_type"] = "latent"
                outputs.append(pipe(**inputs).frames)
        assert not torch.allclose(outputs[0], outputs[1])

    def test_vae_runs_in_float32(self):
        pipe = self.get_sdr_to_hdr_pipeline().to(dtype=torch.bfloat16)
        seen = {}

        def spy(name, fn):
            def wrapped(x, *args, **kwargs):
                seen[name] = (x.dtype, pipe.vae.dtype)
                return fn(x, *args, **kwargs)

            return wrapped

        with (
            mock.patch.object(pipe.vae, "encode", side_effect=spy("encode", pipe.vae.encode)),
            mock.patch.object(pipe.vae, "decode", side_effect=spy("decode", pipe.vae.decode)),
        ):
            video = pipe(**self.get_sdr_to_hdr_inputs()).frames

        assert seen == {"encode": (torch.float32, torch.float32), "decode": (torch.float32, torch.float32)}
        # The VAE is handed back in the dtype it was loaded in.
        assert pipe.vae.dtype == torch.bfloat16
        assert pipe.transformer.dtype == torch.bfloat16
        assert video.dtype == torch.float32

    def test_diffusion_decoder_runs_in_float32(self):
        pipe = self.get_sdr_to_hdr_pipeline(diffusion_decoder=True).to(dtype=torch.bfloat16)
        decoder = pipe.diffusion_decoder
        seen = {}
        decode = decoder.decode

        def spy(z, *args, **kwargs):
            seen["decode"] = (z.dtype, decoder.dtype)
            return decode(z, *args, **kwargs)

        with mock.patch.object(decoder, "decode", side_effect=spy):
            pipe(**self.get_sdr_to_hdr_inputs())
        assert seen["decode"] == (torch.float32, torch.float32)
        assert decoder.dtype == torch.bfloat16

    def test_reflect_pad_and_crop_back(self):
        pipe = self.get_sdr_to_hdr_pipeline()
        seen = {}
        encode, decode = pipe.vae.encode, pipe.vae.decode

        def encode_spy(x, *args, **kwargs):
            seen["encoded_pixels"] = x.detach().clone()
            return encode(x, *args, **kwargs)

        def decode_spy(*args, **kwargs):
            output = decode(*args, **kwargs)
            seen["decoded_shape"] = tuple(output[0].shape)
            return output

        with (
            mock.patch.object(pipe.vae, "encode", side_effect=encode_spy),
            mock.patch.object(pipe.vae, "decode", side_effect=decode_spy),
        ):
            video = pipe(**self.get_sdr_to_hdr_inputs()).frames

        pixels = seen["encoded_pixels"]
        assert pixels.shape == (1, 3, NUM_FRAMES, PADDED_HEIGHT, PADDED_WIDTH)
        # Bottom/right reflect padding: the padded row and column mirror the ones before the last source row/column.
        assert torch.equal(pixels[..., HEIGHT, :], pixels[..., HEIGHT - 2, :])
        assert torch.equal(pixels[..., :, WIDTH], pixels[..., :, WIDTH - 2])
        # Input transform: 8-bit sRGB maps into ACEScct, so black is not -1 and white is not +1 in VAE range.
        assert pixels.min() >= 2 * 0.0729055341958355 - 1 - 1e-6
        assert pixels.max() <= 2 * 0.5547945205479452 - 1 + 1e-6

        assert seen["decoded_shape"] == (1, 3, NUM_FRAMES, PADDED_HEIGHT, PADDED_WIDTH)
        assert video.shape == (1, NUM_FRAMES, HEIGHT, WIDTH, 3)

        latents = pipe(**self.get_sdr_to_hdr_inputs(output_type="latent")).frames
        assert latents.shape[-2:] == (PADDED_HEIGHT // 2, PADDED_WIDTH // 2)

    @pytest.mark.parametrize("frame_rate, expected_rope_fps", [(24.0, 24.0), (30.0, 30.0), (50.0, 30.0), (60.0, 30.0)])
    def test_rope_frame_rate_is_capped_at_30(self, frame_rate, expected_rope_fps):
        pipe = self.get_sdr_to_hdr_pipeline()
        prepare_video_coords = pipe.transformer.rope.prepare_video_coords
        with mock.patch.object(
            pipe.transformer.rope, "prepare_video_coords", side_effect=prepare_video_coords
        ) as coords_spy:
            pipe(**self.get_sdr_to_hdr_inputs(frame_rate=frame_rate, output_type="latent"))
        # The reference latents and the generated latents share the same time base.
        assert [call.kwargs["fps"] for call in coords_spy.call_args_list] == [expected_rope_fps] * 2

    def test_frame_rates_above_30_condition_like_30(self):
        pipe = self.get_sdr_to_hdr_pipeline()
        outputs = {
            frame_rate: pipe(**self.get_sdr_to_hdr_inputs(frame_rate=frame_rate, output_type="latent")).frames
            for frame_rate in (24.0, 30.0, 60.0)
        }
        assert torch.equal(outputs[30.0], outputs[60.0])
        assert not torch.equal(outputs[24.0], outputs[30.0])

    def test_distilled_sigmas_by_default(self):
        pipe = self.get_sdr_to_hdr_pipeline()
        inputs = self.get_sdr_to_hdr_inputs(output_type="latent")
        inputs.pop("sigmas")
        forward = pipe.transformer.forward
        with mock.patch.object(pipe.transformer, "forward", side_effect=forward) as spy:
            pipe(**inputs)
        # One unguided forward per distilled sigma, each at that exact sigma.
        sigmas = [call.kwargs["sigma"].item() / 1000.0 for call in spy.call_args_list]
        assert sigmas == pytest.approx(DISTILLED_SIGMA_VALUES, abs=1e-6)

    def test_first_latent_frame_is_marked_as_keyframe(self):
        pipe = self.get_sdr_to_hdr_pipeline()
        forward = pipe.transformer.forward
        with mock.patch.object(pipe.transformer, "forward", side_effect=forward) as spy:
            pipe(**self.get_sdr_to_hdr_inputs(output_type="latent"))
        kwargs = spy.call_args.kwargs
        assert kwargs["isolate_modalities"] is True
        mask = kwargs["video_keyframes_mask"]
        latent_frames = (NUM_FRAMES - 1) // 2 + 1
        tokens_per_frame = (PADDED_HEIGHT // 2) * (PADDED_WIDTH // 2)
        # [generated | reference] tokens: only the first latent frame of the generated video is marked. (The dummy
        # VAE only downsamples in space, so its reference latents keep all 5 frames.)
        assert mask.shape == (1, kwargs["hidden_states"].shape[1], 1)
        assert mask.shape[1] > latent_frames * tokens_per_frame
        assert mask[:, :tokens_per_frame].eq(1).all()
        assert mask[:, tokens_per_frame:].eq(0).all()

    def test_hdr_transform_survives_save_load(self, tmp_path):
        pipe = self.get_sdr_to_hdr_pipeline(text_components=True)
        pipe.save_pretrained(str(tmp_path), safe_serialization=False)
        loaded = LTX2HDRPipeline.from_pretrained(str(tmp_path))
        assert loaded.hdr_video_processor.config.hdr_transform == "acescct"

    @pytest.mark.skipif(not is_peft_available(), reason="PEFT is required for LoRA loading.")
    def test_sdr_to_hdr_lora_key_format(self):
        # The LTX-2.5 SDR-To-HDR IC-LoRA stores ComfyUI-style keys: `diffusion_model.transformer_blocks.{i}.` followed
        # by these 20 suffixes (rank 128, no alphas), for the video self-attention, text cross-attention and FFN.
        suffixes = [
            f"{module}.lora_{side}.weight"
            for module in (
                "attn1.to_q",
                "attn1.to_k",
                "attn1.to_v",
                "attn1.to_out.0",
                "attn2.to_q",
                "attn2.to_k",
                "attn2.to_v",
                "attn2.to_out.0",
                "ff.net.0.proj",
                "ff.net.2",
            )
            for side in ("A", "B")
        ]
        pipe = self.get_sdr_to_hdr_pipeline()
        modules = dict(pipe.transformer.named_modules())
        generator = torch.Generator().manual_seed(0)
        state_dict, expected = {}, set()
        for block in range(len(pipe.transformer.transformer_blocks)):
            for suffix in suffixes:
                module_name = f"transformer_blocks.{block}.{suffix.rsplit('.lora_', 1)[0]}"
                linear = modules[module_name]
                shape = (4, linear.in_features) if ".lora_A." in suffix else (linear.out_features, 4)
                state_dict[f"diffusion_model.transformer_blocks.{block}.{suffix}"] = torch.randn(
                    shape, generator=generator
                )
                expected.add(module_name)

        base = pipe(**self.get_sdr_to_hdr_inputs(output_type="latent")).frames
        pipe.load_lora_weights(state_dict, adapter_name="sdr_to_hdr")
        pipe.set_adapters("sdr_to_hdr", 1.0)

        from peft.tuners.tuners_utils import BaseTunerLayer

        adapted = {name for name, module in pipe.transformer.named_modules() if isinstance(module, BaseTunerLayer)}
        assert adapted == expected
        with_lora = pipe(**self.get_sdr_to_hdr_inputs(output_type="latent")).frames
        assert not torch.allclose(base, with_lora)


class TestLTX2HDRPipelineLogC3Unchanged(LTX2HDRPipelineTesterConfig, BasePipelineOutputMixin):
    def test_default_hdr_transform_is_logc3(self):
        pipe = self.get_pipeline()
        assert pipe.hdr_video_processor.config.hdr_transform == "logc3"
        assert pipe.config.hdr_transform == "logc3"

    @pytest.mark.parametrize("argument", ["input_colorspace", "output_colorspace"])
    def test_colorspaces_rejected_with_logc3(self, argument):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs[argument] = "srgb_gamma" if argument == "input_colorspace" else "acescg"
        with pytest.raises(ValueError, match="only supported with `hdr_transform='acescct'`"):
            pipe(**inputs)

    def test_logc3_output_slice(self):
        # Recorded on the parent commit, before the SDR-To-HDR path was added.
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["output_type"] = "latent"
        latents = pipe(**inputs).frames.flatten().cpu()
        expected = torch.tensor(
            [-1.4695, -0.8892, -1.27, 1.4034, 0.5713, 1.7472, 0.2216, 0.9364]
            + [-0.6652, -0.8569, 0.755, 0.1796, 0.1709, -0.1783, -0.1346, -0.4674]
        )
        assert torch.allclose(torch.cat([latents[:8], latents[-8:]]), expected, atol=1e-3)
