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
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from diffusers import (
    AutoencoderKLMiniMaxH3,
    AutoencoderKLMiniMaxH3Audio,
    MiniMaxH3Pipeline,
    MiniMaxH3Scheduler,
    MiniMaxH3Transformer3DModel,
)

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


# MiniMax-H3 generates 5 to 15 seconds at a fixed 24 fps, so the shortest admissible request is 124 frames
# (`17 * 7 + 5`, the next length the video VAE can decode). It is affordable on CPU because the canvas is tiny: 32
# pixels is a single `(1, 2, 2)` patch row per latent frame.
NUM_FRAMES = 124
RESOLUTION = 32
NUM_LATENT_FRAMES = 37
NUM_AUDIO_LATENTS = 207
# `prod(encoder_rates)` of the dummy audio VAE, standing in for the released 800 samples per latent.
AUDIO_HOP_LENGTH = 4

# The tokenizer of the released conditioner, for its `<|vision_start|>` / `<|image_pad|>` / `<|vision_end|>` ids.
TINY_QWEN_CKPT_ID = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"


class MiniMaxH3PipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = MiniMaxH3Pipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "image", "last_image", "height", "width", "num_frames", "prompt_embeds"]
    )
    # MiniMax-H3 packs one request into one sequence and rejects a list of prompts, so nothing is batched.
    batch_input_params = frozenset()
    # The checkpoint is guidance-distilled and generates a video, so neither `guidance_scale` nor
    # `num_images_per_prompt` exists.
    optional_input_params = frozenset(
        ["num_inference_steps", "generator", "latents", "audio_latents", "output_type", "return_dict"]
    )

    def get_dummy_components(self):
        torch.manual_seed(0)
        vae = AutoencoderKLMiniMaxH3(
            latent_channels=4,
            block_out_channels=(8, 16),
            layers_per_block=1,
            spatial_downsample_factors=(4, 4),
            temporal_downsample_factors=(2, 2),
            norm_num_groups=8,
            decoder_num_layers=2,
            decoder_num_attention_heads=2,
            decoder_attention_head_dim=8,
            decoder_num_register_tokens=2,
            decoder_ffn_mult=2,
            latents_mean=(0.0,) * 4,
            latents_std=(1.0,) * 4,
        )

        torch.manual_seed(0)
        audio_vae = AutoencoderKLMiniMaxH3Audio(
            encoder_dim=4,
            encoder_rates=(2, 2),
            latent_dim=32,
            latent_channels=8,
            num_attention_heads=2,
            decoder_dim=16,
            decoder_rates=(2, 2),
            decoder_kernel_sizes=(4, 4),
            resblock_kernel_sizes=(3, 7),
            resblock_dilation_sizes=((1, 3), (1, 3)),
            sampling_rate=40 * AUDIO_HOP_LENGTH,
            latents_mean=[0.0] * 8,
            latents_std=[1.0] * 8,
        )

        torch.manual_seed(0)
        transformer = MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            hidden_size=24,
            num_layers=2,
            num_refiner_layers=2,
            ffn_dim=32,
            in_channels=4,
            audio_in_channels=8,
            patch_size=(1, 2, 2),
            text_dim=16,
            freq_dim=8,
            time_embed_hidden_dim=24,
            time_embed_dim=16,
            rope_freq_dim=2,
        )

        torch.manual_seed(0)
        # MiniMax-H3 reads `hidden_states[50]` of its Qwen3-VL conditioner, so the dummy conditioner needs more than
        # 50 decoder layers even though every layer is tiny.
        text_encoder = Qwen3VLForConditionalGeneration(
            Qwen3VLConfig(
                text_config={
                    "hidden_size": 16,
                    "intermediate_size": 16,
                    "num_hidden_layers": 51,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "head_dim": 8,
                    "rope_scaling": {"mrope_section": [1, 1, 2], "rope_type": "default", "type": "default"},
                },
                vision_config={
                    "depth": 2,
                    "hidden_size": 16,
                    "intermediate_size": 16,
                    "num_heads": 2,
                    "out_hidden_size": 16,
                    "patch_size": 4,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "num_position_embeddings": 64,
                    "deepstack_visual_indexes": [0],
                },
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(TINY_QWEN_CKPT_ID)
        processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessorFast(
                patch_size=4, merge_size=2, temporal_patch_size=2, min_pixels=64, max_pixels=256
            ),
            tokenizer=tokenizer,
            video_processor=Qwen3VLVideoProcessor(
                patch_size=4, merge_size=2, temporal_patch_size=2, min_pixels=64, max_pixels=256
            ),
        )

        return {
            "transformer": transformer,
            "vae": vae,
            "audio_vae": audio_vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "processor": processor,
            "scheduler": MiniMaxH3Scheduler(shift=12.0),
            "audio_scheduler": MiniMaxH3Scheduler(shift=3.0),
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "a robot dancing",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "height": RESOLUTION,
            "width": RESOLUTION,
            "num_frames": NUM_FRAMES,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestMiniMaxH3Pipeline(MiniMaxH3PipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        pipe = self.get_pipeline()

        output = pipe(**self.get_dummy_inputs())
        video, audio = output.frames, output.audio

        assert video.shape == (1, NUM_FRAMES, 3, RESOLUTION, RESOLUTION)
        # The audio VAE is mono and takes the two stereo channels as two batch items, which the pipeline stacks back
        # into `(batch, 2, samples)`.
        assert audio.shape == (1, 2, NUM_AUDIO_LATENTS * AUDIO_HOP_LENGTH)
        assert video.min() >= 0.0 and video.max() <= 1.0
        assert torch.isfinite(audio).all()

    @pytest.mark.parametrize("num_keyframes", [0, 1, 2], ids=["text_only", "one_keyframe", "two_keyframes"])
    def test_encode_prompt(self, num_keyframes):
        r"""
        The presentation is encoded in one conditioner call, whatever it carries.

        A keyframe prepends a `"<Picture i>: "` label and a vision block, whose rows are tagged as *video*; a
        text-only presentation is the verbatim prompt and is tagged as text throughout. Every vision block also makes
        Qwen3-VL lay its rotary positions out per modality, which it reads off the `mm_token_type_ids` the processor
        derives from the vision pad ids.
        """
        pipe = self.get_pipeline()
        keyframe = Image.fromarray((np.random.default_rng(0).random((32, 32, 3)) * 255).astype("uint8"))

        prompt_embeds, text_token_tags = pipe.encode_prompt(
            "a robot dancing", [keyframe] * num_keyframes, device=torch_device
        )

        assert prompt_embeds.shape[0] == 1
        assert prompt_embeds.shape[-1] == pipe.transformer.config.text_dim
        assert prompt_embeds.shape[1] == text_token_tags.shape[0]
        assert torch.isfinite(prompt_embeds).all()
        # `0` tags a row of a vision block and `1` a text row, so a text-only presentation carries text rows alone.
        assert set(text_token_tags.tolist()) == ({0, 1} if num_keyframes else {1})

    def test_prompt_embeds(self):
        r"""An encoded presentation can be passed back in, which is how a caller reuses one across requests."""
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        prompt_embeds, text_token_tags = pipe.encode_prompt(inputs.pop("prompt"), device=torch_device)
        inputs["prompt_embeds"] = prompt_embeds
        inputs["text_token_tags"] = text_token_tags
        video = pipe(**inputs).frames

        assert video.shape == (1, NUM_FRAMES, 3, RESOLUTION, RESOLUTION)

    @pytest.mark.parametrize(
        "keyframes,num_condition_rows",
        [(("image",), 1), (("last_image",), 1), (("image", "last_image"), 2)],
        ids=["first", "last", "first_and_last"],
    )
    def test_fl2va_keyframes(self, keyframes, num_condition_rows):
        r"""
        A keyframe contributes one conditioning row per latent patch, and those rows are pinned for the whole loop.

        They are packed in front of the generated video rows and ride along at their own `t = 0.999`, so the loop
        never writes them. On a `RESOLUTION` canvas a keyframe is a single `(1, 2, 2)` patch row.
        """
        pipe = self.get_pipeline()
        keyframe = Image.fromarray((np.random.default_rng(0).random((48, 80, 3)) * 255).astype("uint8"))

        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 3
        inputs.update(dict.fromkeys(keyframes, keyframe))

        seen = []
        inputs["callback_on_step_end"] = lambda pipeline, i, t, kwargs: seen.append(kwargs["latents"].clone()) or {}
        video = pipe(**inputs).frames

        assert video.shape == (1, NUM_FRAMES, 3, RESOLUTION, RESOLUTION)
        # `num_inference_steps` counts sigma grid points, the terminal `0` included, so it drives one model
        # evaluation less.
        assert len(seen) == inputs["num_inference_steps"] - 1
        for latents in seen:
            assert torch.equal(latents[:num_condition_rows], seen[0][:num_condition_rows])
        assert not torch.equal(seen[-1][num_condition_rows:], seen[0][num_condition_rows:])

    @pytest.mark.parametrize("with_keyframe", [False, True], ids=["t2va", "fl2va"])
    def test_generator_reproducibility(self, with_keyframe):
        r"""
        Two runs from the same generator state are identical, two seeds differ.

        Every noise draw of a request — the keyframe conditioning noise, then the video noise, then the audio noise —
        comes off the one generator the request carries, so this is the whole reproducibility contract.
        """
        pipe = self.get_pipeline()
        keyframe = Image.fromarray((np.random.default_rng(0).random((48, 80, 3)) * 255).astype("uint8"))

        def run(seed):
            inputs = self.get_dummy_inputs()
            inputs["generator"] = torch.Generator("cpu").manual_seed(seed)
            if with_keyframe:
                inputs["image"] = keyframe
            output = pipe(**inputs)
            return output.frames, output.audio

        frames, audio = run(7)
        same_frames, same_audio = run(7)
        other_frames, other_audio = run(8)

        assert torch.equal(frames, same_frames)
        assert torch.equal(audio, same_audio)
        assert not torch.equal(frames, other_frames)
        assert not torch.equal(audio, other_audio)

    def test_injected_latents_replace_the_draws(self):
        r"""
        `latents` and `audio_latents` stand in for their draw, which is how a sample is reproduced from outside.

        With both passed in, a `t2va` request draws nothing at all, so two runs with different generators return the
        very same video and soundtrack.
        """
        pipe = self.get_pipeline()
        latents = torch.randn(
            1,
            pipe.vae_latent_channels,
            NUM_LATENT_FRAMES,
            RESOLUTION // pipe.vae_spatial_compression_ratio,
            RESOLUTION // pipe.vae_spatial_compression_ratio,
            generator=torch.Generator("cpu").manual_seed(3),
        )
        audio_latents = torch.randn(
            2, pipe.audio_latent_channels, NUM_AUDIO_LATENTS, generator=torch.Generator("cpu").manual_seed(4)
        )

        outputs = []
        for seed in (7, 8):
            inputs = self.get_dummy_inputs()
            inputs["generator"] = torch.Generator("cpu").manual_seed(seed)
            inputs["latents"] = latents
            inputs["audio_latents"] = audio_latents
            output = pipe(**inputs)
            outputs.append((output.frames, output.audio))

        assert torch.equal(outputs[0][0], outputs[1][0])
        assert torch.equal(outputs[0][1], outputs[1][1])

    @pytest.mark.parametrize("output_type", ["np", "pil", "latent"])
    def test_output_type(self, output_type):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["output_type"] = output_type
        output = pipe(**inputs)

        if output_type == "np":
            assert output.frames.shape == (1, NUM_FRAMES, RESOLUTION, RESOLUTION, 3)
        elif output_type == "pil":
            assert len(output.frames[0]) == NUM_FRAMES
            assert output.frames[0][0].size == (RESOLUTION, RESOLUTION)
        else:
            # `"latent"` keeps the denormalized latents: the video as `(1, C, F, H, W)` and the audio channel-major.
            assert output.frames.shape == (1, 4, NUM_LATENT_FRAMES, RESOLUTION // 16, RESOLUTION // 16)
            assert output.audio.shape == (2, 8, NUM_AUDIO_LATENTS)

    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"prompt_embeds": torch.zeros(1, 4, 16)}, "not both"),
            ({"prompt": None}, "Pass one of"),
            ({"prompt": ["a robot", "a fox"]}, "must be a single string"),
            ({"height": 30, "width": 30}, "multiples of 32"),
            ({"width": None}, "have to be passed together"),
            ({"num_frames": 96}, "must be between"),
            ({"num_frames": 400}, "must be between"),
            # 346 frames are 14.417 seconds, but they are rounded up to 362, i.e. 15.083 seconds: the ceiling holds
            # for the aligned count, so this is rejected rather than silently generating too long a video.
            ({"num_frames": 346}, "rounded up to 362"),
        ],
        ids=[
            "prompt_and_prompt_embeds",
            "neither_prompt_nor_embeds",
            "prompt_list",
            "canvas_not_a_multiple_of_32",
            "height_without_width",
            "shorter_than_five_seconds",
            "longer_than_fifteen_seconds",
            "longer_than_fifteen_seconds_once_aligned",
        ],
    )
    def test_check_inputs(self, overrides, message):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs.update(overrides)

        with pytest.raises(ValueError, match=message):
            pipe(**inputs)

    @pytest.mark.skip(
        "MiniMax-H3 packs one request into one sequence: `prompt` must be a single string and the transformer's batch "
        "axis is a pure replication axis, so there is nothing to batch."
    )
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(
        "MiniMax-H3 packs one request into one sequence: `prompt` must be a single string and the transformer's batch "
        "axis is a pure replication axis, so there is nothing to batch."
    )
    def test_inference_batch_single_identical(self):
        pass

    @pytest.mark.skip(
        "`encode_prompt` returns the per-row modality tags next to the embeddings, and both depend on the keyframes "
        "of the request, so the harness cannot reconstruct its call from the pipeline arguments alone."
    )
    def test_encode_prompt_works_in_isolation(self):
        pass


class TestMiniMaxH3PipelineMemory(MiniMaxH3PipelineTesterConfig, MemoryTesterMixin):
    pass
