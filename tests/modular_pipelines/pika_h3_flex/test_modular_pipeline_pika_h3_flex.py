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

from diffusers.modular_pipelines import (
    PikaH3FlexBlocks,
    PikaH3FlexModularPipeline,
)
from diffusers.modular_pipelines.pika_h3_flex import (
    PikaH3FlexAudioReference,
    PikaH3FlexImageReference,
    PikaH3FlexVideoReference,
)
from diffusers.modular_pipelines.pika_h3_flex.before_encoder import PikaH3FlexRef2VASetupStep, PikaH3FlexResizeStep
from diffusers.modular_pipelines.pika_h3_flex.modular_pipeline import PIKA_H3_FLEX_FPS, resolve_tier_rule

from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularLoraTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


# The blocks every workflow of [`PikaH3FlexBlocks`] runs, in order. A keyframe adds the canvas block and the one
# block that encodes it; whether it anchors the first or the last frame is a matter of the packed layout, not of
# which blocks run, so `fl2va` covers both.
PIKA_H3_FLEX_WORKFLOWS = {
    "t2va": [
        ("text_encoder", "PikaH3FlexTextEncoderStep"),
        ("denoise.no_keyframe_anchors", "PikaH3FlexNoKeyframeAnchorsStep"),
        ("denoise.prepare_layout", "PikaH3FlexPrepareLayoutStep"),
        ("denoise.prepare_latents", "PikaH3FlexPrepareLatentsStep"),
        ("denoise.set_timesteps", "PikaH3FlexSetTimestepsStep"),
        ("denoise.denoise", "PikaH3FlexDenoiseStep"),
        ("denoise.after_denoise", "PikaH3FlexAfterDenoiseStep"),
        ("decode.video", "PikaH3FlexVideoDecodeStep"),
        ("decode.audio", "PikaH3FlexAudioDecodeStep"),
    ],
    "fl2va": [
        ("before_encode", "PikaH3FlexResizeStep"),
        ("text_encoder", "PikaH3FlexFL2VATextEncoderStep"),
        ("vae_encoder", "PikaH3FlexKeyframeVaeEncoderStep"),
        ("denoise.prepare_layout", "PikaH3FlexPrepareLayoutStep"),
        ("denoise.prepare_condition_latents", "PikaH3FlexPrepareConditionLatentsStep"),
        ("denoise.prepare_latents", "PikaH3FlexPrepareLatentsStep"),
        ("denoise.prepare_latents_fl2va", "PikaH3FlexFL2VAPrepareLatentsStep"),
        ("denoise.set_timesteps", "PikaH3FlexSetTimestepsStep"),
        ("denoise.denoise", "PikaH3FlexDenoiseStep"),
        ("denoise.after_denoise", "PikaH3FlexAfterDenoiseStep"),
        ("decode.video", "PikaH3FlexVideoDecodeStep"),
        ("decode.audio", "PikaH3FlexAudioDecodeStep"),
    ],
}
PIKA_H3_FLEX_REF2VA_WORKFLOWS = {
    "ref2va": [
        ("before_encode", "PikaH3FlexRef2VASetupStep"),
        ("text_encoder", "PikaH3FlexRef2VATextEncoderStep"),
        ("vae_encoder", "PikaH3FlexRef2VAReferenceEncoderStep"),
        ("denoise.prepare_layout", "PikaH3FlexRef2VAPrepareLayoutStep"),
        ("denoise.prepare_condition_latents", "PikaH3FlexPrepareConditionLatentsStep"),
        ("denoise.prepare_latents", "PikaH3FlexPrepareLatentsStep"),
        ("denoise.prepare_latents_ref2va", "PikaH3FlexRef2VAPrepareLatentsStep"),
        ("denoise.set_timesteps", "PikaH3FlexSetTimestepsStep"),
        ("denoise.denoise", "PikaH3FlexRef2VADenoiseStep"),
        ("denoise.after_denoise", "PikaH3FlexAfterDenoiseStep"),
        ("decode.video", "PikaH3FlexVideoDecodeStep"),
        ("decode.audio", "PikaH3FlexAudioDecodeStep"),
    ],
}

# What each pruned workflow asks for, pinned explicitly: every component with its class (each half loads its own
# transformer partition, nothing else differs), the pipeline configs it declares, and every input — the optional
# ones with their defaults, the required ones by name. Unlike MiniMax-H3 there is no `num_inference_steps`: the
# six evaluations are the adapters' contract, carried by the schedulers' knots.
_SHARED_COMPONENTS = {
    "text_encoder": "Qwen3VLForConditionalGeneration",
    "tokenizer": "Qwen2Tokenizer",
    "processor": "Qwen3VLProcessor",
    "scheduler": "PikaH3FlexScheduler",
    "audio_scheduler": "PikaH3FlexScheduler",
    "vae": "AutoencoderKLMiniMaxH3",
    "audio_vae": "AutoencoderKLMiniMaxH3Audio",
    "video_processor": "VideoProcessor",
}
# The default canvas, the 768p 16:9 bucket. Every workflow resolves a canvas somewhere — from a keyframe, from the
# bare default — so all three declare the rule as pipeline config; the resolution a `ref2va` reference is
# conditioned at follows the canvas' tier instead and is not a config.
_SHARED_CONFIGS = {"canvas_short_edge": 768, "canvas_max_pixels": 768 * 1344}
_SHARED_INPUTS = {
    "height": None,
    "width": None,
    "generator": None,
    "latents": None,
    "audio_latents": None,
    "attention_kwargs": None,
    "output_type": "pil",
}
PIKA_H3_FLEX_WORKFLOW_DEFAULTS = {
    "t2va": {
        "components": {**_SHARED_COMPONENTS, "transformer": "MiniMaxH3Transformer3DModel"},
        "configs": _SHARED_CONFIGS,
        "required_inputs": ["prompt"],
        "inputs": {**_SHARED_INPUTS, "num_frames": 124},
    },
    "fl2va": {
        "components": {**_SHARED_COMPONENTS, "transformer": "MiniMaxH3Transformer3DModel"},
        "configs": _SHARED_CONFIGS,
        "required_inputs": ["prompt"],
        "inputs": {**_SHARED_INPUTS, "num_frames": 124, "image": None, "last_image": None},
    },
}
PIKA_H3_FLEX_REF2VA_WORKFLOW_DEFAULTS = {
    "ref2va": {
        "components": {
            **_SHARED_COMPONENTS,
            "transformer_ref": "MiniMaxH3Transformer3DModel",
            "image_processor": "VaeImageProcessor",
        },
        "configs": _SHARED_CONFIGS,
        # `num_frames` cannot default on ref2va: reference soundtracks are truncated to the generated duration, so a
        # silent 124-frame default would cut them short. It is required instead.
        "required_inputs": ["prompt", "references", "num_frames"],
        "inputs": _SHARED_INPUTS,
    },
}


class PikaH3FlexModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    """The `t2va` and `fl2va` requests of [`PikaH3FlexBlocks`]: a prompt, optionally with keyframes."""

    pipeline_class = PikaH3FlexModularPipeline
    pipeline_blocks_class = PikaH3FlexBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-pika-h3-flex-modular-pipe"
    expected_latents_shape = (1, 4, 37, 2, 2)
    params = frozenset(["prompt", "image", "last_image", "height", "width", "num_frames"])
    # Pika H3 Flex packs one request into one sequence and rejects a list of prompts, so nothing is batched.
    batch_params = frozenset()
    # The adapters are guidance-distilled and run a fixed six-knot ladder, so neither `guidance_scale` nor
    # `num_inference_steps` exists, and one request generates one video.
    optional_params = frozenset(["latents", "output_type"])
    not_params = frozenset(["num_inference_steps", "guidance_scale", "negative_prompt"])
    output_name = "videos"
    expected_workflow_blocks = PIKA_H3_FLEX_WORKFLOWS
    expected_workflow_defaults = PIKA_H3_FLEX_WORKFLOW_DEFAULTS

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a robot dancing",
            "generator": self.get_generator(seed),
            # Pika H3 Flex generates 5 to 15 seconds at a fixed 24 fps, so 124 frames (`17 * 7 + 5`, the next length
            # the video VAE can decode) is the shortest admissible request; a 32 pixel canvas is one `(1, 2, 2)`
            # patch row per latent frame, which is what makes it affordable on CPU.
            "height": 32,
            "width": 32,
            "num_frames": 124,
            "output_type": "pt",
        }


class TestPikaH3FlexModularPipelineFast(PikaH3FlexModularPipelineTesterConfig, ModularPipelineTesterMixin):
    @pytest.mark.skip(reason="Pika H3 Flex packs one request into one sequence, so a batch of prompts is not a thing.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="Pika H3 Flex packs one request into one sequence, so a batch of prompts is not a thing.")
    def test_inference_batch_single_identical(self):
        pass

    @pytest.mark.skip(reason="One request generates one video, so there is no `num_images_per_prompt`.")
    def test_num_images_per_prompt(self):
        pass

    def test_six_evaluations_from_the_knots(self):
        r"""
        The ladder is the adapters' contract: six evaluations over the released knots, whatever the request says. There
        is no `num_inference_steps` input to say otherwise, and both schedulers expose the same count.
        """
        pipe = self.get_pipeline()

        state = pipe(**self.get_dummy_inputs())

        assert "num_inference_steps" not in pipe.blocks.input_names
        assert state.get("timesteps").shape == (6,)
        assert state.get("audio_timesteps").shape == (6,)
        assert len(state.get("row_timestep_plan")) == 6
        assert pipe.scheduler.sigmas.tolist()[-1] == 0.0

    def test_video_and_audio_outputs(self):
        r"""One call denoises both modalities out of the one packed sequence, and returns both."""
        pipe = self.get_pipeline()

        state = pipe(**self.get_dummy_inputs())
        video, audio, sampling_rate = state.get("videos"), state.get("audio"), state.get("sampling_rate")

        assert video.shape == (1, 124, 3, 32, 32)
        # The audio VAE is mono and takes the two stereo channels as two batch items, which the decoder block stacks
        # back into `(batch, 2, samples)`.
        # 207 audio latents at the tiny VAE's 4 samples per latent
        assert audio.shape == (1, 2, 207 * 4)
        assert video.min() >= 0.0 and video.max() <= 1.0
        assert sampling_rate == pipe.audio_vae.config.sampling_rate

    @pytest.mark.parametrize(
        "keyframes,num_condition_rows",
        [(("image",), 1), (("last_image",), 1), (("image", "last_image"), 2)],
        ids=["first", "last", "first_and_last"],
    )
    def test_fl2va_keyframes(self, keyframes, num_condition_rows):
        r"""
        A keyframe contributes one conditioning row per latent patch, and the layout reserves exactly those rows.

        They are packed in front of the generated video rows and ride along at their own `t = 0.999`, pinned for the
        whole loop, and the after-denoise step drops them again — so the request comes back with the generated frames
        alone, whichever end the keyframes anchor. On a `32` canvas a keyframe is a single `(1, 2, 2)` patch row.
        """
        pipe = self.get_pipeline()
        keyframe = Image.fromarray((np.random.default_rng(0).random((48, 80, 3)) * 255).astype("uint8"))

        inputs = self.get_dummy_inputs()
        inputs.update(dict.fromkeys(keyframes, keyframe))
        state = pipe(**inputs)

        assert state.get("videos").shape == (1, 124, 3, 32, 32)
        assert state.get("num_condition_video_rows") == num_condition_rows

    @pytest.mark.parametrize("with_keyframe", [False, True], ids=["t2va", "fl2va"])
    def test_generator_reproducibility(self, with_keyframe):
        r"""
        Two runs from the same generator state are identical, two seeds differ.

        The blocks draw from the one generator the request carries, in the order they run — a keyframe's conditioning
        noise in the prepare-condition-latents step, the video and the audio noise in the prepare-latents step, and the
        fresh noise of every re-noising evaluation in the loop — which is the whole reproducibility contract of a
        request.
        """
        pipe = self.get_pipeline()
        keyframe = Image.new("RGB", (48, 80))

        def run(seed):
            inputs = self.get_dummy_inputs()
            inputs["generator"] = torch.Generator("cpu").manual_seed(seed)
            if with_keyframe:
                inputs["image"] = keyframe
            state = pipe(**inputs)
            return state.get("videos"), state.get("audio")

        video, audio = run(7)
        same_video, same_audio = run(7)
        other_video, other_audio = run(8)

        assert torch.equal(video, same_video)
        assert torch.equal(audio, same_audio)
        assert not torch.equal(video, other_video)
        assert not torch.equal(audio, other_audio)

    def test_injected_latents_replace_the_initial_draws_only(self):
        r"""
        `latents` and `audio_latents` stand in for the two initial draws, and only for those.

        Unlike MiniMax-H3's Euler loop, every evaluation of the ladder but the last re-noises with fresh noise drawn
        from the request's generator, so injecting the initial noise does not make a `t2va` request independent of its
        seed: the same seed with other initial noise is another sample, and another seed with the same initial noise is
        still another sample.
        """
        pipe = self.get_pipeline()
        # the noise the request would otherwise draw, in the shapes the tiny checkpoint expects back: 4 latent
        # channels over 37 latent frames on the 32/16 canvas, and 8 audio latent channels over 207 latents
        latents = torch.randn(1, 4, 37, 2, 2, generator=torch.Generator("cpu").manual_seed(3))
        audio_latents = torch.randn(2, 8, 207, generator=torch.Generator("cpu").manual_seed(4))
        other_latents = torch.randn(1, 4, 37, 2, 2, generator=torch.Generator("cpu").manual_seed(11))
        other_audio_latents = torch.randn(2, 8, 207, generator=torch.Generator("cpu").manual_seed(12))

        def run(seed, video_noise, audio_noise):
            inputs = self.get_dummy_inputs()
            inputs["generator"] = torch.Generator("cpu").manual_seed(seed)
            inputs["latents"] = video_noise
            inputs["audio_latents"] = audio_noise
            state = pipe(**inputs)
            return state.get("videos"), state.get("audio")

        video, audio = run(7, latents, audio_latents)
        same_video, same_audio = run(7, latents, audio_latents)
        other_noise_video, other_noise_audio = run(7, other_latents, other_audio_latents)
        other_seed_video, other_seed_audio = run(8, latents, audio_latents)

        assert torch.equal(video, same_video)
        assert torch.equal(audio, same_audio)
        assert not torch.equal(video, other_noise_video)
        assert not torch.equal(audio, other_noise_audio)
        assert not torch.equal(video, other_seed_video)
        assert not torch.equal(audio, other_seed_audio)

    @pytest.mark.parametrize("output_type", ["np", "pil", "pt"])
    def test_output_type(self, output_type):
        r"""
        The three formats the video decoder postprocesses to. `output_type` is a video format and nothing else — a
        request that wants latents instead runs a pipeline without the decode blocks, which is what the audio decoder
        does whatever is asked of it, since a waveform has only the one representation.
        """
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["output_type"] = output_type
        state = pipe(**inputs)
        video, audio = state.get("videos"), state.get("audio")

        if output_type == "np":
            assert video.shape == (1, 124, 32, 32, 3)
        elif output_type == "pil":
            assert len(video[0]) == 124
            assert video[0][0].size == (32, 32)
        else:
            assert video.shape == (1, 124, 3, 32, 32)
        assert audio.shape == (1, 2, 207 * 4)

    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"prompt": ["a robot", "a fox"]}, "must be a single string"),
            ({"height": 30, "width": 30}, "multiples of 32"),
            ({"width": None}, "have to be passed together"),
            ({"num_frames": 96}, "must be between"),
            ({"num_frames": 400}, "must be between"),
            ({"output_type": "latent"}, "must be one of 'pil', 'np' or 'pt'"),
        ],
        ids=[
            "prompt_list",
            "canvas_not_a_multiple_of_32",
            "height_without_width",
            "shorter_than_five_seconds",
            "longer_than_fifteen_seconds",
            "output_type_latent",
        ],
    )
    def test_check_inputs(self, overrides, message):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs.update(overrides)

        with pytest.raises(ValueError, match=message):
            pipe(**inputs)


class TestPikaH3FlexModularPipelineLoading(PikaH3FlexModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestPikaH3FlexModularPipelineWorkflow(PikaH3FlexModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestPikaH3FlexModularPipelineMemory(PikaH3FlexModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


class TestPikaH3FlexModularPipelineLoRA(PikaH3FlexModularPipelineTesterConfig, ModularLoraTesterMixin):
    """The generic LoRA behavior over both MiniMax-H3 transformer partitions, which are identically named inside."""

    denoiser_target_modules = {
        "transformer": ["to_q", "to_k", "to_v", "to_out.0"],
        "transformer_ref": ["to_q", "to_k", "to_v", "to_out.0"],
    }


class PikaH3FlexRef2VAModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    """The `ref2va` requests of [`PikaH3FlexBlocks`]: a prompt and an ordered list of references."""

    pipeline_class = PikaH3FlexModularPipeline
    pipeline_blocks_class = PikaH3FlexBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-pika-h3-flex-modular-pipe"
    expected_latents_shape = (1, 4, 37, 2, 2)
    params = frozenset(["prompt", "references", "height", "width", "num_frames"])
    # Pika H3 Flex packs one request into one sequence and rejects a list of prompts, so nothing is batched.
    batch_params = frozenset()
    # The adapters are guidance-distilled and run a fixed six-knot ladder, so neither `guidance_scale` nor
    # `num_inference_steps` exists, and one request generates one video.
    optional_params = frozenset(["latents", "output_type"])
    not_params = frozenset(["num_inference_steps", "guidance_scale", "negative_prompt"])
    output_name = "videos"
    expected_workflow_blocks = PIKA_H3_FLEX_REF2VA_WORKFLOWS
    expected_workflow_defaults = PIKA_H3_FLEX_REF2VA_WORKFLOW_DEFAULTS

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a robot dancing",
            # A reference is conditioned at the resolution of the canvas' tier — a 1024 pixel short edge for the 32
            # pixel canvas below, which is not configurable — so the reference is kept tiny to keep its 1024x1024
            # encode cheap on CPU.
            "references": [PikaH3FlexImageReference(image=Image.new("RGB", (16, 16)))],
            "generator": self.get_generator(seed),
            # Pika H3 Flex generates 5 to 15 seconds at a fixed 24 fps, so 124 frames (`17 * 7 + 5`, the next length
            # the video VAE can decode) is the shortest admissible request; a 32 pixel canvas is one `(1, 2, 2)`
            # patch row per latent frame, which is what makes it affordable on CPU.
            "height": 32,
            "width": 32,
            "num_frames": 124,
            "output_type": "pt",
        }


class TestPikaH3FlexRef2VAModularPipelineFast(PikaH3FlexRef2VAModularPipelineTesterConfig, ModularPipelineTesterMixin):
    @pytest.mark.skip(reason="Pika H3 Flex packs one request into one sequence, so a batch of prompts is not a thing.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="Pika H3 Flex packs one request into one sequence, so a batch of prompts is not a thing.")
    def test_inference_batch_single_identical(self):
        pass

    @pytest.mark.skip(reason="One request generates one video, so there is no `num_images_per_prompt`.")
    def test_num_images_per_prompt(self):
        pass

    def test_video_and_audio_outputs(self):
        r"""A reference conditions the request without binding the generated geometry."""
        pipe = self.get_pipeline()

        state = pipe(**self.get_dummy_inputs())
        video, audio = state.get("videos"), state.get("audio")

        assert video.shape == (1, 124, 3, 32, 32)
        # 207 audio latents at the tiny VAE's 4 samples per latent
        assert audio.shape == (1, 2, 207 * 4)
        assert video.min() >= 0.0 and video.max() <= 1.0
        # a 16x16 reference on the 32 pixel canvas' tier: 1024x1024, i.e. a 64x64 latent packed into 32 * 32 rows
        assert state.get("normalized_references")[0].image.size == (1024, 1024)
        assert state.get("num_condition_video_rows") == 32 * 32

    def test_generator_reproducibility(self):
        r"""
        Two runs from the same generator state are identical, two seeds differ.

        A `ref2va` request draws the reference conditioning noise before the video and the audio noise, and the fresh
        noise of every re-noising evaluation after them, all off the one generator it carries, so this covers the whole
        stream.
        """
        pipe = self.get_pipeline()

        def run(seed):
            inputs = self.get_dummy_inputs()
            inputs["generator"] = torch.Generator("cpu").manual_seed(seed)
            state = pipe(**inputs)
            return state.get("videos"), state.get("audio")

        video, audio = run(7)
        same_video, same_audio = run(7)
        other_video, other_audio = run(8)

        assert torch.equal(video, same_video)
        assert torch.equal(audio, same_audio)
        assert not torch.equal(video, other_video)
        assert not torch.equal(audio, other_audio)

    def test_reference_combinations(self):
        r"""
        Any ordered mix of image, video and audio references is packed, in request order.

        A video reference conditions on its motion *and*, when the request passes one with it, on its soundtrack, so it
        contributes both visual and audio rows; an audio reference contributes audio rows alone and never reaches the
        conditioner. The video reference is put on the canvas' tier rule — the 480p tier's 480 pixel short edge for the
        32 pixel canvas — which is what makes it the expensive reference of the suite.

        The reference rows are pinned for the whole loop and the after-denoise step drops them again, so what a request
        comes back with is the generated frames and soundtrack alone, however many rows the references took.
        """
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        # Two seconds of video, the shortest reference the released package accepts, soundtrack included, and a six
        # second standalone soundtrack. Both waveforms are at 160 Hz, the tiny audio VAE's own rate, so nothing is
        # ever resampled.
        inputs["references"] = [
            PikaH3FlexImageReference(image=Image.new("RGB", (16, 16))),
            PikaH3FlexVideoReference(
                frames=(np.random.default_rng(0).random((2 * PIKA_H3_FLEX_FPS, 64, 64, 3)) * 255).astype("uint8"),
                fps=float(PIKA_H3_FLEX_FPS),
                audio=torch.rand(2, 2 * 160, generator=torch.Generator("cpu").manual_seed(1)) * 2 - 1,
                sample_rate=160,
            ),
            PikaH3FlexAudioReference(
                audio=torch.rand(2, 6 * 160, generator=torch.Generator("cpu").manual_seed(1)) * 2 - 1,
                sample_rate=160,
            ),
        ]

        state = pipe(**inputs)
        references = state.get("normalized_references")

        assert references[1].frames.shape[1:3] == (480, 480)
        assert state.get("num_condition_video_rows") > 32 * 32
        assert state.get("num_condition_audio_rows") > 0
        assert state.get("videos").shape == (1, state.get("num_frames"), 3, 32, 32)
        assert state.get("audio").shape[:2] == (1, 2)

    def test_soundtrack_truncated_to_the_generated_duration(self):
        r"""
        A soundtrack is resampled onto the audio VAE's rate first and truncated there to the samples that cover the
        generated frames, `round(num_frames / 24 * sample_rate)`, as the released package does.
        """
        pipe = self.get_pipeline()
        waveform = torch.rand(2, 15 * 160, generator=torch.Generator("cpu").manual_seed(1)) * 2 - 1

        inputs = self.get_dummy_inputs()
        inputs["references"] = [
            PikaH3FlexImageReference(image=Image.new("RGB", (16, 16))),
            PikaH3FlexAudioReference(audio=waveform),
        ]
        inputs["num_frames"] = 158
        state = pipe(**inputs)
        references = state.get("normalized_references")

        assert references[1].audio.shape == (2, round(158 / 24 * 160))

    @pytest.mark.parametrize(
        "references,message",
        [
            (["a-photo.png"], "never open media files"),
            (
                [
                    PikaH3FlexVideoReference(
                        frames=np.zeros((48, 8, 8, 3), dtype="uint8"), fps=float(PIKA_H3_FLEX_FPS)
                    )
                    for _ in range(4)
                ],
                "at most 3 video references",
            ),
            (
                [PikaH3FlexAudioReference(audio=torch.zeros(2, 2 * 160), sample_rate=160)],
                "cannot be used on its own",
            ),
            ([PikaH3FlexImageReference(image=Image.new("RGB", (20, 100)))], "within 1:4 and 4:1"),
            (
                [
                    PikaH3FlexImageReference(image=Image.new("RGB", (16, 16))),
                    PikaH3FlexVideoReference(
                        frames=np.zeros((24, 8, 8, 3), dtype="uint8"), fps=float(PIKA_H3_FLEX_FPS)
                    ),
                ],
                "must run 2 to 15 seconds",
            ),
            (
                [
                    PikaH3FlexImageReference(image=Image.new("RGB", (16, 16))),
                    PikaH3FlexAudioReference(audio=torch.zeros(2, 20 * 160), sample_rate=160),
                ],
                "must run 2 to 15 seconds",
            ),
            (
                [PikaH3FlexImageReference(image=Image.new("RGB", (16, 16)))]
                + [PikaH3FlexAudioReference(audio=torch.zeros(2, 8 * 160), sample_rate=160) for _ in range(2)],
                "at most 15 seconds in total",
            ),
        ],
        ids=[
            "not_a_reference",
            "too_many_videos",
            "audio_only",
            "image_aspect_ratio",
            "video_too_short",
            "audio_too_long",
            "audio_total_too_long",
        ],
    )
    def test_check_inputs_references(self, references, message):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["references"] = references

        with pytest.raises(ValueError, match=message):
            pipe(**inputs)


class TestPikaH3FlexRef2VAModularPipelineLoading(
    PikaH3FlexRef2VAModularPipelineTesterConfig, ModularLoadingTesterMixin
):
    pass


class TestPikaH3FlexRef2VAModularPipelineWorkflow(
    PikaH3FlexRef2VAModularPipelineTesterConfig, ModularWorkflowTesterMixin
):
    pass


class TestPikaH3FlexRef2VAModularPipelineMemory(PikaH3FlexRef2VAModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


class TestPikaH3FlexModularBlocks:
    """
    The blocks that differ from MiniMax-H3, run on their own rather than through a full request: each is built into a
    one-block pipeline with `init_pipeline()`, so what is pinned here is the block's own contract — the keyframe crop,
    the fl2va conditioning precision and the tier rule a reference is encoded under — at the released geometry rather
    than the shrunken one the pipeline tests use. None of them needs a checkpoint.
    """

    def test_keyframes_are_cropped_onto_the_canvas(self):
        r"""
        A keyframe is center-cropped to the canvas' aspect ratio at its own resolution and then resized, the way the
        released package fits it — never stretched (MiniMax-H3's rule for the first keyframe) nor cover-cropped after
        resizing. A portrait keyframe whose top fifth is red loses that band entirely on a square canvas: the crop
        keeps the middle 48 of its 80 rows, and the red stops at row 16.
        """
        pixels = np.zeros((80, 48, 3), dtype="uint8")
        pixels[:16] = (255, 0, 0)
        pixels[16:] = (0, 0, 255)
        pipe = PikaH3FlexResizeStep().init_pipeline()

        keyframes = pipe(image=Image.fromarray(pixels), height=32, width=32, output="keyframes")

        assert keyframes[0].size == (32, 32)
        assert np.asarray(keyframes[0])[..., 0].max() == 0

    def test_keyframe_canvas_defaults_to_its_aspect_ratio(self):
        r"""
        Without `height` and `width`, the canvas is the 768 short-edge geometry of the first keyframe's own aspect
        ratio — the 768p bucket of that ratio — and both keyframes are fitted onto it.
        """
        pipe = PikaH3FlexResizeStep().init_pipeline()

        state = pipe(image=Image.new("RGB", (160, 90)), last_image=Image.new("RGB", (90, 160)))

        assert (state.get("height"), state.get("width")) == (768, 1344)
        assert [keyframe.size for keyframe in state.get("keyframes")] == [(1344, 768), (1344, 768)]
        assert state.get("keyframe_anchors") == ("first", "last")

    def test_reference_image_resolution_follows_the_canvas_tier(self, canvas, expected_size):
        r"""
        An image reference is encoded at the reference short edge of the tier the *generated* canvas falls in — 1024
        pixels for the 480p buckets, 2048 for the 768p ones — both axes rounded to a multiple of 32, with no area cap.
        """
        height, width = canvas
        pipe = PikaH3FlexRef2VASetupStep().init_pipeline()

        references = pipe(
            references=[PikaH3FlexImageReference(image=Image.new("RGB", (80, 48)))],
            height=height,
            width=width,
            num_frames=124,
            output="normalized_references",
        )

        assert resolve_tier_rule(height, width).reference_short_edge == expected_size[1]
        assert references[0].image.size == expected_size

    def test_video_reference_follows_the_canvas_tier_rule(self):
        r"""
        A video reference is put on the canvas rule of the generated canvas' tier, at its own aspect ratio: the 480p
        tier's 480 pixel short edge under its 832x480 area cap for a 480p request. Its soundtrack is truncated to the
        samples that cover the generated frames.
        """
        pipe = PikaH3FlexRef2VASetupStep().init_pipeline()
        frames = np.zeros((3 * PIKA_H3_FLEX_FPS, 90, 160, 3), dtype="uint8")
        waveform = torch.zeros(2, 10 * 32000)

        references = pipe(
            references=[PikaH3FlexVideoReference(frames=frames, fps=float(PIKA_H3_FLEX_FPS), audio=waveform)],
            height=480,
            width=832,
            num_frames=124,
            output="normalized_references",
        )

        assert references[0].frames.shape == (3 * PIKA_H3_FLEX_FPS, 480, 832, 3)
        assert references[0].audio.shape == (2, round(124 / 24 * 32000))
