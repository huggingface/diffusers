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
    MiniMaxH3Blocks,
    MiniMaxH3ModularPipeline,
)
from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)
from diffusers.modular_pipelines.minimax_h3 import modular_pipeline as minimax_h3_geometry
from diffusers.modular_pipelines.minimax_h3 import (
    references as minimax_h3_references,
)
from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep, MiniMaxH3TextEncoderStep
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MINIMAX_H3_FPS

from ..test_modular_pipelines_common import ModularPipelineTesterMixin, _get_specified_components


# MiniMax-H3 generates 5 to 15 seconds at a fixed 24 fps, so the shortest admissible request is 124 frames
# (`17 * 7 + 5`, the next length the video VAE can decode). It is affordable on CPU because the canvas is tiny: 32
# pixels is a single `(1, 2, 2)` patch row per latent frame.
NUM_FRAMES = 124
RESOLUTION = 32
NUM_LATENT_FRAMES = 37
NUM_AUDIO_LATENTS = 207
# `prod(encoder_rates)` of the tiny audio VAE, standing in for the released 800 samples per latent.
AUDIO_HOP_LENGTH = 4
# The tiny audio VAE's own sample rate, so a reference soundtrack is never resampled.
AUDIO_SAMPLE_RATE = 40 * AUDIO_HOP_LENGTH

# The released short edge of an image reference is 2048 pixels, which packs thousands of conditioning rows.
TEST_REFERENCE_IMAGE_SHORT_EDGE = 64

# Both blocksets read the same repository, as the released one does: `transformer/` serves the `t2va` / `fl2va` half
# and `transformer_ref/` the `ref2va` one, every other component is shared.
TINY_MODULAR_REPO_ID = "hf-internal-testing/tiny-minimax-h3-modular-pipe"


_TAIL = [
    ("denoise.after_denoise", "MiniMaxH3AfterDenoiseStep"),
    ("decode.video", "MiniMaxH3VideoDecodeStep"),
    ("decode.audio", "MiniMaxH3AudioDecodeStep"),
]
T2VA_WORKFLOW = [
    ("text_encoder", "MiniMaxH3TextEncoderStep"),
    ("denoise.no_keyframe_anchors", "MiniMaxH3NoKeyframeAnchorsStep"),
    ("denoise.prepare_layout", "MiniMaxH3PrepareLayoutStep"),
    ("denoise.prepare_latents", "MiniMaxH3PrepareLatentsStep"),
    ("denoise.set_timesteps", "MiniMaxH3SetTimestepsStep"),
    ("denoise.denoise", "MiniMaxH3DenoiseStep"),
    *_TAIL,
]
# A keyframe adds the canvas block and the one block that encodes it; whether it anchors the first or the last frame
# is a matter of the packed layout, not of which blocks run.
FL2VA_WORKFLOW = [
    ("before_encode", "MiniMaxH3ResizeStep"),
    ("text_encoder", "MiniMaxH3FL2VATextEncoderStep"),
    ("vae_encoder", "MiniMaxH3KeyframeVaeEncoderStep"),
    ("denoise.prepare_layout", "MiniMaxH3PrepareLayoutStep"),
    ("denoise.prepare_condition_latents", "MiniMaxH3PrepareConditionLatentsStep"),
    ("denoise.prepare_latents", "MiniMaxH3PrepareLatentsStep"),
    ("denoise.prepare_latents_fl2va", "MiniMaxH3FL2VAPrepareLatentsStep"),
    ("denoise.set_timesteps", "MiniMaxH3SetTimestepsStep"),
    ("denoise.denoise", "MiniMaxH3DenoiseStep"),
    *_TAIL,
]
REF2VA_WORKFLOW = [
    ("before_encode", "MiniMaxH3Ref2VASetupStep"),
    ("text_encoder", "MiniMaxH3Ref2VATextEncoderStep"),
    ("vae_encoder", "MiniMaxH3Ref2VAReferenceEncoderStep"),
    ("denoise.prepare_layout", "MiniMaxH3Ref2VAPrepareLayoutStep"),
    ("denoise.prepare_condition_latents", "MiniMaxH3PrepareConditionLatentsStep"),
    ("denoise.prepare_latents", "MiniMaxH3PrepareLatentsStep"),
    ("denoise.prepare_latents_ref2va", "MiniMaxH3Ref2VAPrepareLatentsStep"),
    ("denoise.set_timesteps", "MiniMaxH3SetTimestepsStep"),
    ("denoise.denoise", "MiniMaxH3Ref2VADenoiseStep"),
    *_TAIL,
]
MINIMAX_H3_WORKFLOWS = {
    "t2va": T2VA_WORKFLOW,
    "fl2va": FL2VA_WORKFLOW,
}
MINIMAX_H3_REF2VA_WORKFLOWS = {"ref2va": REF2VA_WORKFLOW}

# What each pruned workflow asks for, pinned explicitly: every component with its class (each half loads its own
# transformer partition, nothing else differs) and every input — the optional ones with their defaults, the
# required ones by name. MiniMax-H3 declares no pipeline-level configs, which the omitted `configs` key asserts.
_SHARED_COMPONENTS = {
    "text_encoder": "Qwen3VLForConditionalGeneration",
    "tokenizer": "Qwen2Tokenizer",
    "processor": "Qwen3VLProcessor",
    "scheduler": "MiniMaxH3Scheduler",
    "audio_scheduler": "MiniMaxH3Scheduler",
    "vae": "AutoencoderKLMiniMaxH3",
    "audio_vae": "AutoencoderKLMiniMaxH3Audio",
    "video_processor": "VideoProcessor",
}
_SHARED_INPUTS = {
    "height": None,
    "width": None,
    "generator": None,
    "latents": None,
    "audio_latents": None,
    "attention_kwargs": None,
    "output_type": "pil",
}
T2VA_DEFAULTS = {
    "components": {**_SHARED_COMPONENTS, "transformer": "MiniMaxH3Transformer3DModel"},
    "required_inputs": ["prompt", "num_inference_steps"],
    "inputs": {**_SHARED_INPUTS, "num_frames": 124},
}
FL2VA_DEFAULTS = {
    "components": {
        **_SHARED_COMPONENTS,
        "transformer": "MiniMaxH3Transformer3DModel",
        "image_processor": "VaeImageProcessor",
    },
    "required_inputs": ["prompt", "num_inference_steps"],
    "inputs": {**_SHARED_INPUTS, "num_frames": 124, "image": None, "last_image": None},
}
REF2VA_DEFAULTS = {
    "components": {
        **_SHARED_COMPONENTS,
        "transformer_ref": "MiniMaxH3Transformer3DModel",
        "image_processor": "VaeImageProcessor",
    },
    # `num_frames` cannot default on ref2va: reference soundtracks are truncated to the generated duration, so a
    # silent 124-frame default would cut them short. It is required instead.
    "required_inputs": ["prompt", "references", "num_frames", "num_inference_steps"],
    "inputs": _SHARED_INPUTS,
}
MINIMAX_H3_WORKFLOW_DEFAULTS = {
    "t2va": T2VA_DEFAULTS,
    "fl2va": FL2VA_DEFAULTS,
}
MINIMAX_H3_REF2VA_WORKFLOW_DEFAULTS = {"ref2va": REF2VA_DEFAULTS}


def _video_frames(num_frames: int, size: int) -> np.ndarray:
    """Synthesized video reference frames: `uint8` RGB at MiniMax-H3's own 24 fps."""
    return (np.random.default_rng(0).random((num_frames, size, size, 3)) * 255).astype("uint8")


def _waveform(duration: float) -> torch.Tensor:
    """A synthesized soundtrack: a stereo waveform at the tiny audio VAE's own sample rate, so nothing is resampled."""
    return torch.rand(2, round(duration * AUDIO_SAMPLE_RATE), generator=torch.Generator("cpu").manual_seed(1)) * 2 - 1


def _reference_video(num_frames: int, size: int) -> MiniMaxH3VideoReference:
    """A silent video reference at MiniMax-H3's own 24 fps."""
    return MiniMaxH3VideoReference(frames=_video_frames(num_frames, size), fps=float(MINIMAX_H3_FPS))


def _reference_audio(duration: float) -> MiniMaxH3AudioReference:
    """A standalone audio reference at the tiny audio VAE's own sample rate."""
    return MiniMaxH3AudioReference(audio=_waveform(duration), sample_rate=AUDIO_SAMPLE_RATE)


# The synthesized media fixtures the file-decoding tests are built from: a tiny 8 fps clip with a stereo soundtrack.
FIXTURE_FPS = 8.0
FIXTURE_NUM_FRAMES = 8
FIXTURE_SAMPLE_RATE = 8000


def _write_video(path, with_audio: bool) -> None:
    """Encode a tiny 64x32 clip, with a stereo soundtrack when asked, with PyAV."""
    av = pytest.importorskip("av")

    with av.open(str(path), "w") as container:
        # Both streams are declared before anything is muxed, which is what the muxer needs to lay out the file.
        video_stream = container.add_stream("libx264", rate=int(FIXTURE_FPS))
        video_stream.width, video_stream.height, video_stream.pix_fmt = 64, 32, "yuv420p"
        audio_stream = None
        if with_audio:
            audio_stream = container.add_stream("aac", rate=FIXTURE_SAMPLE_RATE)
            audio_stream.codec_context.layout = "stereo"

        for index in range(FIXTURE_NUM_FRAMES):
            pixels = np.full((32, 64, 3), index * 16, dtype="uint8")
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            container.mux(video_stream.encode(frame))
        container.mux(video_stream.encode())

        if audio_stream is None:
            return
        num_samples = int(FIXTURE_NUM_FRAMES / FIXTURE_FPS * FIXTURE_SAMPLE_RATE)
        samples = np.zeros((1, 2 * num_samples), dtype="int16")
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="stereo")
        frame.sample_rate = FIXTURE_SAMPLE_RATE
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="stereo", rate=FIXTURE_SAMPLE_RATE)
        pts = 0
        for resampled in resampler.resample(frame):
            resampled.pts = pts
            pts += resampled.samples
            container.mux(audio_stream.encode(resampled))
        container.mux(audio_stream.encode())


def _media_fixtures(directory) -> tuple:
    """A video with a soundtrack, a still image and an audio clip, as files a reference can be built from."""
    av = pytest.importorskip("av")

    video_path, image_path, audio_path = (
        directory / "reference.mp4",
        directory / "reference.png",
        directory / "reference.wav",
    )
    _write_video(video_path, with_audio=True)
    Image.new("RGB", (64, 32), color=(10, 20, 30)).save(image_path)
    with av.open(str(audio_path), "w") as container:
        stream = container.add_stream("pcm_s16le", rate=FIXTURE_SAMPLE_RATE)
        stream.codec_context.layout = "stereo"
        frame = av.AudioFrame.from_ndarray(
            np.zeros((1, 2 * FIXTURE_SAMPLE_RATE), dtype="int16"), format="s16", layout="stereo"
        )
        frame.sample_rate = FIXTURE_SAMPLE_RATE
        frame.pts = 0
        container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return video_path, image_path, audio_path


@pytest.fixture(autouse=True, scope="module")
def small_references():
    """
    Encode references at `TEST_REFERENCE_IMAGE_SHORT_EDGE` instead of the released 2048 pixels.

    A 2048 pixel short edge packs thousands of conditioning rows, which is minutes per pipeline call on CPU. Every
    request of this module passes `height` and `width`, so the canvas of a generated video is never derived from
    these. `TestMiniMaxH3ReferenceGeometry` restores the released value where it pins it.
    """
    original_init = MiniMaxH3Ref2VASetupStep.__init__

    def small_reference_init(self, *args, **kwargs):
        kwargs.setdefault("reference_image_short_edge", TEST_REFERENCE_IMAGE_SHORT_EDGE)
        original_init(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(MiniMaxH3Ref2VASetupStep, "__init__", small_reference_init)
        patch.setattr(minimax_h3_geometry, "MINIMAX_H3_SHORT_EDGE", TEST_REFERENCE_IMAGE_SHORT_EDGE)
        patch.setattr(minimax_h3_geometry, "MINIMAX_H3_MAX_PIXELS", TEST_REFERENCE_IMAGE_SHORT_EDGE**2 * 2)
        yield


class MiniMaxH3ModularTesterBase(ModularPipelineTesterMixin):
    """What the two blocksets share: the repository, the output surface and the tests that do not apply."""

    pretrained_model_name_or_path = TINY_MODULAR_REPO_ID
    # MiniMax-H3 packs one request into one sequence and rejects a list of prompts, so nothing is batched.
    batch_params = frozenset()
    # The checkpoint is guidance-distilled and generates a video, so neither `guidance_scale` nor
    # `num_images_per_prompt` exists.
    optional_params = frozenset(["num_inference_steps", "latents", "output_type"])
    output_name = "videos"

    @pytest.mark.skip(reason="MiniMax-H3 packs one request into one sequence, so a batch of prompts is not a thing.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="MiniMax-H3 packs one request into one sequence, so a batch of prompts is not a thing.")
    def test_inference_batch_single_identical(self):
        pass

    @pytest.mark.skip(reason="One request generates one video, so there is no `num_images_per_prompt`.")
    def test_num_images_per_prompt(self):
        pass

    def test_load_expected_components_from_pretrained(self, tmp_path):
        r"""
        Every component the repository holds *for this half* is loaded from it.

        The repository holds both checkpoint partitions, as the released one does, so its index also names the
        transformer of the other half — which these blocks never declare and must therefore never load.
        """
        pipe = self.get_pipeline()
        specified = _get_specified_components(self.pretrained_model_name_or_path, cache_dir=tmp_path)

        expected = {name for name in specified if name in pipe.components}
        actual = {
            name
            for name in pipe.components
            if getattr(pipe, name, None) is not None
            and getattr(getattr(pipe, name), "_diffusers_load_id", None) not in (None, "null")
        }

        assert expected == actual, f"Component mismatch: missing={expected - actual}, unexpected={actual - expected}"
        assert "transformer_ref" in specified and "transformer" in specified

    def injected_noise(self, pipe, seed=3):
        r"""The video and audio noise a request would otherwise draw, in the shapes the blocks expect back."""
        latents = torch.randn(
            1,
            pipe.vae_latent_channels,
            NUM_LATENT_FRAMES,
            RESOLUTION // pipe.vae_spatial_compression_ratio,
            RESOLUTION // pipe.vae_spatial_compression_ratio,
            generator=torch.Generator("cpu").manual_seed(seed),
        )
        audio_latents = torch.randn(
            2, pipe.audio_latent_channels, NUM_AUDIO_LATENTS, generator=torch.Generator("cpu").manual_seed(seed + 1)
        )
        return latents, audio_latents

    def test_duration_ceiling_holds_for_the_aligned_count(self):
        r"""
        346 frames are 14.417 seconds, but they are rounded up to 362, i.e. 15.083 seconds: the ceiling holds for the
        aligned count, so this is rejected rather than silently generating too long a video.
        """
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = 346

        with pytest.raises(ValueError, match="rounded up to 362"):
            pipe(**inputs)


class TestMiniMaxH3ModularPipelineFast(MiniMaxH3ModularTesterBase):
    pipeline_class = MiniMaxH3ModularPipeline
    pipeline_blocks_class = MiniMaxH3Blocks

    params = frozenset(["prompt", "image", "last_image", "height", "width", "num_frames"])
    expected_workflow_blocks = MINIMAX_H3_WORKFLOWS
    expected_workflow_defaults = MINIMAX_H3_WORKFLOW_DEFAULTS

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a robot dancing",
            "generator": self.get_generator(seed),
            "num_inference_steps": 2,
            "height": RESOLUTION,
            "width": RESOLUTION,
            "num_frames": NUM_FRAMES,
            "output_type": "pt",
        }

    def test_video_and_audio_outputs(self):
        r"""One call denoises both modalities out of the one packed sequence, and returns both."""
        pipe = self.get_pipeline()

        state = pipe(**self.get_dummy_inputs())
        video, audio = state.get("videos"), state.get("audio")

        assert video.shape == (1, NUM_FRAMES, 3, RESOLUTION, RESOLUTION)
        # The audio VAE is mono and takes the two stereo channels as two batch items, which the decoder block stacks
        # back into `(batch, 2, samples)`.
        assert audio.shape == (1, 2, NUM_AUDIO_LATENTS * AUDIO_HOP_LENGTH)
        assert video.min() >= 0.0 and video.max() <= 1.0
        assert state.get("sampling_rate") == pipe.audio_vae.config.sampling_rate

    @pytest.mark.parametrize(
        "keyframes,num_condition_rows",
        [(("image",), 1), (("last_image",), 1), (("image", "last_image"), 2)],
        ids=["first", "last", "first_and_last"],
    )
    def test_fl2va_keyframes(self, keyframes, num_condition_rows):
        r"""
        A keyframe contributes one conditioning row per latent patch, and those rows are pinned for the whole loop.

        They are packed in front of the generated video rows and ride along at their own `t = 0.999`. The loop only
        ever writes the generated rows, so the conditioning rows the encoder block produced survive it untouched,
        which is what the denoised sequence is checked against here. On a `RESOLUTION` canvas a keyframe is a single
        `(1, 2, 2)` patch row.
        """
        pipe = self.get_pipeline()
        keyframe = Image.fromarray((np.random.default_rng(0).random((48, 80, 3)) * 255).astype("uint8"))

        inputs = self.get_dummy_inputs()
        inputs.update(dict.fromkeys(keyframes, keyframe))
        state = pipe(**inputs)

        assert state.get("videos").shape == (1, NUM_FRAMES, 3, RESOLUTION, RESOLUTION)
        assert state.get("num_condition_video_rows") == num_condition_rows
        condition_latents = state.get("condition_latents")
        assert condition_latents.shape[0] == num_condition_rows
        assert torch.equal(state.get("latents")[:num_condition_rows], condition_latents)

    @pytest.mark.parametrize("with_keyframe", [False, True], ids=["t2va", "fl2va"])
    def test_generator_reproducibility(self, with_keyframe):
        r"""
        Two runs from the same generator state are identical, two seeds differ.

        The blocks draw from the one generator the request carries, in the order they run — the keyframe conditioning
        noise in the VAE encoder step, then the video and the audio noise in the prepare-latents step — which is the
        whole reproducibility contract of a request.
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

    def test_injected_latents_replace_the_draws(self):
        r"""
        `latents` and `audio_latents` stand in for their draw, which is how a sample is reproduced from outside.

        With both passed in, a `t2va` request draws nothing at all, so two runs with different generators return the
        very same video and soundtrack.
        """
        pipe = self.get_pipeline()
        latents, audio_latents = self.injected_noise(pipe)

        outputs = []
        for seed in (7, 8):
            inputs = self.get_dummy_inputs()
            inputs["generator"] = torch.Generator("cpu").manual_seed(seed)
            inputs["latents"] = latents
            inputs["audio_latents"] = audio_latents
            state = pipe(**inputs)
            outputs.append((state.get("videos"), state.get("audio")))

        assert torch.equal(outputs[0][0], outputs[1][0])
        assert torch.equal(outputs[0][1], outputs[1][1])

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

        inputs = self.get_dummy_inputs()
        if num_keyframes:
            inputs["image"] = keyframe
        if num_keyframes == 2:
            inputs["last_image"] = keyframe
        state = pipe(**inputs)
        prompt_embeds, text_token_tags = state.get("prompt_embeds"), state.get("text_token_tags")

        assert prompt_embeds.shape[0] == 1
        assert prompt_embeds.shape[-1] == pipe.transformer.config.text_dim
        assert prompt_embeds.shape[1] == text_token_tags.shape[0]
        assert torch.isfinite(prompt_embeds).all()
        # `0` tags a row of a vision block and `1` a text row, so a text-only presentation carries text rows alone.
        assert set(text_token_tags.tolist()) == ({0, 1} if num_keyframes else {1})

    def test_text_encoder_block_standalone(self):
        r"""
        The text encoder block runs on its own, without the denoiser or the VAEs.

        Encoding a presentation once and reusing it across requests is the modular way of passing `prompt_embeds`,
        so the block may only touch the components it declares.
        """
        pipe = MiniMaxH3TextEncoderStep().init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)

        outputs = pipe(prompt="a robot dancing", output=["prompt_embeds", "text_token_tags"])

        assert set(pipe.components) == {"text_encoder", "tokenizer", "processor"}
        assert outputs["prompt_embeds"].shape[0] == 1
        assert outputs["prompt_embeds"].shape[1] == outputs["text_token_tags"].shape[0]
        # `1` tags a text row, and a text-only presentation is the verbatim prompt.
        assert set(outputs["text_token_tags"].tolist()) == {1}

    @pytest.mark.parametrize("output_type", ["np", "pil", "latent"])
    def test_output_type(self, output_type):
        r"""`"latent"` stops before both VAEs and keeps the denormalized latents of either modality."""
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["output_type"] = output_type
        state = pipe(**inputs)
        video, audio = state.get("videos"), state.get("audio")

        if output_type == "np":
            assert video.shape == (1, NUM_FRAMES, RESOLUTION, RESOLUTION, 3)
        elif output_type == "pil":
            assert len(video[0]) == NUM_FRAMES
            assert video[0][0].size == (RESOLUTION, RESOLUTION)
        else:
            # The video as `(1, C, F, H, W)` and the audio channel-major.
            assert video.shape == (1, 4, NUM_LATENT_FRAMES, RESOLUTION // 16, RESOLUTION // 16)
            assert audio.shape == (2, 8, NUM_AUDIO_LATENTS)

    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"prompt": ["a robot", "a fox"]}, "must be a single string"),
            ({"height": 30, "width": 30}, "multiples of 32"),
            ({"width": None}, "have to be passed together"),
            ({"num_frames": 96}, "must be between"),
            ({"num_frames": 400}, "must be between"),
        ],
        ids=[
            "prompt_list",
            "canvas_not_a_multiple_of_32",
            "height_without_width",
            "shorter_than_five_seconds",
            "longer_than_fifteen_seconds",
        ],
    )
    def test_check_inputs(self, overrides, message):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs.update(overrides)

        with pytest.raises(ValueError, match=message):
            pipe(**inputs)


class TestMiniMaxH3Ref2VAModularPipelineFast(MiniMaxH3ModularTesterBase):
    pipeline_class = MiniMaxH3ModularPipeline
    pipeline_blocks_class = MiniMaxH3Blocks

    params = frozenset(["prompt", "references", "height", "width", "num_frames"])
    expected_workflow_blocks = MINIMAX_H3_REF2VA_WORKFLOWS
    expected_workflow_defaults = MINIMAX_H3_REF2VA_WORKFLOW_DEFAULTS

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a robot dancing",
            "references": [MiniMaxH3ImageReference(image=Image.new("RGB", (48, 80)))],
            "generator": self.get_generator(seed),
            "num_inference_steps": 2,
            "height": RESOLUTION,
            "width": RESOLUTION,
            "num_frames": NUM_FRAMES,
            "output_type": "pt",
        }

    def test_video_and_audio_outputs(self):
        r"""A reference conditions the request without binding the generated geometry."""
        pipe = self.get_pipeline()

        state = pipe(**self.get_dummy_inputs())
        video, audio = state.get("videos"), state.get("audio")

        assert video.shape == (1, NUM_FRAMES, 3, RESOLUTION, RESOLUTION)
        assert audio.shape == (1, 2, NUM_AUDIO_LATENTS * AUDIO_HOP_LENGTH)
        assert video.min() >= 0.0 and video.max() <= 1.0

    def test_generator_reproducibility(self):
        r"""
        Two runs from the same generator state are identical, two seeds differ.

        A `ref2va` request draws the reference conditioning noise before the video and the audio noise, all three off
        the one generator it carries, so this covers the whole stream.
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

    def test_injected_latents_replace_the_draws(self):
        r"""
        `latents` and `audio_latents` stand in for their draw, and only for theirs.

        A `ref2va` request draws the reference conditioning noise *before* the two target draws, so injecting the
        targets replaces those two alone: the same seed with other noise is another sample, and another seed with the
        same noise is still another sample, through the conditioning noise the references are augmented with.
        """
        pipe = self.get_pipeline()
        latents, audio_latents = self.injected_noise(pipe)
        other_latents, other_audio_latents = self.injected_noise(pipe, seed=11)

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
        other_seed_video, _ = run(8, latents, audio_latents)

        assert torch.equal(video, same_video)
        assert torch.equal(audio, same_audio)
        assert not torch.equal(video, other_noise_video)
        assert not torch.equal(audio, other_noise_audio)
        assert not torch.equal(video, other_seed_video)

    @pytest.mark.parametrize(
        "kinds,num_frames",
        [(("video",), NUM_FRAMES), (("image", "audio"), NUM_FRAMES), (("video", "image"), NUM_FRAMES)],
        ids=["video", "image_audio", "video_image"],
    )
    def test_reference_combinations(self, kinds, num_frames):
        r"""
        Any ordered mix of image, video and audio references is packed, in request order.

        A video reference conditions on its motion *and*, when the request passes one with it, on its soundtrack, so
        it contributes both visual and audio rows; an audio reference contributes audio rows alone and never reaches
        the conditioner.

        The reference rows are pinned for the whole loop, which is what the denoised sequence is checked against: the
        visual anchors keep their noise-augmented values and the audio anchors stay exactly as the encoder produced
        them.
        """
        media = {
            "image": MiniMaxH3ImageReference(image=Image.new("RGB", (48, 80))),
            # A one-second video reference, soundtrack included, and a six-second standalone soundtrack.
            "video": MiniMaxH3VideoReference(
                frames=_video_frames(MINIMAX_H3_FPS, 64),
                fps=float(MINIMAX_H3_FPS),
                audio=_waveform(1.0),
                sample_rate=AUDIO_SAMPLE_RATE,
            ),
            "audio": _reference_audio(6.0),
        }

        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["references"] = [media[kind] for kind in kinds]
        inputs["num_frames"] = num_frames

        state = pipe(**inputs)

        num_condition_rows = state.get("num_condition_video_rows")
        num_audio_condition_rows = state.get("num_condition_audio_rows")
        assert num_condition_rows > 0
        assert (num_audio_condition_rows > 0) == any(kind in ("video", "audio") for kind in kinds)
        assert state.get("videos").shape == (1, state.get("num_frames"), 3, RESOLUTION, RESOLUTION)
        # The loop only ever writes the generated rows, so the anchors the encoder block produced survive it.
        assert torch.equal(state.get("latents")[:num_condition_rows], state.get("condition_latents"))
        if num_audio_condition_rows:
            assert torch.equal(
                state.get("audio_latents")[:num_audio_condition_rows], state.get("audio_condition_latents")
            )

    @pytest.mark.parametrize(
        "kinds", [("image",), ("video",), ("video", "image")], ids=["image", "video", "video_image"]
    )
    def test_encode_prompt(self, kinds):
        r"""
        The presentation is encoded in one conditioner call, whatever references it labels.

        The rows of a reference's vision block are tagged as *video*, and those blocks are what makes Qwen3-VL lay its
        rotary positions out per modality run, which it reads off the `mm_token_type_ids` the processor derives from
        the vision pad ids. A video reference contributes one timestamped block per merged frame pair.
        """
        pipe = self.get_pipeline()
        media = {"image": MiniMaxH3ImageReference(image=Image.new("RGB", (48, 80))), "video": _reference_video(25, 32)}
        inputs = self.get_dummy_inputs()
        inputs["references"] = [media[kind] for kind in kinds]
        state = pipe(**inputs)
        prompt_embeds, text_token_tags = state.get("prompt_embeds"), state.get("text_token_tags")

        assert prompt_embeds.shape[0] == 1
        assert prompt_embeds.shape[-1] == pipe.transformer_ref.config.text_dim
        assert prompt_embeds.shape[1] == text_token_tags.shape[0]
        assert torch.isfinite(prompt_embeds).all()
        # `0` tags a row of a vision block and `1` a text row, and every reference here carries a vision block.
        assert set(text_token_tags.tolist()) == {0, 1}

    @pytest.mark.parametrize("media_type", ["pil", "np", "pt"], ids=["pil", "numpy", "torch"])
    def test_reference_media_layouts(self, media_type):
        r"""
        Image and video references are in-memory media, in any of the layouts diffusers accepts: images, a
        channels-last `np.ndarray` or a channels-first `torch.Tensor`, `uint8` or floating point over `[0, 1]`. All
        three carry the same pixels, and media that already is at the resolution the reference resolves to reaches the
        VAE untouched.
        """
        pipe = self.get_pipeline()
        pixels = _video_frames(4, TEST_REFERENCE_IMAGE_SHORT_EDGE)
        if media_type == "pil":
            image, frames = Image.fromarray(pixels[0]), [Image.fromarray(frame) for frame in pixels]
        elif media_type == "np":
            image, frames = pixels[0] / 255.0, pixels
        else:
            image, frames = torch.from_numpy(pixels[0]).permute(2, 0, 1), torch.from_numpy(pixels).permute(0, 3, 1, 2)

        inputs = self.get_dummy_inputs()
        inputs["references"] = [
            MiniMaxH3ImageReference(image=image),
            MiniMaxH3VideoReference(frames=frames, fps=float(MINIMAX_H3_FPS)),
        ]
        state = pipe(**inputs)
        references = state.get("normalized_references")

        assert np.array_equal(np.asarray(references[0].image), pixels[0])
        assert np.array_equal(references[1].frames, pixels)

    def test_reference_rates_default_to_the_model_rates(self):
        r"""
        A reference that leaves its rates out is taken to already be at MiniMax-H3's own: the frames flow through
        untouched, without a resampling pass and without a copy, and so do the samples of a waveform.
        """
        pipe = self.get_pipeline()
        frames = _video_frames(NUM_FRAMES, TEST_REFERENCE_IMAGE_SHORT_EDGE)
        waveform = _waveform(2.0)

        inputs = self.get_dummy_inputs()
        inputs["references"] = [MiniMaxH3VideoReference(frames=frames), MiniMaxH3AudioReference(audio=waveform)]
        state = pipe(**inputs)
        references = state.get("normalized_references")

        assert np.shares_memory(references[0].frames, frames)
        assert torch.equal(references[1].audio, waveform)

    def test_reference_sample_rate_override_resamples(self):
        r"""A waveform that says it carries another rate is resampled onto the audio VAE's own."""
        pipe = self.get_pipeline()
        waveform = _waveform(2.0)

        inputs = self.get_dummy_inputs()
        inputs["references"] = [
            MiniMaxH3ImageReference(image=Image.new("RGB", (48, 80))),
            MiniMaxH3AudioReference(audio=waveform, sample_rate=AUDIO_SAMPLE_RATE // 2),
        ]
        state = pipe(**inputs)
        references = state.get("normalized_references")

        # Half the audio VAE's rate, so the same samples span twice as many of the VAE's own.
        assert references[1].audio.shape == (2, 2 * waveform.shape[-1])

    def test_check_inputs_references(self, references, message):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["references"] = references

        with pytest.raises(ValueError, match=message):
            pipe(**inputs)


class TestMiniMaxH3Reference:
    """
    The three reference dataclasses, which hold in-memory media and the rate it carries.

    Each knows its own modality, so nothing has to derive it from which fields happen to be set, and the fields a
    modality has no use for do not exist on it.
    """

    def test_reference_defaults(self):
        r"""A reference knows its own modality, and defaults to MiniMax-H3's own frame rate for its frames."""
        assert MiniMaxH3VideoReference(frames=_video_frames(2, 32)).fps == float(MINIMAX_H3_FPS)
        assert MiniMaxH3ImageReference(image=Image.new("RGB", (32, 32))).kind == "image"
        assert not MiniMaxH3ImageReference(image=Image.new("RGB", (32, 32))).has_audio
        assert _reference_video(2, 32).kind == "video"
        assert not _reference_video(2, 32).has_audio
        assert _reference_audio(6.0).kind == "audio"
        assert _reference_audio(6.0).has_audio

    def test_video_reference_has_audio_follows_its_soundtrack(self):
        r"""A video reference contributes audio rows exactly when it was given a soundtrack of its own."""
        frames = _video_frames(2, 32)

        assert not MiniMaxH3VideoReference(frames=frames).has_audio
        assert MiniMaxH3VideoReference(frames=frames, audio=_waveform(1.0)).has_audio

    def test_a_modality_only_carries_its_own_fields(self):
        r"""
        The fields that used to be dead weight are gone: an image reference has no rates, and no reference can be
        built holding two media at once.
        """
        image = MiniMaxH3ImageReference(image=Image.new("RGB", (32, 32)))

        assert not hasattr(image, "fps") and not hasattr(image, "sample_rate")
        with pytest.raises(TypeError):
            MiniMaxH3ImageReference(image=Image.new("RGB", (32, 32)), audio=_waveform(1.0))
        with pytest.raises(TypeError):
            MiniMaxH3AudioReference(audio=_waveform(1.0), frames=_video_frames(2, 32))


class TestMiniMaxH3ReferenceLoading:
    """
    Decoding references off the filesystem, which is the caller's job rather than the blocks'.

    None of this needs a checkpoint: [`MiniMaxH3VideoReference.from_file`] and [`MiniMaxH3AudioReference.from_file`]
    read a container with PyAV and return a reference carrying the rates it reports.
    """

    def test_decode_video_carries_frames_rate_and_soundtrack(self, tmp_path):
        r"""
        A decoded video brings the three things the model conditions on: the frames, the rate they were shot at, and
        the container's soundtrack as this reference's own.
        """
        video_path, _, _ = _media_fixtures(tmp_path)

        video = MiniMaxH3VideoReference.from_file(video_path)

        assert isinstance(video, MiniMaxH3VideoReference) and video.kind == "video"
        assert video.frames.shape == (FIXTURE_NUM_FRAMES, 32, 64, 3) and video.frames.dtype == np.uint8
        assert video.fps == FIXTURE_FPS
        assert video.has_audio and video.audio.shape[0] == 2 and video.sample_rate == FIXTURE_SAMPLE_RATE

    def test_decode_audio_carries_the_sample_rate(self, tmp_path):
        r"""A decoded audio file brings the rate its samples are at, which is what they are resampled from."""
        _, _, audio_path = _media_fixtures(tmp_path)

        audio = MiniMaxH3AudioReference.from_file(audio_path)

        assert isinstance(audio, MiniMaxH3AudioReference) and audio.kind == "audio"
        assert audio.audio.shape[0] == 2 and audio.audio.dtype == torch.float32
        assert audio.sample_rate == FIXTURE_SAMPLE_RATE

    def test_decoded_rates_can_be_corrected(self, tmp_path):
        r"""A container whose metadata is wrong is corrected on the reference the decode returned."""
        video_path, _, _ = _media_fixtures(tmp_path)

        video = MiniMaxH3VideoReference.from_file(video_path)
        video.fps = 12.0

        assert video.fps == 12.0 and video.sample_rate == FIXTURE_SAMPLE_RATE

    def test_decode_silent_video_carries_no_soundtrack(self, tmp_path):
        r"""There is nothing to adopt from a container without an audio stream."""
        silent_path = tmp_path / "silent.mp4"
        _write_video(silent_path, with_audio=False)

        reference = MiniMaxH3VideoReference.from_file(silent_path)

        assert not reference.has_audio and reference.sample_rate is None
        assert reference.fps == FIXTURE_FPS

    def test_decoding_needs_pyav(self, tmp_path, monkeypatch):
        r"""Decoding a video or an audio file is a PyAV job, and says so when PyAV is not installed."""
        video_path, _, audio_path = _media_fixtures(tmp_path)
        monkeypatch.setattr(minimax_h3_references, "is_av_available", lambda: False)

        with pytest.raises(ImportError, match="pip install av"):
            MiniMaxH3VideoReference.from_file(video_path)
        with pytest.raises(ImportError, match="pip install av"):
            MiniMaxH3AudioReference.from_file(audio_path)

    @pytest.mark.parametrize(
        "loader,path",
        [
            (MiniMaxH3ImageReference.from_file, "missing.png"),
            (MiniMaxH3VideoReference.from_file, "missing.mp4"),
            (MiniMaxH3AudioReference.from_file, "missing.wav"),
        ],
        ids=["image", "video", "audio"],
    )
    def test_from_file_needs_a_real_path(self, loader, path):
        r"""The path has to name a file it can open, or a URL."""
        with pytest.raises(ValueError, match="not a valid path"):
            loader(path)


class TestMiniMaxH3ReferenceGeometry:
    """
    How a reference is prepared, which is pure geometry and needs no checkpoint.

    These are the passes `MiniMaxH3Ref2VASetupStep` runs a reference through before any VAE sees it.
    """

    def test_video_reference_vision_blocks(self):
        r"""
        The conditioner reads a reference video at 2 fps and Qwen3-VL merges every two of those frames into one
        vision block, labelled with the mean timestamp of the pair.

        A 25 frame reference at 24 fps samples three frames, which merge into two blocks at 0.25 and 1.0 seconds.
        """
        frames, timestamps = MiniMaxH3Ref2VATextEncoderStep._sample_video_condition_frames(
            _video_frames(25, 32), fps=24.0, sample_fps=2.0, temporal_patch=2
        )

        assert len(frames) == 3
        assert timestamps == [0.25, 1.0]

    def test_reference_image_geometry(self):
        r"""
        A reference image is encoded at a 2048 pixel short edge, both axes rounded to a multiple of 32, with no area
        cap — the released geometry, so the setup step is constructed explicitly rather than through the
        `small_references` fixture's shrunken default.
        """
        pipe = MiniMaxH3Ref2VASetupStep(reference_image_short_edge=2048).init_pipeline()

        references = pipe(
            references=[MiniMaxH3ImageReference(image=Image.new("RGB", (80, 48)))],
            num_frames=124,
            output="normalized_references",
        )

        assert references[0].image.size == (3424, 2048)

    def test_video_reference_resampled_to_the_model_frame_rate(self):
        r"""
        A video reference is resampled onto MiniMax-H3's own 24 fps by dropping and duplicating whole frames, which is
        what `ffmpeg`'s `fps` filter did in the reference implementation: a frame whose successor rounds onto the same
        output slot is dropped, so a 30 fps reference loses one frame in five, the later of the two that tie for a
        slot. Frames already at 24 fps are returned as they are, without a copy.
        """
        # Every frame carries its own index as its pixel value, so the resampled frames name the ones that
        # survived. The frames sit at the canvas their aspect resolves to under the `small_references` fixture, so
        # the resize pass is a no-op and the 24 fps route returns the input without a copy.
        size = TEST_REFERENCE_IMAGE_SHORT_EDGE
        frames = np.arange(30, dtype="uint8").reshape(-1, 1, 1, 1) * np.ones((1, size, size, 3), dtype="uint8")

        resampled = MiniMaxH3Ref2VASetupStep._normalize_video_condition(
            frames, fps=30.0, num_frames=NUM_FRAMES, canvas_multiple=32
        )

        assert [int(frame[0, 0, 0]) for frame in resampled] == [
            index for index in range(30) if index not in (2, 7, 12, 17, 22, 27)
        ]
        untouched = MiniMaxH3Ref2VASetupStep._normalize_video_condition(
            frames, fps=float(MINIMAX_H3_FPS), num_frames=NUM_FRAMES, canvas_multiple=32
        )
        assert np.shares_memory(untouched, frames)
