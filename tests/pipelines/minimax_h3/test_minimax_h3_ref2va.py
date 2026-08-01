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
    MiniMaxH3Ref2VAPipeline,
    MiniMaxH3Scheduler,
    MiniMaxH3Transformer3DModel,
)
from diffusers.pipelines.minimax_h3 import MiniMaxH3Reference, packing, packing_ref2va

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
# The dummy audio VAE's own sample rate, so a reference soundtrack is never resampled.
AUDIO_SAMPLE_RATE = 40 * AUDIO_HOP_LENGTH

# A reference image is encoded at a 2048 pixel short edge, which packs thousands of conditioning rows. Tests shrink
# it to keep the CPU runs fast; `test_reference_image_geometry` pins the released value.
TEST_REFERENCE_IMAGE_SHORT_EDGE = 64

# The tokenizer of the released conditioner, for its `<|vision_start|>` / `<|image_pad|>` / `<|vision_end|>` ids.
TINY_QWEN_CKPT_ID = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"


def _video_frames(num_frames: int, size: int) -> np.ndarray:
    """Synthesized video reference frames: `uint8` RGB at MiniMax-H3's own 24 fps.

    References are in-memory media, so nothing here is written to a file: the pipeline takes the frames a caller
    decoded, plus the frame rate they carry.
    """
    return (np.random.default_rng(0).random((num_frames, size, size, 3)) * 255).astype("uint8")


def _waveform(duration: float) -> torch.Tensor:
    """A synthesized soundtrack: a stereo waveform at the dummy audio VAE's own sample rate, so nothing is resampled."""
    return torch.rand(2, round(duration * AUDIO_SAMPLE_RATE), generator=torch.Generator("cpu").manual_seed(1)) * 2 - 1


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


def _reference_video(num_frames: int, size: int) -> MiniMaxH3Reference:
    """A silent video reference at MiniMax-H3's own 24 fps."""
    return MiniMaxH3Reference(video=_video_frames(num_frames, size), fps=float(packing.MINIMAX_H3_FPS))


def _reference_audio(duration: float) -> MiniMaxH3Reference:
    """A standalone audio reference at the dummy audio VAE's own sample rate."""
    return MiniMaxH3Reference(audio=_waveform(duration), sample_rate=AUDIO_SAMPLE_RATE)


@pytest.fixture(autouse=True, scope="module")
def small_references():
    """
    Encode references at `TEST_REFERENCE_IMAGE_SHORT_EDGE` instead of the released 2048 pixels.

    A 2048 pixel short edge packs thousands of conditioning rows, which is minutes per pipeline call on CPU. The
    fixture is module-scoped because the cached `base_pipe_output` is built once per class, before any
    function-scoped fixture runs. `test_reference_image_geometry` restores the released value.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(packing_ref2va, "MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE", TEST_REFERENCE_IMAGE_SHORT_EDGE)
        patch.setattr(packing, "MINIMAX_H3_SHORT_EDGE", TEST_REFERENCE_IMAGE_SHORT_EDGE)
        patch.setattr(packing, "MINIMAX_H3_MAX_PIXELS", TEST_REFERENCE_IMAGE_SHORT_EDGE**2 * 2)
        yield


class MiniMaxH3Ref2VAPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = MiniMaxH3Ref2VAPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "references", "height", "width", "num_frames", "prompt_embeds"]
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
        transformer_ref = MiniMaxH3Transformer3DModel(
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
            "transformer_ref": transformer_ref,
            "vae": vae,
            "audio_vae": audio_vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "processor": processor,
            "scheduler": MiniMaxH3Scheduler(shift=12.0),
            "audio_scheduler": MiniMaxH3Scheduler(shift=3.0),
        }

    def get_dummy_inputs(self):
        # A `ref2va` presentation labels every reference in the prompt, so the conditioner has to see the very
        # references the request carries. The tests pass the encoded presentation directly, which is also how a
        # caller reuses one across requests.
        return {
            "prompt_embeds": torch.randn(1, 7, 16, generator=torch.Generator("cpu").manual_seed(0)),
            "text_token_tags": torch.ones(7, dtype=torch.long),
            "references": [MiniMaxH3Reference(image=Image.new("RGB", (48, 80)))],
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "height": RESOLUTION,
            "width": RESOLUTION,
            "num_frames": NUM_FRAMES,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestMiniMaxH3Ref2VAPipeline(MiniMaxH3Ref2VAPipelineTesterConfig, PipelineTesterMixin):
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

    def test_reference_image_geometry(self, monkeypatch):
        r"""
        A reference image is encoded at a 2048 pixel short edge, both axes rounded to a multiple of 32, with no area
        cap. The `small_references` fixture shrinks that short edge for every other test, so restore it here.
        """
        monkeypatch.setattr(packing_ref2va, "MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE", 2048)

        assert packing_ref2va.resolve_reference_image_size(80, 48) == (2048, 3424)
        assert packing_ref2va.resolve_reference_image_size(48, 80) == (3424, 2048)

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

        references, _ = pipe.prepare_references(
            [
                MiniMaxH3Reference(image=image),
                MiniMaxH3Reference(video=frames, fps=float(packing.MINIMAX_H3_FPS)),
            ],
            NUM_FRAMES,
        )

        assert np.array_equal(np.asarray(references[0].image), pixels[0])
        assert np.array_equal(references[1].frames, pixels)

    def test_video_reference_resampled_to_the_model_frame_rate(self):
        r"""
        A video reference is resampled onto MiniMax-H3's own 24 fps by dropping and duplicating whole frames, which is
        what `ffmpeg`'s `fps` filter did in the reference implementation: a frame whose successor rounds onto the same
        output slot is dropped, so a 30 fps reference loses one frame in five, the later of the two that tie for a
        slot. Frames already at 24 fps are returned as they are, without a copy.
        """
        # Every frame carries its own index as its pixel value, so the resampled frames name the ones that survived.
        frames = np.arange(30, dtype="uint8").reshape(-1, 1, 1, 1) * np.ones((1, 2, 2, 3), dtype="uint8")

        resampled = packing_ref2va.resample_reference_frames(frames, 30.0)

        assert [int(frame[0, 0, 0]) for frame in resampled] == [
            index for index in range(30) if index not in (2, 7, 12, 17, 22, 27)
        ]
        assert packing_ref2va.resample_reference_frames(frames, float(packing.MINIMAX_H3_FPS)) is frames

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
        media = {"image": MiniMaxH3Reference(image=Image.new("RGB", (48, 80))), "video": _reference_video(25, 32)}
        references, _ = pipe.prepare_references([media[kind] for kind in kinds], NUM_FRAMES)

        prompt_embeds, text_token_tags = pipe.encode_prompt("a robot dancing", references, device=torch_device)

        assert prompt_embeds.shape[0] == 1
        assert prompt_embeds.shape[-1] == pipe.transformer_ref.config.text_dim
        assert prompt_embeds.shape[1] == text_token_tags.shape[0]
        assert torch.isfinite(prompt_embeds).all()
        # `0` tags a row of a vision block and `1` a text row, and every reference here carries a vision block.
        assert set(text_token_tags.tolist()) == {0, 1}
        # A 25 frame reference decoded at 24 fps is read at 2 fps, so it is three frames merged into two blocks.
        for reference in references:
            if reference.kind == "video":
                assert reference.block_timestamps == [0.25, 1.0]

    @pytest.mark.parametrize(
        "kinds,num_frames",
        [(("video",), NUM_FRAMES), (("image", "audio"), None), (("video", "image"), NUM_FRAMES)],
        ids=["video", "image_audio", "video_image"],
    )
    def test_reference_combinations(self, kinds, num_frames):
        r"""
        Any ordered mix of image, video and audio references is packed, in request order.

        A video reference conditions on its motion *and*, when the request passes one with it, on its soundtrack, so it
        contributes both visual and audio rows; an audio reference contributes audio rows alone and never reaches the
        conditioner. The `image_audio` case also leaves `num_frames` to the references, which is admissible because
        exactly one of them carries audio: the duration is that soundtrack's, snapped up to the next `17 * n + 5` the
        video VAE can decode.

        The reference rows are pinned for the whole loop, which is what the callback checks: the visual anchors keep
        their noise-augmented values and the audio anchors stay exactly as the encoder produced them.
        """
        media = {
            "image": MiniMaxH3Reference(image=Image.new("RGB", (48, 80))),
            # A one-second video reference, soundtrack included, and a six-second standalone soundtrack.
            "video": MiniMaxH3Reference(
                video=_video_frames(packing.MINIMAX_H3_FPS, 64),
                fps=float(packing.MINIMAX_H3_FPS),
                audio=_waveform(1.0),
                sample_rate=AUDIO_SAMPLE_RATE,
            ),
            "audio": _reference_audio(6.0),
        }

        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["references"] = [media[kind] for kind in kinds]
        inputs["num_frames"] = num_frames
        inputs["num_inference_steps"] = 3

        references, resolved_num_frames = pipe.prepare_references(inputs["references"], num_frames)
        condition_rows, audio_condition_rows = pipe.encode_references(references)
        num_condition_rows = 0 if condition_rows is None else condition_rows.shape[0]
        num_audio_condition_rows = 0 if audio_condition_rows is None else audio_condition_rows.shape[0]
        assert num_condition_rows > 0
        assert (num_audio_condition_rows > 0) == any(kind in ("video", "audio") for kind in kinds)

        seen = []
        inputs["callback_on_step_end"] = (
            lambda pipeline, i, t, kwargs: seen.append((kwargs["latents"].clone(), kwargs["audio_latents"].clone()))
            or {}
        )
        video = pipe(**inputs).frames

        assert video.shape == (1, resolved_num_frames, 3, RESOLUTION, RESOLUTION)
        # `num_inference_steps` counts sigma grid points, the terminal `0` included, so it drives one model
        # evaluation less.
        assert len(seen) == inputs["num_inference_steps"] - 1
        for latents, audio_latents in seen:
            assert torch.equal(latents[:num_condition_rows], seen[0][0][:num_condition_rows])
            if num_audio_condition_rows:
                assert torch.equal(audio_latents[:num_audio_condition_rows], audio_condition_rows)

    def test_num_frames_left_open_needs_exactly_one_audio_reference(self):
        r"""Leaving `num_frames` out is ambiguous unless exactly one reference carries audio."""
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = None
        inputs["references"] = [MiniMaxH3Reference(image=Image.new("RGB", (48, 80)))]

        with pytest.raises(ValueError, match="may only be left to the references"):
            pipe(**inputs)

    def test_num_frames_left_open_holds_the_ceiling_for_the_aligned_count(self):
        r"""
        A soundtrack just under 15 seconds is rejected rather than silently stretched past the ceiling.

        14.99 seconds round to 360 frames, which the video VAE's `17 * n + 5` grid rounds up to 362, i.e. 15.083
        seconds. The duration the request generates is the aligned one, so that is what the bound holds for.
        """
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = None
        inputs["references"] = [MiniMaxH3Reference(image=Image.new("RGB", (48, 80))), _reference_audio(14.99)]

        with pytest.raises(ValueError, match="rounds up to 362 frames"):
            pipe(**inputs)

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
            output = pipe(**inputs)
            return output.frames, output.audio

        frames, audio = run(7)
        same_frames, same_audio = run(7)
        other_frames, other_audio = run(8)

        assert torch.equal(frames, same_frames)
        assert torch.equal(audio, same_audio)
        assert not torch.equal(frames, other_frames)
        assert not torch.equal(audio, other_audio)

    @pytest.mark.parametrize(
        "references,message",
        [
            ([], "at least one reference"),
            ([_reference_audio(6.0)], "cannot be used"),
            ([MiniMaxH3Reference(image=Image.new("RGB", (32, 32)))] * 10, "at most 9 image references"),
            (
                [MiniMaxH3Reference(image=Image.new("RGB", (32, 32)))] + [_reference_video(2, 32)] * 4,
                "at most 3 video references",
            ),
            ([{"image": Image.new("RGB", (32, 32))}], "must be a \\[`MiniMaxH3Reference`\\]"),
        ],
        ids=[
            "no_references",
            "audio_alone",
            "too_many_images",
            "too_many_videos",
            "dict_entry",
        ],
    )
    def test_check_inputs_references(self, references, message):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["references"] = references

        with pytest.raises(ValueError, match=message):
            pipe(**inputs)

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({}, "exactly one of"),
            ({"image": Image.new("RGB", (32, 32)), "video": _video_frames(2, 32)}, "exactly one of"),
            ({"image": Image.new("RGB", (32, 32)), "audio": _waveform(1.0)}, "exactly one of"),
            ({"image": "astronaut.png"}, "not a valid path"),
            ({"video": "motion.mp4"}, "not a valid path"),
            ({"audio": "voice.wav"}, "not a valid path"),
        ],
        ids=[
            "no_medium",
            "image_and_video",
            "image_and_audio",
            "missing_image_file",
            "missing_video_file",
            "missing_audio_file",
        ],
    )
    def test_reference_construction(self, kwargs, message):
        r"""
        A [`MiniMaxH3Reference`] resolves itself at construction: exactly one medium, bar a video's own soundtrack, and
        a path that names a file it can actually open.
        """
        with pytest.raises(ValueError, match=message):
            MiniMaxH3Reference(**kwargs)

    def test_reference_decodes_files(self, tmp_path):
        r"""
        A reference takes a path as well as in-memory media, and decodes it as it is built: the frames and the samples
        themselves, plus the rates the container reports, which is what the model conditions on.
        """
        video_path, image_path, audio_path = _media_fixtures(tmp_path)

        video = MiniMaxH3Reference(video=str(video_path))
        image = MiniMaxH3Reference(image=str(image_path))
        audio = MiniMaxH3Reference(audio=str(audio_path))

        assert video.kind == "video"
        assert video.video.shape == (FIXTURE_NUM_FRAMES, 32, 64, 3) and video.video.dtype == np.uint8
        assert video.fps == FIXTURE_FPS
        # The container carries a soundtrack, which a video reference conditions on as its own.
        assert video.has_audio and video.audio.shape[0] == 2 and video.sample_rate == FIXTURE_SAMPLE_RATE
        assert image.kind == "image" and image.image.size == (64, 32)
        assert audio.kind == "audio" and audio.audio.shape[0] == 2 and audio.sample_rate == FIXTURE_SAMPLE_RATE
        assert audio.audio.dtype == torch.float32

    def test_reference_silent_file_carries_no_soundtrack(self, tmp_path):
        r"""There is nothing to adopt from a container without an audio stream."""
        silent_path = tmp_path / "silent.mp4"
        _write_video(silent_path, with_audio=False)

        reference = MiniMaxH3Reference(video=str(silent_path))

        assert not reference.has_audio and reference.sample_rate is None
        assert reference.fps == FIXTURE_FPS

    def test_reference_rates_override_the_container(self, tmp_path):
        r"""A rate the request passes wins over the one the container reports, whose metadata may be wrong."""
        video_path, _, audio_path = _media_fixtures(tmp_path)

        video = MiniMaxH3Reference(video=str(video_path), fps=12.0, sample_rate=16000)
        audio = MiniMaxH3Reference(audio=str(audio_path), sample_rate=16000)

        # The container reports `FIXTURE_FPS` and `FIXTURE_SAMPLE_RATE`, and the request overrides both.
        assert video.fps == 12.0 and video.sample_rate == 16000
        assert audio.sample_rate == 16000

    def test_reference_file_needs_pyav(self, tmp_path, monkeypatch):
        r"""Decoding a video or an audio file is a PyAV job, and says so when PyAV is not installed."""
        video_path, image_path, audio_path = _media_fixtures(tmp_path)
        monkeypatch.setattr(packing_ref2va, "is_av_available", lambda: False)

        with pytest.raises(ImportError, match="pip install av"):
            MiniMaxH3Reference(video=str(video_path))
        with pytest.raises(ImportError, match="pip install av"):
            MiniMaxH3Reference(audio=str(audio_path))
        # An image never needs it.
        assert MiniMaxH3Reference(image=str(image_path)).image.size == (64, 32)

    def test_reference_defaults(self):
        r"""A reference knows its own modality, and defaults to MiniMax-H3's own frame rate for its frames."""
        assert MiniMaxH3Reference(video=_video_frames(2, 32)).fps == float(packing.MINIMAX_H3_FPS)
        assert MiniMaxH3Reference(image=Image.new("RGB", (32, 32))).kind == "image"
        assert not MiniMaxH3Reference(image=Image.new("RGB", (32, 32))).has_audio
        assert _reference_video(2, 32).kind == "video"
        assert _reference_audio(6.0).kind == "audio"
        assert _reference_audio(6.0).has_audio

    def test_reference_rates_default_to_the_model_rates(self):
        r"""
        A reference that leaves its rates out is taken to already be at MiniMax-H3's own: the frames flow through
        untouched, without a resampling pass and without a copy, and so do the samples of a waveform.
        """
        pipe = self.get_pipeline()
        frames = _video_frames(NUM_FRAMES, TEST_REFERENCE_IMAGE_SHORT_EDGE)
        waveform = _waveform(2.0)

        references, _ = pipe.prepare_references(
            [MiniMaxH3Reference(video=frames), MiniMaxH3Reference(audio=waveform)], NUM_FRAMES
        )

        assert np.shares_memory(references[0].frames, frames)
        assert torch.equal(references[1].waveform, waveform)

    def test_reference_sample_rate_override_resamples(self):
        r"""A waveform that says it carries another rate is resampled onto the audio VAE's own."""
        pipe = self.get_pipeline()
        waveform = _waveform(2.0)

        references, _ = pipe.prepare_references(
            [
                MiniMaxH3Reference(image=Image.new("RGB", (48, 80))),
                MiniMaxH3Reference(audio=waveform, sample_rate=AUDIO_SAMPLE_RATE // 2),
            ],
            NUM_FRAMES,
        )

        # Half the audio VAE's rate, so the same samples span twice as many of the VAE's own.
        assert references[1].waveform.shape == (2, 2 * waveform.shape[-1])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"image": Image.new("RGB", (48, 80))},
            {"video": _video_frames(24, 64), "fps": float(packing.MINIMAX_H3_FPS)},
            {"video": _video_frames(24, 64), "audio": _waveform(1.0), "sample_rate": AUDIO_SAMPLE_RATE},
        ],
        ids=["image", "video", "video_with_soundtrack"],
    )
    def test_single_reference_arguments(self, kwargs):
        r"""
        A single-reference request may name its medium on the call itself, which is the very same request as the
        one-item `references` list it is turned into.
        """
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        del inputs["references"]
        output = pipe(**inputs, **kwargs).frames

        reference_inputs = self.get_dummy_inputs()
        reference_inputs["references"] = [MiniMaxH3Reference(**kwargs)]
        assert torch.equal(output, pipe(**reference_inputs).frames)

    def test_single_reference_arguments_exclude_references(self):
        r"""The two ways of passing one reference are mutually exclusive."""
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["image"] = Image.new("RGB", (48, 80))

        with pytest.raises(ValueError, match="not both"):
            pipe(**inputs)

    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"prompt": "a robot dancing"}, "not both"),
            ({"prompt_embeds": None, "text_token_tags": None}, "Pass one of"),
            ({"text_token_tags": None}, "have to be passed together"),
            ({"text_token_tags": torch.ones(3, dtype=torch.long)}, "one tag per row"),
            ({"height": 30, "width": 30}, "multiples of 32"),
            ({"num_frames": 96}, "must be between"),
            # 346 frames are 14.417 seconds, but they are rounded up to 362, i.e. 15.083 seconds: the ceiling holds
            # for the aligned count, so this is rejected rather than silently generating too long a video.
            ({"num_frames": 346}, "rounded up to 362"),
        ],
        ids=[
            "prompt_and_prompt_embeds",
            "neither_prompt_nor_embeds",
            "embeds_without_tags",
            "tag_count_mismatch",
            "canvas_not_a_multiple_of_32",
            "shorter_than_five_seconds",
            "longer_than_fifteen_seconds_once_aligned",
        ],
    )
    def test_check_inputs(self, overrides, message):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs.update(overrides)

        with pytest.raises(ValueError, match=message):
            pipe(**inputs)

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
        "`encode_prompt` returns the per-row modality tags next to the embeddings, and both depend on the prepared "
        "references, so the harness cannot reconstruct its call from the pipeline arguments alone."
    )
    def test_encode_prompt_works_in_isolation(self):
        pass


class TestMiniMaxH3Ref2VAPipelineMemory(MiniMaxH3Ref2VAPipelineTesterConfig, MemoryTesterMixin):
    pass
