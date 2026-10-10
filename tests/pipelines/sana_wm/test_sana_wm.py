# Copyright 2026 The HuggingFace Team and SANA-WM Authors. All rights reserved.
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

import inspect

import numpy as np
import PIL.Image
import pytest
import torch
from transformers import Gemma2Config, Gemma2Model, GemmaTokenizer

import diffusers
from diffusers import (
    AutoencoderKLLTX2Video,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    SanaWMPipeline,
    SanaWMPipelineOutput,
    SanaWMTransformer3DModel,
)
from diffusers.pipelines.sana_wm import SanaWMLTX2Refiner
from diffusers.pipelines.sana_wm.cam_utils import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    action_string_to_c2w,
    resize_and_center_crop,
    snap_num_frames,
    transform_intrinsics_for_crop,
)

from ...testing_utils import assert_tensors_close, enable_full_determinism, torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


SINGLE_VIDEO_SKIP_REASON = (
    "SanaWMPipeline generates exactly one video per call (one first-frame image + one camera trajectory); "
    "`prompt`/`image`/`c2w` take no batch dimension and there is no `num_videos_per_prompt`."
)


class SanaWMPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = SanaWMPipeline
    required_input_params_in_call_signature = frozenset(
        [
            "image",
            "prompt",
            "negative_prompt",
            "c2w",
            "action",
            "intrinsics",
            "height",
            "width",
            "num_frames",
            "guidance_scale",
            "prompt_embeds",
            "prompt_attention_mask",
            "negative_prompt_embeds",
            "negative_prompt_attention_mask",
        ]
    )
    # One first-frame image + one camera trajectory per call: nothing is batched.
    batch_input_params = frozenset()
    # The pipeline has no `num_images_per_prompt` (single video per call) and samples its own noise around the
    # VAE-encoded first frame, so it takes no user `latents` either.
    optional_input_params = frozenset(["num_inference_steps", "generator", "output_type", "return_dict"])
    # `frames` is the un-batched `(num_frames, channels, height, width)` video for `output_type="pt"`, so a single
    # element of `pipe(...)[0]` is one `(channels, height, width)` frame.
    output_shape = (3, 32, 32)

    # Dummy video geometry. The tiny LTX-2 VAE below compresses 2x spatially and 2x temporally.
    num_frames = 5
    height = 32
    width = 32

    def get_dummy_components(self):
        tokenizer = GemmaTokenizer.from_pretrained("hf-internal-testing/dummy-gemma")

        torch.manual_seed(0)
        text_encoder_config = Gemma2Config(
            head_dim=16,
            hidden_size=8,
            initializer_range=0.02,
            intermediate_size=16,
            max_position_embeddings=512,
            num_attention_heads=2,
            num_hidden_layers=1,
            num_key_value_heads=1,
            # Match the tokenizer so real prompts (and the chi-prompt prefix) can be embedded.
            vocab_size=len(tokenizer),
        )
        text_encoder = Gemma2Model(text_encoder_config)

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

        # `num_layers=2` with `softmax_every_n=2` covers both block variants (GDN and softmax camera attention).
        torch.manual_seed(0)
        transformer = SanaWMTransformer3DModel(
            in_channels=4,
            num_layers=2,
            hidden_size=32,
            num_attention_heads=2,
            softmax_every_n=2,
            linear_head_dim=16,
            t_kernel_size=3,
            conv_kernel_size=4,
            caption_channels=text_encoder_config.hidden_size,
            mlp_ratio=2.0,
            # Plücker rays are packed per VAE temporal chunk: 6 dims * temporal compression ratio.
            chunk_plucker_channels=6 * vae.temporal_compression_ratio,
            chunk_plucker_post_attn_blocks=2,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    def get_dummy_image(self):
        # A non-square source image so the resize + center-crop (and the matching intrinsics rescale) is exercised.
        pixels = np.random.RandomState(0).randint(0, 256, size=(40, 48, 3), dtype=np.uint8)
        return PIL.Image.fromarray(pixels)

    def get_dummy_inputs(self):
        return {
            "image": self.get_dummy_image(),
            "prompt": "A car driving across a desert",
            # `w-4` rolls out 4 forward steps plus the identity anchor = `num_frames` poses.
            "action": "w-4",
            # [fx, fy, cx, cy] in the pixel coordinates of the 48x40 source image.
            "intrinsics": [40.0, 40.0, 24.0, 20.0],
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "max_sequence_length": 16,
            # A short chi-prompt keeps the tokenized prefix small (the default is the long release prefix).
            "chi_prompt": ["Describe the scene: "],
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` videos are `(num_frames, channels, height, width)`.
            "output_type": "pt",
        }


class TestSanaWMPipeline(SanaWMPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        pipe = self.get_pipeline().to(torch_device)

        output = pipe(**self.get_dummy_inputs())

        assert isinstance(output, SanaWMPipelineOutput)
        assert output.frames.shape == (self.num_frames, *self.output_shape)
        assert not torch.isnan(output.frames).any()
        assert output.c2w.shape == (self.num_frames, 4, 4)
        # (batch, latent channels, (num_frames - 1) // 2 + 1, height // 2, width // 2)
        assert output.latent.shape == (1, 4, 3, self.height // 2, self.width // 2)

    def test_latent_output_matches_returned_latent(self):
        pipe = self.get_pipeline().to(torch_device)

        latent_output = pipe(**{**self.get_dummy_inputs(), "output_type": "latent"})
        video_output = pipe(**self.get_dummy_inputs())

        assert_tensors_close(latent_output.frames, video_output.latent, atol=1e-5, msg="Latent outputs differ.")

    def test_c2w_matches_equivalent_action(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        output_action = pipe(**inputs)[0]

        inputs = self.get_dummy_inputs()
        inputs["c2w"] = action_string_to_c2w(inputs.pop("action"))
        output_c2w = pipe(**inputs)[0]

        assert_tensors_close(output_c2w, output_action, atol=1e-5, msg="`c2w` and the equivalent `action` differ.")

    def test_intrinsics_matrix_matches_vector(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        output_vec = pipe(**inputs)[0]

        inputs = self.get_dummy_inputs()
        fx, fy, cx, cy = inputs["intrinsics"]
        inputs["intrinsics"] = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        output_mat = pipe(**inputs)[0]

        assert_tensors_close(output_mat, output_vec, atol=1e-5, msg="3x3 `K` and `[fx, fy, cx, cy]` differ.")

    @pytest.mark.parametrize(
        "overrides",
        [
            {"action": None},
            {"c2w": np.broadcast_to(np.eye(4, dtype=np.float32), (5, 4, 4)).copy()},
            {"intrinsics": None},
            {"action": None, "c2w": np.eye(4, dtype=np.float32)},
        ],
        ids=["no-trajectory", "both-c2w-and-action", "no-intrinsics", "bad-c2w-shape"],
    )
    def test_check_inputs_rejects_bad_camera_inputs(self, overrides):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs.update(overrides)

        with pytest.raises(ValueError):
            pipe(**inputs)

    @pytest.mark.skip(reason=SINGLE_VIDEO_SKIP_REASON)
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason=SINGLE_VIDEO_SKIP_REASON)
    def test_inference_batch_single_identical(self):
        pass


class TestSanaWMPipelineMemory(SanaWMPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SANA-WM pipeline."""


class TestSanaWMCamUtils:
    """Camera helpers: action DSL, intrinsics math, resize-and-crop and frame-count snapping."""

    def test_action_dsl_forward_only(self):
        c2w = action_string_to_c2w("w-5", translation_speed=0.1)
        # 5 action frames + leading identity = 6 total
        assert c2w.shape == (6, 4, 4)
        assert c2w.dtype == np.float32
        # First frame is identity (the anchor).
        np.testing.assert_allclose(c2w[0], np.eye(4, dtype=np.float32), atol=1e-6)
        # 'w' moves forward (+Z in OpenCV convention).
        assert float(c2w[-1, 2, 3]) == pytest.approx(0.5, abs=1e-5)
        # No yaw / pitch -> rotation is identity throughout.
        for i in range(c2w.shape[0]):
            np.testing.assert_allclose(c2w[i, :3, :3], np.eye(3), atol=1e-6)

    def test_action_dsl_concat_segments(self):
        c2w = action_string_to_c2w("w-3,a-2", translation_speed=0.1)
        assert c2w.shape == (6, 4, 4)  # 3 + 2 + identity anchor

    @pytest.mark.parametrize(
        "action",
        ["", "x-5", "w-0"],
        ids=["empty", "unknown-key", "zero-length-segment"],
    )
    def test_action_dsl_rejects_bad_input(self, action):
        with pytest.raises(ValueError):
            action_string_to_c2w(action)

    def test_action_dsl_none_segment_is_idle(self):
        c2w = action_string_to_c2w("none-3", translation_speed=0.1)
        assert c2w.shape == (4, 4, 4)
        # No motion -> all frames are identity.
        for i in range(c2w.shape[0]):
            np.testing.assert_allclose(c2w[i], np.eye(4), atol=1e-6)

    def test_transform_intrinsics_for_crop_scalar(self):
        # (fx, fy, cx, cy) for a 1000x500 source, resized to 1280x704, then
        # center-cropped to 1280x704 (no extra crop offset).
        intr = np.array([800.0, 800.0, 500.0, 250.0], dtype=np.float32)
        out = transform_intrinsics_for_crop(intr, src_size=(1000, 500), resized_size=(1280, 704), crop_offset=(0, 0))
        assert float(out[0]) == pytest.approx(800.0 * 1280 / 1000, abs=1e-4)  # fx scales with x
        assert float(out[1]) == pytest.approx(800.0 * 704 / 500, abs=1e-4)
        assert float(out[2]) == pytest.approx(500.0 * 1280 / 1000, abs=1e-4)
        assert float(out[3]) == pytest.approx(250.0 * 704 / 500, abs=1e-4)

    def test_transform_intrinsics_for_crop_with_offset(self):
        intr = np.array([800.0, 800.0, 500.0, 250.0], dtype=np.float32)
        # After resize, an extra crop offset shifts the principal point.
        out = transform_intrinsics_for_crop(
            intr, src_size=(1000, 500), resized_size=(2000, 1000), crop_offset=(360, 148)
        )
        assert float(out[2]) == pytest.approx(500.0 * 2.0 - 360.0, abs=1e-4)
        assert float(out[3]) == pytest.approx(250.0 * 2.0 - 148.0, abs=1e-4)

    def test_resize_and_center_crop_default_target(self):
        src = PIL.Image.new("RGB", (1691, 930))
        cropped, src_size, resized_size, crop_offset = resize_and_center_crop(src)
        assert cropped.size == (TARGET_WIDTH, TARGET_HEIGHT)
        assert src_size == (1691, 930)
        # Resize preserves aspect; one of the resized dimensions equals the target.
        resized_width, resized_height = resized_size
        assert resized_width >= TARGET_WIDTH
        assert resized_height >= TARGET_HEIGHT
        crop_left, crop_top = crop_offset
        assert crop_left >= 0
        assert crop_top >= 0
        # Center crop produces 0 offset on the dimension that hit the target exactly.
        assert crop_left == 0 or crop_top == 0

    # The LTX-2 VAE requires a (8k + 1)-shaped temporal dim, so ``snap_num_frames`` rounds to
    # the nearest such value (ties break to the ceil).
    @pytest.mark.parametrize("num_frames", [1, 9, 17, 81, 161, 321, 801])
    def test_snap_num_frames_is_a_noop_on_8k_plus_1(self, num_frames):
        assert snap_num_frames(num_frames) == num_frames

    @pytest.mark.parametrize(
        ("num_frames", "expected"),
        [
            (2, 1),
            (10, 9),  # 10 is closer to 9 than 17
            (80, 81),  # 80 is closer to 81 than 73
            (100, 97),  # 100 is closer to 97 than 105
        ],
    )
    def test_snap_num_frames_to_8k_plus_1(self, num_frames, expected):
        assert snap_num_frames(num_frames) == expected

    def test_snap_num_frames_respects_upper_bound(self):
        # ``upper_bound`` caps the result (the snap falls back to the floor).
        assert snap_num_frames(100, upper_bound=100) <= 100
        assert snap_num_frames(100, upper_bound=100) == 97


class TestSanaWMRegistration:
    """Verify the SANA-WM symbols are reachable through the public diffusers surface."""

    @pytest.mark.parametrize(
        "name", ["SanaWMPipeline", "SanaWMTransformer3DModel", "SanaWMLTX2Refiner", "SanaWMPipelineOutput"]
    )
    def test_top_level_symbols(self, name):
        assert hasattr(diffusers, name), f"{name!r} not exported from diffusers top-level"

    def test_pipeline_output_dataclass(self):
        frames = np.zeros((3, 8, 8, 3), dtype=np.float32)
        c2w = np.broadcast_to(np.eye(4, dtype=np.float32), (3, 4, 4)).copy()
        latent = torch.zeros(1, 16, 1, 4, 4)
        output = SanaWMPipelineOutput(frames=frames, c2w=c2w, latent=latent)
        assert tuple(output.frames.shape) == (3, 8, 8, 3)
        assert tuple(output.c2w.shape) == (3, 4, 4)
        assert tuple(output.latent.shape) == (1, 16, 1, 4, 4)

    def test_refiner_is_pipeline_with_ar_call_defaults(self):
        # The refiner is a standalone DiffusionPipeline.
        assert issubclass(SanaWMLTX2Refiner, DiffusionPipeline)

        # Its denoising entry point is ``__call__`` with the canonical AR defaults.
        params = inspect.signature(SanaWMLTX2Refiner.__call__).parameters
        assert "block_size" in params
        assert "kv_max_frames" in params
        # AR mode is on by default.
        assert params["block_size"].default == 3
        assert params["kv_max_frames"].default == 11

    def test_pipeline_call_takes_generator_not_seed(self):
        # Pipelines take a `generator`; `seed` shortcuts are not part of the diffusers interface.
        params = inspect.signature(SanaWMPipeline.__call__).parameters
        assert "seed" not in params
        assert "refiner_seed" not in params

    def test_refiner_is_not_a_component_of_the_base_pipeline(self):
        # The two stages run as separate pipelines, so the base one neither holds a
        # refiner nor exposes a switch for it.
        assert "refiner" not in inspect.signature(SanaWMPipeline.__init__).parameters
        assert "use_refiner" not in inspect.signature(SanaWMPipeline.__call__).parameters

    def test_refiner_takes_an_optional_vae_for_decoding(self):
        # With a `vae` the refiner returns video; without one, refined latents.
        params = inspect.signature(SanaWMLTX2Refiner.__init__).parameters
        assert "vae" in params
        assert params["vae"].default is None
        assert "output_type" in inspect.signature(SanaWMLTX2Refiner.__call__).parameters
