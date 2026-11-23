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


import numpy as np
import PIL.Image
import pytest
import torch
from transformers import Gemma2Config, Gemma2Model, GemmaTokenizer

from diffusers import (
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
    SanaWMPipeline,
    SanaWMPipelineOutput,
    SanaWMTransformer3DModel,
)
from diffusers.pipelines.sana_wm.cam_utils import action_string_to_c2w

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
