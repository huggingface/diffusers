# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

import tempfile
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    TextToVideoMSPipeline,
    UNet3DConditionModel,
)
from diffusers.utils import skip_mps, torch_device

from ...pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class TextToVideoMSPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = TextToVideoMSPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    # No `output_type`.
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "num_images_per_prompt",
            "generator",
            "latents",
            "return_dict",
            "callback",
            "callback_steps",
        ]
    )

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet3DConditionModel(
            block_out_channels=(32, 64, 64, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
            up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
            cross_attention_dim=32,
            attention_head_dim=4,
            use_linear_projection=True,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=512,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
        }
        return inputs

    def test_text_to_video_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = TextToVideoMSPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        frames = sd_pipe(**inputs).frames
        image_slice = frames[0][-3:, -3:, -1]

        assert frames[0].shape == (64, 64, 3)
        expected_slice = np.array([166, 184, 167, 118, 102, 123, 108, 93, 114])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_pix2pix_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = TextToVideoMSPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        negative_prompt = "french fries"
        frames = sd_pipe(**inputs, negative_prompt=negative_prompt).frames
        image_slice = frames[0][-3:, -3:, -1]

        assert frames[0].shape == (64, 64, 3)
        expected_slice = np.array([166, 181, 167, 119, 99, 124, 110, 94, 114])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_pix2pix_dpm_multistep(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["scheduler"] = DPMSolverMultistepScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        sd_pipe = TextToVideoMSPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        frames = sd_pipe(**inputs).frames
        image_slice = frames[0][-3:, -3:, -1]

        assert frames[0].shape == (64, 64, 3)
        expected_slice = np.array([170, 190, 180, 140, 121, 136, 121, 97, 122])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    # (todo): sayakpaul
    @unittest.skip(reason="Batching needs to be properly figured out first for this pipeline.")
    def test_inference_batch_consistent(self):
        pass

    # (todo): sayakpaul
    @unittest.skip(reason="Batching needs to be properly figured out first for this pipeline.")
    def test_inference_batch_single_identical(self):
        pass

    @unittest.skip(reason="`num_images_per_prompt` argument is not supported for this pipeline.")
    def test_num_images_per_prompt(self):
        pass

    # Overriding since the output type for this pipeline differs from that of
    # text-to-image pipelines.
    @skip_mps
    def test_attention_slicing_forward_pass(self):
        self._test_attention_slicing_forward_pass()

    def _test_attention_slicing_forward_pass(self, expected_max_diff=4e-3):
        if not self.test_attention_slicing:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # Warmup pass when using mps (see #372)
        if torch_device == "mps":
            _ = pipe(**self.get_dummy_inputs(torch_device))

        inputs = self.get_dummy_inputs(torch_device)
        output_without_slicing = pipe(**inputs).frames[0]

        pipe.enable_attention_slicing(slice_size=1)
        inputs = self.get_dummy_inputs(torch_device)
        output_with_slicing = pipe(**inputs).frames[0]

        max_diff = np.abs((output_with_slicing / 255.0) - (output_without_slicing / 255.0)).max()
        self.assertLess(max_diff, expected_max_diff, "Attention slicing should not affect the inference results")

        avg_diff = np.abs(output_without_slicing - output_without_slicing).mean()
        self.assertLess(avg_diff, 10, f"Error image deviates {avg_diff} pixels on average")

    # Overriding since the output type for this pipeline differs from that of
    # text-to-image pipelines.
    @skip_mps
    def test_dict_tuple_outputs_equivalent(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # Warmup pass when using mps (see #372)
        if torch_device == "mps":
            _ = pipe(**self.get_dummy_inputs(torch_device))

        output = pipe(**self.get_dummy_inputs(torch_device)).frames[0]
        output_tuple = pipe(**self.get_dummy_inputs(torch_device), return_dict=False)[0][0]

        max_diff = np.abs(output / 255.0 - output_tuple / 255.0).max()
        self.assertLess(max_diff, 1e-4)

    @skip_mps
    def test_progress_bar(self):
        return super().test_progress_bar()

    # Overriding since the output type for this pipeline differs from that of
    # text-to-image pipelines.
    @skip_mps
    def test_save_load_local(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # Warmup pass when using mps (see #372)
        if torch_device == "mps":
            _ = pipe(**self.get_dummy_inputs(torch_device))

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs).frames[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs).frames[0]

        max_diff = np.abs((output / 255.0) - (output_loaded / 255.0)).max()
        self.assertLess(max_diff, 1e-4)

    # Overriding since the output type for this pipeline differs from that of
    # text-to-image pipelines.
    @skip_mps
    def test_save_load_optional_components(self):
        if not hasattr(self.pipeline_class, "_optional_components"):
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # Warmup pass when using mps (see #372)
        if torch_device == "mps":
            _ = pipe(**self.get_dummy_inputs(torch_device))

        # set all optional components to None
        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs).frames[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in pipe._optional_components:
            self.assertTrue(
                getattr(pipe_loaded, optional_component) is None,
                f"`{optional_component}` did not stay set to None after loading.",
            )

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs).frames[0]

        max_diff = np.abs((output / 255.0) - (output_loaded / 255.0)).max()
        self.assertLess(max_diff, 1e-4)
