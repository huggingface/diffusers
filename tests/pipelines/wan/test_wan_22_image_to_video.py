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

import tempfile
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline, WanTransformer3DModel

from ...testing_utils import (
    enable_full_determinism,
    torch_device,
)
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class Wan22ImageToVideoPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = WanImageToVideoPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "return_dict",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        ]
    )
    test_xformers_attention = False
    supports_dduf = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=3,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
        )

        torch.manual_seed(0)
        scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=36,
            out_channels=16,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=2,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
        )

        torch.manual_seed(0)
        transformer_2 = WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=36,
            out_channels=16,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=2,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
        )

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer_2": transformer_2,
            "image_encoder": None,
            "image_processor": None,
            "boundary_ratio": 0.875,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        image_height = 16
        image_width = 16
        image = Image.new("RGB", (image_width, image_height))
        inputs = {
            "image": image,
            "prompt": "dance monkey",
            "negative_prompt": "negative",  # TODO
            "height": image_height,
            "width": image_width,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(
            **components,
        )
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        video = pipe(**inputs).frames
        generated_video = video[0]
        self.assertEqual(generated_video.shape, (9, 3, 16, 16))

        # fmt: off
        expected_slice = torch.tensor([0.4527, 0.4526, 0.4498, 0.4539, 0.4521, 0.4524, 0.4533, 0.4535, 0.5154,
                                       0.5353, 0.5200, 0.5174, 0.5434, 0.5301, 0.5199, 0.5216])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        self.assertTrue(
            torch.allclose(generated_slice, expected_slice, atol=1e-3),
            f"generated_slice: {generated_slice}, expected_slice: {expected_slice}",
        )

    @unittest.skip("Test not supported")
    def test_attention_slicing_forward_pass(self):
        pass

    def test_save_load_optional_components(self, expected_max_difference=1e-4):
        optional_component = ["transformer", "image_encoder", "image_processor"]

        components = self.get_dummy_components()
        for component in optional_component:
            components[component] = None
        components["boundary_ratio"] = 1.0  # for wan 2.2 14B, transformer is not used when boundary_ratio is 1.0

        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        torch.manual_seed(0)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir, safe_serialization=False)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            for component in pipe_loaded.components.values():
                if hasattr(component, "set_default_attn_processor"):
                    component.set_default_attn_processor()
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for component in optional_component:
            self.assertTrue(
                getattr(pipe_loaded, component) is None,
                f"`{component}` did not stay set to None after loading.",
            )

        inputs = self.get_dummy_inputs(generator_device)
        torch.manual_seed(0)
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(output.detach().cpu().numpy() - output_loaded.detach().cpu().numpy()).max()
        self.assertLess(max_diff, expected_max_difference)


class Wan225BImageToVideoPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = WanImageToVideoPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "return_dict",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        ]
    )
    test_xformers_attention = False
    supports_dduf = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=3,
            z_dim=48,
            in_channels=12,
            out_channels=12,
            is_residual=True,
            patch_size=2,
            latents_mean=[0.0] * 48,
            latents_std=[1.0] * 48,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            scale_factor_spatial=16,
            scale_factor_temporal=4,
            temperal_downsample=[False, True, True],
        )

        torch.manual_seed(0)
        scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=48,
            out_channels=48,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=2,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
        )

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer_2": None,
            "image_encoder": None,
            "image_processor": None,
            "boundary_ratio": None,
            "expand_timesteps": True,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        image_height = 32
        image_width = 32
        image = Image.new("RGB", (image_width, image_height))
        inputs = {
            "image": image,
            "prompt": "dance monkey",
            "negative_prompt": "negative",  # TODO
            "height": image_height,
            "width": image_width,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(
            **components,
        )
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        video = pipe(**inputs).frames
        generated_video = video[0]
        self.assertEqual(generated_video.shape, (9, 3, 32, 32))

        # fmt: off
        expected_slice = torch.tensor([[0.4833, 0.4305, 0.5100, 0.4299, 0.5056, 0.4298, 0.5052, 0.4332, 0.5550,
                                       0.6092, 0.5536, 0.5928, 0.5199, 0.5864, 0.6705, 0.5493]])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        self.assertTrue(
            torch.allclose(generated_slice, expected_slice, atol=1e-3),
            f"generated_slice: {generated_slice}, expected_slice: {expected_slice}",
        )

    @unittest.skip("Test not supported")
    def test_attention_slicing_forward_pass(self):
        pass

    def test_components_function(self):
        init_components = self.get_dummy_components()
        init_components.pop("boundary_ratio")
        init_components.pop("expand_timesteps")
        pipe = self.pipeline_class(**init_components)

        self.assertTrue(hasattr(pipe, "components"))
        self.assertTrue(set(pipe.components.keys()) == set(init_components.keys()))

    def test_save_load_optional_components(self, expected_max_difference=1e-4):
        optional_component = ["transformer_2", "image_encoder", "image_processor"]

        components = self.get_dummy_components()
        for component in optional_component:
            components[component] = None
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        torch.manual_seed(0)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir, safe_serialization=False)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            for component in pipe_loaded.components.values():
                if hasattr(component, "set_default_attn_processor"):
                    component.set_default_attn_processor()
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for component in optional_component:
            self.assertTrue(
                getattr(pipe_loaded, component) is None,
                f"`{component}` did not stay set to None after loading.",
            )

        inputs = self.get_dummy_inputs(generator_device)
        torch.manual_seed(0)
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(output.detach().cpu().numpy() - output_loaded.detach().cpu().numpy()).max()
        self.assertLess(max_diff, expected_max_difference)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=2e-3)

    @unittest.skip("Test not supported")
    def test_callback_inputs(self):
        pass
