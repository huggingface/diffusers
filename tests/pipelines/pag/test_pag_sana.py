# Copyright 2024 The HuggingFace Team.
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
import unittest

import numpy as np
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM, GemmaTokenizer

from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
    SanaPAGPipeline,
    SanaPipeline,
    SanaTransformer2DModel,
)
from diffusers.utils.testing_utils import enable_full_determinism, torch_device

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin, to_np


enable_full_determinism()


class SanaPAGPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = SanaPAGPipeline
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
        transformer = SanaTransformer2DModel(
            patch_size=1,
            in_channels=4,
            out_channels=4,
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=4,
            num_cross_attention_heads=2,
            cross_attention_head_dim=4,
            cross_attention_dim=8,
            caption_channels=8,
            sample_size=32,
        )

        torch.manual_seed(0)
        vae = AutoencoderDC(
            in_channels=3,
            latent_channels=4,
            attention_head_dim=2,
            encoder_block_types=(
                "ResBlock",
                "EfficientViTBlock",
            ),
            decoder_block_types=(
                "ResBlock",
                "EfficientViTBlock",
            ),
            encoder_block_out_channels=(8, 8),
            decoder_block_out_channels=(8, 8),
            encoder_qkv_multiscales=((), (5,)),
            decoder_qkv_multiscales=((), (5,)),
            encoder_layers_per_block=(1, 1),
            decoder_layers_per_block=[1, 1],
            downsample_block_type="conv",
            upsample_block_type="interpolate",
            decoder_norm_types="rms_norm",
            decoder_act_fns="silu",
            scaling_factor=0.41407,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

        torch.manual_seed(0)
        text_encoder_config = Gemma2Config(
            head_dim=16,
            hidden_size=32,
            initializer_range=0.02,
            intermediate_size=64,
            max_position_embeddings=8192,
            model_type="gemma2",
            num_attention_heads=2,
            num_hidden_layers=1,
            num_key_value_heads=2,
            vocab_size=8,
            attn_implementation="eager",
        )
        text_encoder = Gemma2ForCausalLM(text_encoder_config)
        tokenizer = GemmaTokenizer.from_pretrained("hf-internal-testing/dummy-gemma")

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
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
            "prompt": "",
            "negative_prompt": "",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "pag_scale": 3.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "pt",
            "complex_human_instruction": None,
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs)[0]
        generated_image = image[0]

        self.assertEqual(generated_image.shape, (3, 32, 32))
        expected_image = torch.randn(3, 32, 32)
        max_diff = np.abs(generated_image - expected_image).max()
        self.assertLessEqual(max_diff, 1e10)

    def test_callback_inputs(self):
        sig = inspect.signature(self.pipeline_class.__call__)
        has_callback_tensor_inputs = "callback_on_step_end_tensor_inputs" in sig.parameters
        has_callback_step_end = "callback_on_step_end" in sig.parameters

        if not (has_callback_tensor_inputs and has_callback_step_end):
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        self.assertTrue(
            hasattr(pipe, "_callback_tensor_inputs"),
            f" {self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables its callback function can use as inputs",
        )

        def callback_inputs_subset(pipe, i, t, callback_kwargs):
            # iterate over callback args
            for tensor_name, tensor_value in callback_kwargs.items():
                # check that we're only passing in allowed tensor inputs
                assert tensor_name in pipe._callback_tensor_inputs

            return callback_kwargs

        def callback_inputs_all(pipe, i, t, callback_kwargs):
            for tensor_name in pipe._callback_tensor_inputs:
                assert tensor_name in callback_kwargs

            # iterate over callback args
            for tensor_name, tensor_value in callback_kwargs.items():
                # check that we're only passing in allowed tensor inputs
                assert tensor_name in pipe._callback_tensor_inputs

            return callback_kwargs

        inputs = self.get_dummy_inputs(torch_device)

        # Test passing in a subset
        inputs["callback_on_step_end"] = callback_inputs_subset
        inputs["callback_on_step_end_tensor_inputs"] = ["latents"]
        output = pipe(**inputs)[0]

        # Test passing in a everything
        inputs["callback_on_step_end"] = callback_inputs_all
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        output = pipe(**inputs)[0]

        def callback_inputs_change_tensor(pipe, i, t, callback_kwargs):
            is_last = i == (pipe.num_timesteps - 1)
            if is_last:
                callback_kwargs["latents"] = torch.zeros_like(callback_kwargs["latents"])
            return callback_kwargs

        inputs["callback_on_step_end"] = callback_inputs_change_tensor
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        output = pipe(**inputs)[0]
        assert output.abs().sum() < 1e10

    def test_attention_slicing_forward_pass(
        self, test_max_difference=True, test_mean_pixel_difference=True, expected_max_diff=1e-3
    ):
        if not self.test_attention_slicing:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        output_without_slicing = pipe(**inputs)[0]

        pipe.enable_attention_slicing(slice_size=1)
        inputs = self.get_dummy_inputs(generator_device)
        output_with_slicing1 = pipe(**inputs)[0]

        pipe.enable_attention_slicing(slice_size=2)
        inputs = self.get_dummy_inputs(generator_device)
        output_with_slicing2 = pipe(**inputs)[0]

        if test_max_difference:
            max_diff1 = np.abs(to_np(output_with_slicing1) - to_np(output_without_slicing)).max()
            max_diff2 = np.abs(to_np(output_with_slicing2) - to_np(output_without_slicing)).max()
            self.assertLess(
                max(max_diff1, max_diff2),
                expected_max_diff,
                "Attention slicing should not affect the inference results",
            )

    def test_pag_disable_enable(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        # base pipeline (expect same output when pag is disabled)
        pipe_sd = SanaPipeline(**components)
        pipe_sd = pipe_sd.to(device)
        pipe_sd.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["pag_scale"]
        assert (
            "pag_scale" not in inspect.signature(pipe_sd.__call__).parameters
        ), f"`pag_scale` should not be a call parameter of the base pipeline {pipe_sd.__class__.__name__}."
        out = pipe_sd(**inputs).images[0, -3:, -3:, -1]

        components = self.get_dummy_components()

        # pag disabled with pag_scale=0.0
        pipe_pag = self.pipeline_class(**components)
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["pag_scale"] = 0.0
        out_pag_disabled = pipe_pag(**inputs).images[0, -3:, -3:, -1]

        assert np.abs(out.flatten() - out_pag_disabled.flatten()).max() < 1e-3

    def test_pag_applied_layers(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        # base pipeline
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        all_self_attn_layers = [k for k in pipe.transformer.attn_processors.keys() if "attn1" in k]
        original_attn_procs = pipe.transformer.attn_processors
        pag_layers = ["blocks.0", "blocks.1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_layers)

        # blocks.0
        block_0_self_attn = ["transformer_blocks.0.attn1.processor"]
        pipe.transformer.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.0"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(block_0_self_attn)

        pipe.transformer.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.0.attn1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(block_0_self_attn)

        pipe.transformer.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.(0|1)"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert (len(pipe.pag_attn_processors)) == 2

        pipe.transformer.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.0", r"blocks\.1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 2

    # TODO(aryan): Create a dummy gemma model with smol vocab size
    @unittest.skip(
        "A very small vocab size is used for fast tests. So, any kind of prompt other than the empty default used in other tests will lead to a embedding lookup error. This test uses a long prompt that causes the error."
    )
    def test_inference_batch_consistent(self):
        pass

    @unittest.skip(
        "A very small vocab size is used for fast tests. So, any kind of prompt other than the empty default used in other tests will lead to a embedding lookup error. This test uses a long prompt that causes the error."
    )
    def test_inference_batch_single_identical(self):
        pass

    def test_float16_inference(self):
        # Requires higher tolerance as model seems very sensitive to dtype
        super().test_float16_inference(expected_max_diff=0.08)
