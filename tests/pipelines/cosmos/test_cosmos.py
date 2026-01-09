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

import inspect
import json
import os
import tempfile
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKLCosmos, CosmosTextToWorldPipeline, CosmosTransformer3DModel, EDMEulerScheduler

from ...testing_utils import enable_full_determinism, torch_device
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin, to_np
from .cosmos_guardrail import DummyCosmosSafetyChecker


enable_full_determinism()


class CosmosTextToWorldPipelineWrapper(CosmosTextToWorldPipeline):
    @staticmethod
    def from_pretrained(*args, **kwargs):
        kwargs["safety_checker"] = DummyCosmosSafetyChecker()
        return CosmosTextToWorldPipeline.from_pretrained(*args, **kwargs)


class CosmosTextToWorldPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = CosmosTextToWorldPipelineWrapper
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
    supports_dduf = False
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = CosmosTransformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=16,
            num_layers=2,
            mlp_ratio=2,
            text_embed_dim=32,
            adaln_lora_dim=4,
            max_size=(4, 32, 32),
            patch_size=(1, 2, 2),
            rope_scale=(2.0, 1.0, 1.0),
            concat_padding_mask=True,
            extra_pos_embed_type="learnable",
        )

        torch.manual_seed(0)
        vae = AutoencoderKLCosmos(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            encoder_block_out_channels=(8, 8, 8, 8),
            decode_block_out_channels=(8, 8, 8, 8),
            attention_resolutions=(8,),
            resolution=64,
            num_layers=2,
            patch_size=4,
            patch_type="haar",
            scaling_factor=1.0,
            spatial_compression_ratio=4,
            temporal_compression_ratio=4,
        )

        torch.manual_seed(0)
        scheduler = EDMEulerScheduler(
            sigma_min=0.002,
            sigma_max=80,
            sigma_data=0.5,
            sigma_schedule="karras",
            num_train_timesteps=1000,
            prediction_type="epsilon",
            rho=7.0,
            final_sigmas_type="sigma_min",
        )
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            # We cannot run the Cosmos Guardrail for fast tests due to the large model size
            "safety_checker": DummyCosmosSafetyChecker(),
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "height": 32,
            "width": 32,
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "pt",
        }

        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        video = pipe(**inputs).frames
        generated_video = video[0]
        self.assertEqual(generated_video.shape, (9, 3, 32, 32))

        # fmt: off
        expected_slice = torch.tensor([0.0, 0.9686, 0.8549, 0.8078, 0.0, 0.8431, 1.0, 0.4863, 0.7098, 0.1098, 0.8157, 0.4235, 0.6353, 0.2549, 0.5137, 0.5333])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        self.assertTrue(torch.allclose(generated_slice, expected_slice, atol=1e-3))

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

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(batch_size=3, expected_max_diff=1e-2)

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

    def test_vae_tiling(self, expected_diff_max: float = 0.2):
        generator_device = "cpu"
        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe.to("cpu")
        pipe.set_progress_bar_config(disable=None)

        # Without tiling
        inputs = self.get_dummy_inputs(generator_device)
        inputs["height"] = inputs["width"] = 128
        output_without_tiling = pipe(**inputs)[0]

        # With tiling
        pipe.vae.enable_tiling(
            tile_sample_min_height=96,
            tile_sample_min_width=96,
            tile_sample_stride_height=64,
            tile_sample_stride_width=64,
        )
        inputs = self.get_dummy_inputs(generator_device)
        inputs["height"] = inputs["width"] = 128
        output_with_tiling = pipe(**inputs)[0]

        self.assertLess(
            (to_np(output_without_tiling) - to_np(output_with_tiling)).max(),
            expected_diff_max,
            "VAE tiling should not affect the inference results",
        )

    def test_save_load_optional_components(self, expected_max_difference=1e-4):
        self.pipeline_class._optional_components.remove("safety_checker")
        super().test_save_load_optional_components(expected_max_difference=expected_max_difference)
        self.pipeline_class._optional_components.append("safety_checker")

    def test_serialization_with_variants(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        model_components = [
            component_name
            for component_name, component in pipe.components.items()
            if isinstance(component, torch.nn.Module)
        ]
        model_components.remove("safety_checker")
        variant = "fp16"

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir, variant=variant, safe_serialization=False)

            with open(f"{tmpdir}/model_index.json", "r") as f:
                config = json.load(f)

            for subfolder in os.listdir(tmpdir):
                if not os.path.isfile(subfolder) and subfolder in model_components:
                    folder_path = os.path.join(tmpdir, subfolder)
                    is_folder = os.path.isdir(folder_path) and subfolder in config
                    assert is_folder and any(p.split(".")[1].startswith(variant) for p in os.listdir(folder_path))

    def test_torch_dtype_dict(self):
        components = self.get_dummy_components()
        if not components:
            self.skipTest("No dummy components defined.")

        pipe = self.pipeline_class(**components)

        specified_key = next(iter(components.keys()))

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            pipe.save_pretrained(tmpdirname, safe_serialization=False)
            torch_dtype_dict = {specified_key: torch.bfloat16, "default": torch.float16}
            loaded_pipe = self.pipeline_class.from_pretrained(
                tmpdirname, safety_checker=DummyCosmosSafetyChecker(), torch_dtype=torch_dtype_dict
            )

        for name, component in loaded_pipe.components.items():
            if name == "safety_checker":
                continue
            if isinstance(component, torch.nn.Module) and hasattr(component, "dtype"):
                expected_dtype = torch_dtype_dict.get(name, torch_dtype_dict.get("default", torch.float32))
                self.assertEqual(
                    component.dtype,
                    expected_dtype,
                    f"Component '{name}' has dtype {component.dtype} but expected {expected_dtype}",
                )

    @unittest.skip(
        "The pipeline should not be runnable without a safety checker. The test creates a pipeline without passing in "
        "a safety checker, which makes the pipeline default to the actual Cosmos Guardrail. The Cosmos Guardrail is "
        "too large and slow to run on CI."
    )
    def test_encode_prompt_works_in_isolation(self):
        pass
