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
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from diffusers import (
    AutoencoderKLWan,
    Cosmos2_5_TransferPipeline,
    CosmosControlNetModel,
    CosmosTransformer3DModel,
    UniPCMultistepScheduler,
)

from ...testing_utils import enable_full_determinism, torch_device
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin, to_np
from .cosmos_guardrail import DummyCosmosSafetyChecker


enable_full_determinism()


class Cosmos2_5_TransferWrapper(Cosmos2_5_TransferPipeline):
    @staticmethod
    def from_pretrained(*args, **kwargs):
        if "safety_checker" not in kwargs or kwargs["safety_checker"] is None:
            safety_checker = DummyCosmosSafetyChecker()
            device_map = kwargs.get("device_map", "cpu")
            torch_dtype = kwargs.get("torch_dtype")
            if device_map is not None or torch_dtype is not None:
                safety_checker = safety_checker.to(device_map, dtype=torch_dtype)
            kwargs["safety_checker"] = safety_checker
        return Cosmos2_5_TransferPipeline.from_pretrained(*args, **kwargs)


class Cosmos2_5_TransferPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = Cosmos2_5_TransferWrapper
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
        # Transformer with img_context support for Transfer2.5
        transformer = CosmosTransformer3DModel(
            in_channels=16 + 1,
            out_channels=16,
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
            controlnet_block_every_n=1,
            img_context_dim_in=32,
            img_context_num_tokens=4,
            img_context_dim_out=32,
        )

        torch.manual_seed(0)
        controlnet = CosmosControlNetModel(
            n_controlnet_blocks=2,
            in_channels=16 + 1 + 1,  # control latent channels + condition_mask + padding_mask
            latent_channels=16 + 1 + 1,  # base latent channels (16) + condition_mask (1) + padding_mask (1) = 18
            model_channels=32,
            num_attention_heads=2,
            attention_head_dim=16,
            mlp_ratio=2,
            text_embed_dim=32,
            adaln_lora_dim=4,
            patch_size=(1, 2, 2),
            max_size=(4, 32, 32),
            rope_scale=(2.0, 1.0, 1.0),
            extra_pos_embed_type="learnable",  # Match transformer's config
            img_context_dim_in=32,
            img_context_dim_out=32,
            use_crossattn_projection=False,  # Test doesn't need this projection
        )

        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=3,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
        )

        torch.manual_seed(0)
        scheduler = UniPCMultistepScheduler()

        torch.manual_seed(0)
        config = Qwen2_5_VLConfig(
            text_config={
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rope_scaling": {
                    "mrope_section": [1, 1, 2],
                    "rope_type": "default",
                    "type": "default",
                },
                "rope_theta": 1000000.0,
            },
            vision_config={
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_heads": 2,
                "out_hidden_size": 16,
            },
            hidden_size=16,
            vocab_size=152064,
            vision_end_token_id=151653,
            vision_start_token_id=151652,
            vision_token_id=151654,
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration(config)
        tokenizer = Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

        components = {
            "transformer": transformer,
            "controlnet": controlnet,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
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
            "num_frames": 3,
            "max_sequence_length": 16,
            "output_type": "pt",
        }

        return inputs

    def test_components_function(self):
        init_components = self.get_dummy_components()
        init_components = {k: v for k, v in init_components.items() if not isinstance(v, (str, int, float))}
        pipe = self.pipeline_class(**init_components)
        self.assertTrue(hasattr(pipe, "components"))
        self.assertTrue(set(pipe.components.keys()) == set(init_components.keys()))

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        video = pipe(**inputs).frames
        generated_video = video[0]
        self.assertEqual(generated_video.shape, (3, 3, 32, 32))
        self.assertTrue(torch.isfinite(generated_video).all())

    def test_inference_with_controls(self):
        """Test inference with control inputs (ControlNet)."""
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        # Add control video input - should be a video tensor
        inputs["controls"] = [torch.randn(3, 3, 32, 32)]  # num_frames, channels, height, width
        inputs["controls_conditioning_scale"] = 1.0

        video = pipe(**inputs).frames
        generated_video = video[0]
        self.assertEqual(generated_video.shape, (3, 3, 32, 32))
        self.assertTrue(torch.isfinite(generated_video).all())

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
            for tensor_name in callback_kwargs.keys():
                assert tensor_name in pipe._callback_tensor_inputs
            return callback_kwargs

        def callback_inputs_all(pipe, i, t, callback_kwargs):
            for tensor_name in pipe._callback_tensor_inputs:
                assert tensor_name in callback_kwargs
            for tensor_name in callback_kwargs.keys():
                assert tensor_name in pipe._callback_tensor_inputs
            return callback_kwargs

        inputs = self.get_dummy_inputs(torch_device)

        inputs["callback_on_step_end"] = callback_inputs_subset
        inputs["callback_on_step_end_tensor_inputs"] = ["latents"]
        _ = pipe(**inputs)[0]

        inputs["callback_on_step_end"] = callback_inputs_all
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        _ = pipe(**inputs)[0]

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
        self._test_inference_batch_single_identical(batch_size=2, expected_max_diff=1e-2)

    def test_attention_slicing_forward_pass(
        self, test_max_difference=True, test_mean_pixel_difference=True, expected_max_diff=1e-3
    ):
        if not getattr(self, "test_attention_slicing", True):
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

    def test_serialization_with_variants(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        model_components = [
            component_name
            for component_name, component in pipe.components.items()
            if isinstance(component, torch.nn.Module)
        ]
        # Remove components that aren't saved as standard diffusers models
        if "safety_checker" in model_components:
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
            # Skip components that are not loaded from disk or have special handling
            if name == "safety_checker":
                continue
            if isinstance(component, torch.nn.Module) and hasattr(component, "dtype"):
                expected_dtype = torch_dtype_dict.get(name, torch_dtype_dict.get("default", torch.float32))
                self.assertEqual(
                    component.dtype,
                    expected_dtype,
                    f"Component '{name}' has dtype {component.dtype} but expected {expected_dtype}",
                )

    def test_save_load_optional_components(self, expected_max_difference=1e-4):
        self.pipeline_class._optional_components.remove("safety_checker")
        super().test_save_load_optional_components(expected_max_difference=expected_max_difference)
        self.pipeline_class._optional_components.append("safety_checker")

    @unittest.skip(
        "The pipeline should not be runnable without a safety checker. The test creates a pipeline without passing in "
        "a safety checker, which makes the pipeline default to the actual Cosmos Guardrail. The Cosmos Guardrail is "
        "too large and slow to run on CI."
    )
    def test_encode_prompt_works_in_isolation(self):
        pass
