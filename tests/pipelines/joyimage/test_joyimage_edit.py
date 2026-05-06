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

import unittest
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    JoyImageEditPipeline,
    JoyImageEditTransformer3DModel,
)
from diffusers.hooks import apply_group_offloading

from ...testing_utils import enable_full_determinism, require_torch_accelerator, torch_device
from ..pipeline_params import TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class JoyImageEditPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = JoyImageEditPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = frozenset(["prompt", "image"])
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

    def setUp(self):
        super().setUp()
        self._bucket_patcher = patch(
            "diffusers.pipelines.joyimage.image_processor.find_best_bucket",
            return_value=(32, 32),
        )
        self._bucket_patcher.start()

    def tearDown(self):
        self._bucket_patcher.stop()
        super().tearDown()

    def get_dummy_components(self):
        tiny_ckpt_id = "huangfeice/tiny-random-Qwen3VLForConditionalGeneration"

        torch.manual_seed(0)
        transformer = JoyImageEditTransformer3DModel(
            patch_size=[1, 2, 2],
            in_channels=16,
            hidden_size=32,
            num_attention_heads=2,
            text_dim=16,
            num_layers=1,
            rope_dim_list=[4, 6, 6],
            theta=256,
        )

        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=3,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        processor = Qwen3VLProcessor.from_pretrained(tiny_ckpt_id)
        processor.image_processor.min_pixels = 4 * 28 * 28
        processor.image_processor.max_pixels = 4 * 28 * 28

        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(tiny_ckpt_id)
        text_encoder.resize_token_embeddings(len(processor.tokenizer))

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": processor.tokenizer,
            "processor": processor,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        inputs = {
            "prompt": "a cat sitting on a bench",
            "image": Image.new("RGB", (32, 32)),
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 32,
            "width": 32,
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
        image = pipe(**inputs).images
        generated_image = image[0]

        self.assertEqual(generated_image.shape, (3, 32, 32))

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(batch_size=3, expected_max_diff=1e-1)

    @unittest.skip("num_images_per_prompt not applicable: each prompt is bound to a reference image")
    def test_num_images_per_prompt(self):
        pass

    @unittest.skip("Test not supported")
    def test_attention_slicing_forward_pass(self):
        pass

    @pytest.mark.xfail(condition=True, reason="Preconfigured embeddings need to be revisited.", strict=False)
    def test_encode_prompt_works_in_isolation(self, extra_required_param_value_dict=None, atol=1e-4, rtol=1e-4):
        super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict, atol, rtol)

    @require_torch_accelerator
    def test_group_offloading_inference(self):
        # Qwen3VLForConditionalGeneration (the text encoder) is incompatible with leaf_level group
        # offloading. Its Qwen3VLVisionModel.fast_pos_embed_interpolate reads
        # `self.pos_embed.weight.device` to create intermediate tensors before the Embedding's
        # pre_forward hook fires, so the intermediate tensors land on CPU while hidden_states
        # (produced by the Conv3d patch_embed) land on CUDA, causing a device mismatch.
        #
        # block_level works correctly: since Qwen3VLForConditionalGeneration has no ModuleList as a
        # direct child, the entire model forms one unmatched group that onloads atomically before any
        # submodule code runs, so pos_embed.weight.device is CUDA by the time it is read.
        #
        # For leaf_level we therefore move the text encoder to the target device directly (the same
        # pattern the base test already uses for the VAE) and only apply leaf_level offloading to
        # the diffusers-native transformer.
        if not self.test_group_offloading:
            return

        def create_pipe():
            torch.manual_seed(0)
            components = self.get_dummy_components()
            pipe = self.pipeline_class(**components)
            pipe.set_progress_bar_config(disable=None)
            return pipe

        def run_forward(pipe):
            torch.manual_seed(0)
            inputs = self.get_dummy_inputs(torch_device)
            return pipe(**inputs)[0]

        pipe = create_pipe().to(torch_device)
        output_without_group_offloading = run_forward(pipe)

        # block_level: the full text encoder becomes one group (no direct ModuleList children), so
        # the atomc onload/offload is safe.
        pipe = create_pipe()
        for component_name in ["transformer", "text_encoder"]:
            component = getattr(pipe, component_name, None)
            if component is None:
                continue
            if hasattr(component, "enable_group_offload"):
                component.enable_group_offload(
                    torch.device(torch_device), offload_type="block_level", num_blocks_per_group=1
                )
            else:
                apply_group_offloading(
                    component,
                    onload_device=torch.device(torch_device),
                    offload_type="block_level",
                    num_blocks_per_group=1,
                )
        pipe.vae.to(torch_device)
        output_with_block_level = run_forward(pipe)

        pipe = create_pipe()
        pipe.transformer.enable_group_offload(torch.device(torch_device), offload_type="leaf_level")
        pipe.text_encoder.to(torch_device)
        pipe.vae.to(torch_device)
        output_with_leaf_level = run_forward(pipe)

        if torch.is_tensor(output_without_group_offloading):
            output_without_group_offloading = output_without_group_offloading.detach().cpu().numpy()
            output_with_block_level = output_with_block_level.detach().cpu().numpy()
            output_with_leaf_level = output_with_leaf_level.detach().cpu().numpy()

        self.assertTrue(np.allclose(output_without_group_offloading, output_with_block_level, atol=1e-4))
        self.assertTrue(np.allclose(output_without_group_offloading, output_with_leaf_level, atol=1e-4))

    @unittest.skip("Qwen3VLForConditionalGeneration does not support leaf-level group offloading")
    def test_pipeline_level_group_offloading_inference(self):
        pass

    @unittest.skip("Qwen3VLForConditionalGeneration does not support sequential CPU offloading")
    def test_sequential_cpu_offload_forward_pass(self):
        pass

    @unittest.skip("Qwen3VLForConditionalGeneration does not support sequential CPU offloading")
    def test_sequential_offload_forward_pass_twice(self):
        pass
