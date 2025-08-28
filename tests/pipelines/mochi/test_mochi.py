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

import gc
import inspect
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKLMochi, FlowMatchEulerDiscreteScheduler, MochiPipeline, MochiTransformer3DModel

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    nightly,
    numpy_cosine_similarity_distance,
    require_big_accelerator,
    require_torch_accelerator,
    torch_device,
)
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import FasterCacheTesterMixin, FirstBlockCacheTesterMixin, PipelineTesterMixin, to_np


enable_full_determinism()


class MochiPipelineFastTests(
    PipelineTesterMixin, FasterCacheTesterMixin, FirstBlockCacheTesterMixin, unittest.TestCase
):
    pipeline_class = MochiPipeline
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
    test_layerwise_casting = True
    test_group_offloading = True

    def get_dummy_components(self, num_layers: int = 2):
        torch.manual_seed(0)
        transformer = MochiTransformer3DModel(
            patch_size=2,
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=num_layers,
            pooled_projection_dim=16,
            in_channels=12,
            out_channels=None,
            qk_norm="rms_norm",
            text_embed_dim=32,
            time_embed_dim=4,
            activation_fn="swiglu",
            max_sequence_length=16,
        )
        transformer.pos_frequencies.data = transformer.pos_frequencies.new_full(transformer.pos_frequencies.shape, 0)

        torch.manual_seed(0)
        vae = AutoencoderKLMochi(
            latent_channels=12,
            out_channels=3,
            encoder_block_out_channels=(32, 32, 32, 32),
            decoder_block_out_channels=(32, 32, 32, 32),
            layers_per_block=(1, 1, 1, 1, 1),
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler()
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

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
            "prompt": "dance monkey",
            "negative_prompt": "",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 4.5,
            "height": 16,
            "width": 16,
            # 6 * k + 1 is the recommendation
            "num_frames": 7,
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

        self.assertEqual(generated_video.shape, (7, 3, 16, 16))
        expected_video = torch.randn(7, 3, 16, 16)
        max_diff = np.abs(generated_video - expected_video).max()
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

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(batch_size=3, expected_max_diff=1e-3)

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


@nightly
@require_torch_accelerator
@require_big_accelerator
class MochiPipelineIntegrationTests(unittest.TestCase):
    prompt = "A painting of a squirrel eating a burger."

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_mochi(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload(device=torch_device)
        prompt = self.prompt

        videos = pipe(
            prompt=prompt,
            height=480,
            width=848,
            num_frames=19,
            generator=generator,
            num_inference_steps=2,
            output_type="pt",
        ).frames

        video = videos[0]
        expected_video = torch.randn(1, 19, 480, 848, 3).numpy()

        max_diff = numpy_cosine_similarity_distance(video.cpu(), expected_video)
        assert max_diff < 1e-3, f"Max diff is too high. got {video}"
