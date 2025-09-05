# coding=utf-8
# Copyright 2025 Latte Team and HuggingFace Inc.
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
import tempfile
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    FasterCacheConfig,
    LattePipeline,
    LatteTransformer3DModel,
    PyramidAttentionBroadcastConfig,
)
from diffusers.utils.import_utils import is_xformers_available

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import (
    FasterCacheTesterMixin,
    PipelineTesterMixin,
    PyramidAttentionBroadcastTesterMixin,
    to_np,
)


enable_full_determinism()


class LattePipelineFastTests(
    PipelineTesterMixin, PyramidAttentionBroadcastTesterMixin, FasterCacheTesterMixin, unittest.TestCase
):
    pipeline_class = LattePipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    required_optional_params = PipelineTesterMixin.required_optional_params
    test_layerwise_casting = True
    test_group_offloading = True

    pab_config = PyramidAttentionBroadcastConfig(
        spatial_attention_block_skip_range=2,
        temporal_attention_block_skip_range=2,
        cross_attention_block_skip_range=2,
        spatial_attention_timestep_skip_range=(100, 700),
        temporal_attention_timestep_skip_range=(100, 800),
        cross_attention_timestep_skip_range=(100, 800),
        spatial_attention_block_identifiers=["transformer_blocks"],
        temporal_attention_block_identifiers=["temporal_transformer_blocks"],
        cross_attention_block_identifiers=["transformer_blocks"],
    )

    faster_cache_config = FasterCacheConfig(
        spatial_attention_block_skip_range=2,
        temporal_attention_block_skip_range=2,
        spatial_attention_timestep_skip_range=(-1, 901),
        temporal_attention_timestep_skip_range=(-1, 901),
        unconditional_batch_skip_range=2,
        attention_weight_callback=lambda _: 0.5,
    )

    def get_dummy_components(self, num_layers: int = 1):
        torch.manual_seed(0)
        transformer = LatteTransformer3DModel(
            sample_size=8,
            num_layers=num_layers,
            patch_size=2,
            attention_head_dim=8,
            num_attention_heads=3,
            caption_channels=32,
            in_channels=4,
            cross_attention_dim=24,
            out_channels=8,
            attention_bias=True,
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=1000,
            norm_type="ada_norm_single",
            norm_elementwise_affine=False,
            norm_eps=1e-6,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL()

        scheduler = DDIMScheduler()
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        components = {
            "transformer": transformer.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder.eval(),
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
            "negative_prompt": "low quality",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "video_length": 1,
            "output_type": "pt",
            "clean_caption": False,
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

        self.assertEqual(generated_video.shape, (1, 3, 8, 8))
        expected_video = torch.randn(1, 3, 8, 8)
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

    @unittest.skip("Not supported.")
    def test_attention_slicing_forward_pass(self):
        pass

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        super()._test_xformers_attention_forwardGenerator_pass(test_mean_pixel_difference=False)

    @unittest.skip("Test not supported because `encode_prompt()` has multiple returns.")
    def test_encode_prompt_works_in_isolation(self):
        pass

    def test_save_load_optional_components(self):
        if not hasattr(self.pipeline_class, "_optional_components"):
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)

        prompt = inputs["prompt"]
        generator = inputs["generator"]

        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = pipe.encode_prompt(prompt)

        # inputs with prompt converted to embeddings
        inputs = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt": None,
            "negative_prompt_embeds": negative_prompt_embeds,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "video_length": 1,
            "mask_feature": False,
            "output_type": "pt",
            "clean_caption": False,
        }

        # set all optional components to None
        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)

        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir, safe_serialization=False)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.to(torch_device)

            for component in pipe_loaded.components.values():
                if hasattr(component, "set_default_attn_processor"):
                    component.set_default_attn_processor()

            pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in pipe._optional_components:
            self.assertTrue(
                getattr(pipe_loaded, optional_component) is None,
                f"`{optional_component}` did not stay set to None after loading.",
            )

        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(max_diff, 1.0)


@slow
@require_torch_accelerator
class LattePipelineIntegrationTests(unittest.TestCase):
    prompt = "A painting of a squirrel eating a burger."

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_latte(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = LattePipeline.from_pretrained("maxin-cn/Latte-1", torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload(device=torch_device)
        prompt = self.prompt

        videos = pipe(
            prompt=prompt,
            height=512,
            width=512,
            generator=generator,
            num_inference_steps=2,
            clean_caption=False,
        ).frames

        video = videos[0]
        expected_video = torch.randn(1, 512, 512, 3).numpy()

        max_diff = numpy_cosine_similarity_distance(video.flatten(), expected_video)
        assert max_diff < 1e-3, f"Max diff is too high. got {video.flatten()}"
