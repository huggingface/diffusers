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

import diffusers
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageLayeredPipeline,
    QwenImageTransformer2DModel,
)

from ...testing_utils import enable_full_determinism, torch_device
from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin, to_np


enable_full_determinism()


class QwenImageLayeredPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = QwenImageLayeredPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"height", "width", "cross_attention_kwargs"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = frozenset(["image"])
    image_latents_params = frozenset(["latents"])
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
        tiny_ckpt_id = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"

        torch.manual_seed(0)
        transformer = QwenImageTransformer2DModel(
            patch_size=2,
            in_channels=16,
            out_channels=4,
            num_layers=2,
            attention_head_dim=16,
            num_attention_heads=3,
            joint_attention_dim=16,
            guidance_embeds=False,
            axes_dims_rope=(8, 4, 4),
        )

        torch.manual_seed(0)
        z_dim = 4
        vae = AutoencoderKLQwenImage(
            base_dim=z_dim * 6,
            z_dim=z_dim,
            dim_mult=[1, 2, 4],
            num_res_blocks=1,
            temperal_downsample=[False, True],
            latents_mean=[0.0] * z_dim,
            latents_std=[1.0] * z_dim,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler()

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
        tokenizer = Qwen2Tokenizer.from_pretrained(tiny_ckpt_id)
        processor = Qwen2VLProcessor.from_pretrained(tiny_ckpt_id)

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "processor": processor,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        inputs = {
            "prompt": "dance monkey",
            "image": Image.new("RGB", (32, 32)),
            "negative_prompt": "bad quality",
            "generator": generator,
            "true_cfg_scale": 1.0,
            "layers": 2,
            "num_inference_steps": 2,
            "max_sequence_length": 16,
            "resolution": 640,
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
        images = pipe(**inputs).images
        self.assertEqual(len(images), 1)

        generated_layers = images[0]
        self.assertEqual(generated_layers.shape, (inputs["layers"], 3, 640, 640))

        # fmt: off
        expected_slice_layer_0 = torch.tensor([0.5752, 0.6324, 0.4913, 0.4421, 0.4917, 0.4923, 0.4790, 0.4299, 0.4029, 0.3506, 0.3302, 0.3352, 0.3579, 0.4422, 0.5086, 0.5961])
        expected_slice_layer_1 = torch.tensor([0.5103, 0.6606, 0.5652, 0.6512, 0.5900, 0.5814, 0.5873, 0.5083, 0.5058, 0.4131, 0.4321, 0.5300, 0.3507, 0.4826, 0.4745, 0.5426])
        # fmt: on

        layer_0_slice = torch.cat([generated_layers[0].flatten()[:8], generated_layers[0].flatten()[-8:]])
        layer_1_slice = torch.cat([generated_layers[1].flatten()[:8], generated_layers[1].flatten()[-8:]])

        self.assertTrue(torch.allclose(layer_0_slice, expected_slice_layer_0, atol=1e-3))
        self.assertTrue(torch.allclose(layer_1_slice, expected_slice_layer_1, atol=1e-3))

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-1):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["generator"] = self.get_generator(0)

        logger = diffusers.logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        batched_inputs = {}
        batched_inputs.update(inputs)

        for name in self.batch_params:
            if name not in inputs:
                continue

            value = inputs[name]
            if name == "prompt":
                len_prompt = len(value)
                batched_inputs[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]
                batched_inputs[name][-1] = 100 * "very long"
            else:
                batched_inputs[name] = batch_size * [value]

        if "generator" in inputs:
            batched_inputs["generator"] = [self.get_generator(i) for i in range(batch_size)]

        if "batch_size" in inputs:
            batched_inputs["batch_size"] = batch_size

        batched_inputs["num_inference_steps"] = inputs["num_inference_steps"]

        output = pipe(**inputs).images
        output_batch = pipe(**batched_inputs).images

        self.assertEqual(len(output_batch), batch_size)

        max_diff = np.abs(to_np(output_batch[0][0]) - to_np(output[0][0])).max()
        self.assertLess(max_diff, expected_max_diff)
