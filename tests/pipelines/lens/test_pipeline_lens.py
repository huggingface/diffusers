# coding=utf-8
# Copyright 2025 Microsoft and HuggingFace Inc.
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
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from transformers import GptOssConfig, PreTrainedTokenizerFast

from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    LensPipeline,
    LensTransformer2DModel,
)
from diffusers.pipelines.lens.pipeline_lens import LensGptOssEncoder
from diffusers.utils.testing_utils import enable_full_determinism, torch_device

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


def _build_dummy_gptoss_tokenizer():
    vocab = {
        "[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "<|return|>": 4,
        "hello": 5, "world": 6, "a": 7, "cat": 8, "the": 9, "is": 10,
        "on": 11, "mat": 12, "painting": 13, "of": 14, "squirrel": 15,
        "eating": 16, "burger": 17, "Describe": 18, "image": 19,
    }
    tok = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.decoder = WordPieceDecoder()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok)
    fast.pad_token = "[PAD]"
    fast.eos_token = "[SEP]"
    fast.chat_template = (
        "{% for message in messages %}"
        "{{ message.content }}"
        "{% if not loop.last %} {% endif %}"
        "{% endfor %}"
    )
    return fast


class LensPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = LensPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    required_optional_params = PipelineTesterMixin.required_optional_params
    test_layerwise_casting = True
    test_group_offloading = True
    supports_dduf = False

    def get_dummy_components(self, num_layers: int = 1):
        torch.manual_seed(0)
        transformer = LensTransformer2DModel(
            patch_size=2,
            in_channels=64,
            out_channels=16,
            num_layers=num_layers,
            attention_head_dim=20,
            num_attention_heads=1,
            inner_dim=20,
            enc_hidden_dim=32,
            axes_dims_rope=(4, 8, 8),
            gate_mlp=True,
            rms_norm=True,
            multi_layer_encoder_feature=True,
            selected_layer_index=(0,),
        )
        torch.manual_seed(0)
        vae = AutoencoderKLFlux2(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            latent_channels=16,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(32,),
            layers_per_block=1,
            norm_num_groups=32,
            use_quant_conv=False,
            use_post_quant_conv=False,
        )
        scheduler = FlowMatchEulerDiscreteScheduler()

        config = GptOssConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=64,
            head_dim=16,
            layer_types=["full_attention", "full_attention"],
            vocab_size=1000,
        )
        text_encoder = LensGptOssEncoder(config)
        tokenizer = _build_dummy_gptoss_tokenizer()

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
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 16,
            "width": 16,
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

        self.assertEqual(generated_image.shape, (3, 2, 2))
        expected_image = torch.randn(3, 2, 2)
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
            for tensor_name, tensor_value in callback_kwargs.items():
                assert tensor_name in pipe._callback_tensor_inputs
            return callback_kwargs

        def callback_inputs_all(pipe, i, t, callback_kwargs):
            for tensor_name in pipe._callback_tensor_inputs:
                assert tensor_name in callback_kwargs
            for tensor_name, tensor_value in callback_kwargs.items():
                assert tensor_name in pipe._callback_tensor_inputs
            return callback_kwargs

        inputs = self.get_dummy_inputs(torch_device)
        inputs["callback_on_step_end"] = callback_inputs_subset
        inputs["callback_on_step_end_tensor_inputs"] = ["latents"]
        output = pipe(**inputs)[0]
        self.assertIsNotNone(output)

        inputs["callback_on_step_end"] = callback_inputs_all
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        output = pipe(**inputs)[0]
        self.assertIsNotNone(output)
