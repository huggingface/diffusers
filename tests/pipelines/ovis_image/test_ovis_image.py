# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

import torch
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    OvisImagePipeline,
    OvisImageTransformer2DModel,
)

from ...testing_utils import torch_device
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


class OvisImagePipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = OvisImagePipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    output_shape = (3, 16, 16)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = OvisImageTransformer2DModel(
            patch_size=1,
            in_channels=4,
            out_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            axes_dims_rope=(4, 4, 8),
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=1,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )
        scheduler = FlowMatchEulerDiscreteScheduler()
        tokenizer = Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")
        torch.manual_seed(0)
        text_encoder = Qwen3Model(
            Qwen3Config(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                vocab_size=tokenizer.vocab_size + 4,
                max_position_embeddings=512,
            )
        )
        return {
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer": transformer,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "a cat",
            "negative_prompt": "bad quality",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 2.0,
            "height": 16,
            "width": 16,
            "max_sequence_length": 32,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestOvisImagePipeline(OvisImagePipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self, base_pipe_output):
        # `test_output` already pins the shape; this one guards against a NaN/inf output.
        assert torch.isfinite(base_pipe_output).all()

    def test_guidance_scale_is_set(self):
        # The `guidance_scale` property reads `self._guidance_scale`, which `__call__` must initialize.
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        pipe(**inputs)
        assert pipe.guidance_scale == inputs["guidance_scale"]

    def test_max_sequence_length_is_used(self):
        # `max_sequence_length` should bound the encoded prompt length.
        pipe = self.get_pipeline().to(torch_device)
        embeds_16 = pipe.encode_prompt(
            "a cat", do_classifier_free_guidance=False, device=torch_device, max_sequence_length=16
        )[0]
        embeds_32 = pipe.encode_prompt(
            "a cat", do_classifier_free_guidance=False, device=torch_device, max_sequence_length=32
        )[0]
        assert embeds_16.shape[1] == 16
        assert embeds_32.shape[1] == 32


class TestOvisImagePipelineMemory(OvisImagePipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the OvisImage pipeline."""
