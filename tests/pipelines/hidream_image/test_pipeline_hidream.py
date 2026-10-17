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
from transformers import (
    AutoConfig,
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    LlamaForCausalLM,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    HiDreamImagePipeline,
    HiDreamImageTransformer2DModel,
)

from ...testing_utils import assert_tensors_close, enable_full_determinism
from ..testing_utils import (
    BasePipelineTesterConfig,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class HiDreamImagePipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = HiDreamImagePipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "negative_prompt"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (3, 128, 128)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = HiDreamImageTransformer2DModel(
            patch_size=2,
            in_channels=4,
            out_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=4,
            caption_channels=[32, 16],
            text_emb_dim=64,
            num_routed_experts=4,
            num_activated_experts=2,
            axes_dims_rope=(4, 2, 2),
            max_resolution=(32, 32),
            llama_layers=(0, 1),
        ).eval()
        torch.manual_seed(0)
        vae = AutoencoderKL(scaling_factor=0.3611, shift_factor=0.1159)
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
            max_position_embeddings=128,
        )

        torch.manual_seed(0)
        text_encoder = CLIPTextModelWithProjection(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_2 = CLIPTextModelWithProjection(clip_text_encoder_config)

        torch.manual_seed(0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder_3 = T5EncoderModel(config).eval()

        torch.manual_seed(0)
        text_encoder_4 = LlamaForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        text_encoder_4.generation_config.pad_token_id = 1
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_3 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer_4 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "text_encoder_3": text_encoder_3,
            "tokenizer_3": tokenizer_3,
            "text_encoder_4": text_encoder_4,
            "tokenizer_4": tokenizer_4,
            "transformer": transformer,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestHiDreamImagePipeline(HiDreamImagePipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs())[0]
        generated_image = image[0]
        assert generated_image.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.3841, 0.5585, 0.4529, 0.5123, 0.4314, 0.5122, 0.4885, 0.4554, 0.4653, 0.3369, 0.1799, 0.6126, 0.6202, 0.2871, 0.5525, 0.5524])
        # fmt: on

        generated_slice = generated_image.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert_tensors_close(generated_slice, expected_slice, atol=5e-3)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=3e-4):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)


class TestHiDreamImagePipelineMemory(HiDreamImagePipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the HiDream Image pipeline."""


class TestHiDreamImagePipelineLoRA(HiDreamImagePipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the HiDream Image pipeline."""


class TestHiDreamImagePipelineLoRAMemory(HiDreamImagePipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the HiDream Image pipeline."""
