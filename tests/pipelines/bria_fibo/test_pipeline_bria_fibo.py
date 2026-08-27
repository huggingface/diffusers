# Copyright 2024 Bria AI and The HuggingFace Team. All rights reserved.
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

import pytest
import torch
from transformers import AutoTokenizer
from transformers.models.smollm3.modeling_smollm3 import SmolLM3Config, SmolLM3ForCausalLM

from diffusers import (
    AutoencoderKLWan,
    BriaFiboPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.models.transformers.transformer_bria_fibo import BriaFiboTransformer2DModel

from ...testing_utils import assert_tensors_close, torch_device
from ..testing_utils import (
    BasePipelineTesterConfig,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


class BriaFiboPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = BriaFiboPipeline
    required_input_params_in_call_signature = frozenset(["prompt", "height", "width", "guidance_scale"])
    batch_input_params = frozenset(["prompt"])
    output_shape = (3, 32, 32)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = BriaFiboTransformer2DModel(
            patch_size=1,
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=64,
            text_encoder_dim=32,
            pooled_projection_dim=None,
            axes_dims_rope=[0, 4, 4],
        )

        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=160,
            decoder_base_dim=256,
            num_res_blocks=2,
            out_channels=12,
            patch_size=2,
            scale_factor_spatial=16,
            scale_factor_temporal=4,
            temperal_downsample=[False, True, True],
            z_dim=16,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        torch.manual_seed(0)
        text_encoder = SmolLM3ForCausalLM(
            SmolLM3Config(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                # `vocab_size` stays at the SmolLM3 default: the pipeline hardcodes the beginning-of-text id
                # (128000) for empty prompts, so a smaller vocabulary would not be a valid text encoder here.
            )
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer": transformer,
            "vae": vae,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "{'text': 'A painting of a squirrel eating a burger'}",
            "negative_prompt": "bad, ugly",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 32,
            "width": 32,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestBriaFiboPipeline(BriaFiboPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        generated_image = image[0]
        assert generated_image.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.3804, 0.4799, 0.5112, 0.5784, 0.3970, 0.4709, 0.4027, 0.4882, 0.5627, 0.4635, 0.4718, 0.3876, 0.5741, 0.2936, 0.5912, 0.4014])
        # fmt: on

        generated_slice = generated_image.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert_tensors_close(generated_slice, expected_slice, atol=1e-3)

    @pytest.mark.skip("will not be supported due to dim-fusion")
    def test_encode_prompt_works_in_isolation(self):
        pass

    def test_bria_fibo_different_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)

        inputs = self.get_dummy_inputs()
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = "a different prompt"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = (output_same_prompt - output_different_prompts).abs().max()
        assert max_diff > 1e-6

    def test_image_output_shape(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs()

        height_width_pairs = [(32, 32), (64, 64), (32, 64)]
        for height, width in height_width_pairs:
            inputs.update({"height": height, "width": width})
            image = pipe(**inputs).images[0]
            _, output_height, output_width = image.shape
            assert (output_height, output_width) == (height, width)


class TestBriaFiboPipelineMemory(BriaFiboPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Bria FIBO pipeline."""


class TestBriaFiboPipelineLoRA(BriaFiboPipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the Bria FIBO pipeline."""

    @pytest.mark.skip(
        "`_load_lora_into_text_encoder` only infers per-module ranks for CLIP-style names "
        "(`.q_proj`/`.k_proj`/`.v_proj`/`.out_proj`/`.fc1`/`.fc2`, see `src/diffusers/loaders/lora_base.py`), so the "
        "LLaMA-style `.o_proj` on the SmolLM3 text encoder falls back to the default rank and the non-uniform "
        "`rank_pattern` this test builds cannot round-trip."
    )
    def test_simple_inference_with_partial_text_lora(self):
        pass


class TestBriaFiboPipelineLoRAMemory(BriaFiboPipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the Bria FIBO pipeline."""
