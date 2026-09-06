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

import types

import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    LLaDAImagePipeline,
    LLaDAImageQueryFormerModel,
    LLaDAImageSigVQModel,
    LLaDAImageTextProjectionModel,
    LLaDAImageTransformer2DModel,
)
from diffusers.utils import is_transformers_version

from ...testing_utils import torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


class LLaDAImagePipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = LLaDAImagePipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "image", "generation_mode", "height", "width", "guidance_scale", "prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    text_stack_component_names = ["text_encoder", "tokenizer", "queryformer", "text_projection"]
    group_offloading_leaf_level_exclude_modules = ["text_encoder"]
    group_offloading_block_level_exclude_modules = ["vae", "text_encoder"]
    output_shape = (3, 8, 8)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = LLaDAImageTransformer2DModel(
            in_channels=4,
            dim=32,
            n_layers=1,
            n_refiner_layers=1,
            n_heads=2,
            cap_feat_dim=24,
            semantic_feat_dim=20,
            axes_dims=(4, 6, 6),
            axes_lens=(2048, 32, 32),
        )
        torch.manual_seed(0)
        queryformer = LLaDAImageQueryFormerModel(
            num_queries=4,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=32,
        )
        torch.manual_seed(0)
        text_projection = LLaDAImageTextProjectionModel(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            projection_dim=24,
        )
        torch.manual_seed(0)
        sigvq = LLaDAImageSigVQModel(
            image_size=8,
            patch_size=2,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            codebook_size=32,
            codebook_embed_dim=8,
            semantic_embed_dim=20,
        )
        torch.manual_seed(0)
        vae = AutoencoderKLFlux2(
            sample_size=8,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=1,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
        )
        torch.manual_seed(0)
        text_encoder = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32000,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")

        return {
            "scheduler": FlowMatchEulerDiscreteScheduler(stochastic_sampling=False),
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "queryformer": queryformer,
            "text_projection": text_projection,
            "sigvq": sigvq,
            "transformer": transformer,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "a tiny cat",
            "height": 8,
            "width": 8,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "generator": self.get_generator(0),
            "max_sequence_length": 12,
            "output_type": "pt",
        }


class TestLLaDAImagePipeline(LLaDAImagePipelineTesterConfig, PipelineTesterMixin):
    def test_transformers_5_default_rope_is_materialized(self):
        if not is_transformers_version(">=", "5.0.0"):
            pytest.skip("This regression only affects Transformers 5 and later.")

        components = self.get_dummy_components()
        rotary_emb = torch.nn.Module()
        rotary_emb.config = types.SimpleNamespace(
            head_dim=8,
            hidden_size=16,
            num_attention_heads=2,
            partial_rotary_factor=1.0,
            rope_theta=10000.0,
        )
        rotary_emb.rope_type = "default"
        rotary_emb.register_buffer("inv_freq", torch.zeros(4), persistent=False)
        language_model = torch.nn.Module()
        language_model.rotary_emb = rotary_emb
        components["text_encoder"].model.language_model = language_model

        self.pipeline_class(**components)

        expected = torch.tensor([1.0, 0.1, 0.01, 0.001])
        torch.testing.assert_close(rotary_emb.inv_freq, expected)
        torch.testing.assert_close(rotary_emb.original_inv_freq, expected)

    def test_vq_conditioned(self):
        pipe = self.get_pipeline().to(torch_device)

        def generate_bd_image_logic(text_encoder, data, block_length, steps, gen_length, cfg_scale):
            input_ids = data["input_ids"]
            image_tokens = torch.arange(gen_length, device=input_ids.device).unsqueeze(0) + 157184
            return torch.cat([input_ids, image_tokens], dim=1)

        pipe.text_encoder.generate_bd_image_logic = types.MethodType(generate_bd_image_logic, pipe.text_encoder)
        inputs = self.get_dummy_inputs()
        inputs.update(generation_mode="vq", height=16, width=16, output_type="latent")
        output = pipe(**inputs).images
        assert output.shape == (1, 4, 8, 8)

    def test_image_editing(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs.update(image=torch.rand(1, 3, 8, 8), generation_mode="editing")
        output = pipe(**inputs).images
        assert output.shape == (1, *self.output_shape)
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("generation_mode", ["unknown", "editing"])
    def test_invalid_generation_mode_inputs(self, generation_mode):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["generation_mode"] = generation_mode
        with pytest.raises(ValueError):
            pipe(**inputs)


class TestLLaDAImagePipelineMemory(LLaDAImagePipelineTesterConfig, MemoryTesterMixin):
    pass
