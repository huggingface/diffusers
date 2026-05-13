# Copyright 2026 The HuggingFace Team.
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

import tempfile
import unittest

import torch
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model, T5TokenizerFast

from diffusers import (
    AnimaAutoBlocks,
    AnimaModularPipeline,
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    CosmosTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
)

from ...testing_utils import enable_full_determinism, require_peft_backend
from ..test_modular_pipelines_common import ModularGuiderTesterMixin, ModularPipelineTesterMixin


enable_full_determinism()


ANIMA_TEXT2IMAGE_WORKFLOWS = {
    "text2image": [
        ("text_encoder", "AnimaTextEncoderStep"),
        ("denoise.input", "AnimaTextInputStep"),
        ("denoise.prepare_latents", "AnimaPrepareLatentsStep"),
        ("denoise.set_timesteps", "AnimaSetTimestepsStep"),
        ("denoise.denoise", "AnimaDenoiseStep"),
        ("decode.decode", "AnimaVaeDecoderStep"),
        ("decode.postprocess", "AnimaProcessImagesOutputStep"),
    ],
}


def get_dummy_components():
    torch.manual_seed(0)
    transformer = CosmosTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=2,
        mlp_ratio=2,
        text_embed_dim=16,
        adaln_lora_dim=4,
        max_size=(4, 32, 32),
        patch_size=(1, 2, 2),
        rope_scale=(1.0, 4.0, 4.0),
        concat_padding_mask=True,
        extra_pos_embed_type=None,
    )

    torch.manual_seed(0)
    vae = AutoencoderKLQwenImage(
        base_dim=24,
        z_dim=4,
        dim_mult=[1, 2, 4],
        num_res_blocks=1,
        temperal_downsample=[False, True],
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
    )

    torch.manual_seed(0)
    text_conditioner = AnimaTextConditioner(
        source_dim=16,
        target_dim=16,
        model_dim=16,
        num_layers=2,
        num_attention_heads=4,
        target_vocab_size=32128,
        min_sequence_length=16,
    )

    torch.manual_seed(0)
    text_encoder_config = Qwen3Config(
        vocab_size=152064,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        head_dim=4,
        attention_bias=False,
    )
    text_encoder = Qwen3Model(text_encoder_config).eval()
    tokenizer = Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")
    t5_tokenizer = T5TokenizerFast.from_pretrained("hf-internal-testing/tiny-random-t5")
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)

    return {
        "transformer": transformer,
        "vae": vae,
        "scheduler": scheduler,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "t5_tokenizer": t5_tokenizer,
        "text_conditioner": text_conditioner,
    }


class AnimaTextConditionerFastTests(unittest.TestCase):
    def test_conditioner_output_shape_and_padding(self):
        conditioner = AnimaTextConditioner(
            source_dim=16,
            target_dim=16,
            model_dim=16,
            num_layers=2,
            num_attention_heads=4,
            target_vocab_size=128,
            min_sequence_length=8,
        )
        source_hidden_states = torch.randn(2, 5, 16)
        target_input_ids = torch.randint(0, 128, (2, 4))
        source_attention_mask = torch.ones(2, 5)
        target_attention_mask = torch.ones(2, 4)
        target_attention_mask[1, -1] = 0

        output = conditioner(
            source_hidden_states=source_hidden_states,
            target_input_ids=target_input_ids,
            source_attention_mask=source_attention_mask,
            target_attention_mask=target_attention_mask,
        )

        self.assertEqual(output.shape, (2, 8, 16))
        self.assertTrue(torch.allclose(output[1, 3], torch.zeros_like(output[1, 3]), atol=1e-5))
        self.assertTrue(torch.allclose(output[:, 4:], torch.zeros_like(output[:, 4:]), atol=1e-5))


class TestAnimaModularPipelineFast(ModularPipelineTesterMixin, ModularGuiderTesterMixin):
    pipeline_class = AnimaModularPipeline
    pipeline_blocks_class = AnimaAutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-anima-modular-pipe"
    params = frozenset(["prompt", "height", "width", "negative_prompt"])
    batch_params = frozenset(["prompt", "negative_prompt"])
    expected_workflow_blocks = ANIMA_TEXT2IMAGE_WORKFLOWS

    def get_pipeline(self, components_manager=None, torch_dtype=torch.float32):
        pipe = self.pipeline_blocks_class().init_pipeline(components_manager=components_manager)
        pipe.update_components(**get_dummy_components())
        pipe.to(dtype=torch_dtype)
        pipe.set_progress_bar_config(disable=None)
        return pipe

    def get_dummy_inputs(self, seed=0):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return {
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "pt",
        }

    def test_inference_empty_negative_prompt(self):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["negative_prompt"] = ""
        output = pipe(**inputs).images

        assert output.shape == (1, 3, 32, 32)
        assert not torch.isnan(output).any()

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=5e-4)

    def test_save_load_components(self):
        pipe = self.get_pipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir, safe_serialization=True)
            pipe = self.pipeline_class.from_pretrained(tmpdir)
            pipe.load_components()

        assert isinstance(pipe.text_conditioner, AnimaTextConditioner)
        assert isinstance(pipe.transformer, CosmosTransformer3DModel)

    def test_lora_state_dict_conversion(self):
        state_dict = {
            "diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight": torch.randn(2, 32),
            "diffusion_model.blocks.0.self_attn.q_proj.lora_B.weight": torch.randn(32, 2),
            "diffusion_model.blocks.0.adaln_modulation_cross_attn.1.lora_A.weight": torch.randn(2, 32),
            "diffusion_model.blocks.0.adaln_modulation_cross_attn.1.lora_B.weight": torch.randn(4, 2),
            "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_A.weight": torch.randn(2, 16),
            "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_B.weight": torch.randn(16, 2),
        }

        converted_state_dict = self.pipeline_class.lora_state_dict(state_dict)

        assert "transformer.transformer_blocks.0.attn1.to_q.lora_A.weight" in converted_state_dict
        assert "transformer.transformer_blocks.0.norm2.linear_1.lora_B.weight" in converted_state_dict
        assert "text_conditioner.blocks.0.self_attn.q_proj.lora_A.weight" in converted_state_dict

    @require_peft_backend
    def test_load_lora_weights(self):
        pipe = self.get_pipeline()
        state_dict = {
            "diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight": torch.randn(2, 32),
            "diffusion_model.blocks.0.self_attn.q_proj.lora_B.weight": torch.randn(32, 2),
            "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_A.weight": torch.randn(2, 16),
            "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_B.weight": torch.randn(16, 2),
        }

        pipe.load_lora_weights(state_dict, adapter_name="dummy")

        assert "dummy" in pipe.transformer.peft_config
        assert "dummy" in pipe.text_conditioner.peft_config
