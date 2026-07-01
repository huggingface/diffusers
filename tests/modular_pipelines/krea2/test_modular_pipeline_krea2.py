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

import unittest

import torch
from transformers import Qwen3VLConfig, Qwen3VLModel

from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    Krea2Pipeline,
    Krea2Transformer2DModel,
)
from diffusers.modular_pipelines import Krea2AutoBlocks, Krea2ModularPipeline


class DummyTokenizerBatch:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class DummyKrea2Tokenizer:
    def __call__(
        self,
        text,
        truncation=False,
        padding=None,
        max_length=None,
        return_tensors=None,
    ):
        batch_size = len(text)
        sequence_length = max_length if padding == "max_length" else 5
        input_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long)
        return DummyTokenizerBatch(input_ids=input_ids, attention_mask=attention_mask)


class Krea2ModularPipelineFastTests(unittest.TestCase):
    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = Krea2Transformer2DModel(
            in_channels=16,
            num_layers=2,
            attention_head_dim=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=32,
            timestep_embed_dim=8,
            text_hidden_dim=16,
            num_text_layers=3,
            text_num_attention_heads=2,
            text_num_key_value_heads=1,
            text_intermediate_size=16,
            num_layerwise_text_blocks=1,
            num_refiner_text_blocks=1,
            axes_dims_rope=(4, 2, 2),
            rope_theta=1000.0,
        )

        torch.manual_seed(0)
        z_dim = 4
        vae = AutoencoderKLQwenImage(
            base_dim=z_dim * 6,
            z_dim=z_dim,
            dim_mult=[1, 2, 4],
            num_res_blocks=1,
            temperal_downsample=[False, True],
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=1.15,
            base_image_seq_len=256,
            max_image_seq_len=6400,
        )

        torch.manual_seed(0)
        config = Qwen3VLConfig(
            text_config={
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 8,
            },
            vision_config={
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_heads": 2,
                "out_hidden_size": 16,
            },
            vocab_size=152064,
        )
        text_encoder = Qwen3VLModel(config).eval()
        tokenizer = DummyKrea2Tokenizer()

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_select_layers": (0, 1, 2),
        }

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "generator": torch.Generator(device="cpu").manual_seed(seed),
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "pt",
        }

    def get_regular_pipeline(self):
        pipe = Krea2Pipeline(**self.get_dummy_components())
        pipe.to("cpu")
        pipe.set_progress_bar_config(disable=None)
        return pipe

    def get_modular_pipeline(self):
        pipe = Krea2ModularPipeline(blocks=Krea2AutoBlocks())
        pipe.update_components(**self.get_dummy_components())
        pipe.to("cpu")
        pipe.set_progress_bar_config(disable=None)
        return pipe

    def test_inference_matches_regular_pipeline(self):
        regular_pipe = self.get_regular_pipeline()
        modular_pipe = self.get_modular_pipeline()

        regular_image = regular_pipe(**self.get_dummy_inputs()).images
        modular_image = modular_pipe(**self.get_dummy_inputs(), output="images")

        assert torch.allclose(regular_image, modular_image, atol=1e-4)

    def test_inference_without_guidance_matches_regular_pipeline(self):
        regular_pipe = self.get_regular_pipeline()
        modular_pipe = self.get_modular_pipeline()

        regular_inputs = self.get_dummy_inputs()
        regular_inputs["guidance_scale"] = 0.0
        modular_inputs = self.get_dummy_inputs()
        modular_inputs["guidance_scale"] = 0.0

        regular_image = regular_pipe(**regular_inputs).images
        modular_image = modular_pipe(**modular_inputs, output="images")

        assert torch.allclose(regular_image, modular_image, atol=1e-4)

    def test_workflow_blocks(self):
        blocks = Krea2AutoBlocks()
        workflow = blocks.get_workflow("text2image")
        assert list(workflow.sub_blocks.keys()) == [
            "text_encoder",
            "denoise.input",
            "denoise.prepare_latents",
            "denoise.set_timesteps",
            "denoise.prepare_position_ids",
            "denoise.denoise",
            "denoise.after_denoise",
            "decode.decode",
            "decode.postprocess",
        ]
