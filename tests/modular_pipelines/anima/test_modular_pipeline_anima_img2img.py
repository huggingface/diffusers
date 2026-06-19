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

import numpy as np
import PIL.Image
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

from ...testing_utils import enable_full_determinism
from ..test_modular_pipelines_common import ModularPipelineTesterMixin


enable_full_determinism()


ANIMA_IMG2IMG_WORKFLOWS = {
    "img2img": [
        ("text_encoder", "AnimaTextEncoderStep"),
        ("denoise.set_timesteps", "AnimaImg2ImgSetTimestepsStep"),
        ("denoise.denoise.text_conditioning", "AnimaTextConditioningStep"),
        ("denoise.denoise.input", "AnimaTextInputStep"),
        ("denoise.denoise.vae_encoder", "AnimaImg2ImgVaeEncoderStep"),
        ("denoise.denoise.denoise", "AnimaDenoiseStep"),
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


def get_dummy_image(height=32, width=32):
    image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return PIL.Image.fromarray(image_array)


class TestAnimaImg2ImgModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = AnimaModularPipeline
    pipeline_blocks_class = AnimaAutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-anima-modular-pipe"
    params = frozenset(["prompt", "image", "strength", "height", "width", "negative_prompt"])
    batch_params = frozenset(["prompt", "negative_prompt"])
    expected_workflow_blocks = ANIMA_IMG2IMG_WORKFLOWS

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
            "image": get_dummy_image(32, 32),
            "strength": 0.8,
            "generator": generator,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "pt",
        }

    def test_inference_basic(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        output = pipe(**inputs).images

        assert output.shape == (1, 3, 32, 32)
        assert not torch.isnan(output).any()

    def test_inference_strength_low(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["strength"] = 0.3
        output = pipe(**inputs).images

        assert output.shape == (1, 3, 32, 32)
        assert not torch.isnan(output).any()

    def test_inference_strength_high(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["strength"] = 0.95
        output = pipe(**inputs).images

        assert output.shape == (1, 3, 32, 32)
        assert not torch.isnan(output).any()

    def test_inference_empty_negative_prompt(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["negative_prompt"] = ""
        output = pipe(**inputs).images

        assert output.shape == (1, 3, 32, 32)
        assert not torch.isnan(output).any()

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=5e-4)