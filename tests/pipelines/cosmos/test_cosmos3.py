# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import json
from types import SimpleNamespace

import torch

from diffusers import AutoencoderKLWan, Cosmos3OmniPipeline, Cosmos3OmniTransformer, UniPCMultistepScheduler


class DummyTokenizer:
    eos_token_id = 11

    def __init__(self):
        self.conversations = []

    def convert_tokens_to_ids(self, token):
        return {"<|vision_start|>": 20}.get(token, 0)

    def apply_chat_template(self, conversations, **kwargs):
        self.conversations.append(conversations)
        return SimpleNamespace(input_ids=[1, 2])


def get_dummy_pipeline(**kwargs):
    transformer = Cosmos3OmniTransformer(
        hidden_size=16,
        intermediate_size=32,
        head_dim=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        latent_channel=1,
        latent_patch_size=1,
        patch_latent_dim=1,
        vocab_size=32,
    )

    torch.manual_seed(0)
    vae = AutoencoderKLWan(
        base_dim=3,
        z_dim=16,
        dim_mult=[1, 1, 1, 1],
        num_res_blocks=1,
        temperal_downsample=[False, True, True],
    )

    return Cosmos3OmniPipeline(
        transformer=transformer,
        text_tokenizer=DummyTokenizer(),
        vae=vae,
        scheduler=UniPCMultistepScheduler(),
        enable_safety_checker=False,
        **kwargs,
    )


def test_cosmos3_pipeline_saves_edge_configuration(tmp_path):
    pipeline = get_dummy_pipeline(default_use_system_prompt=False, use_native_flow_schedule=True)

    assert not pipeline.config.enable_safety_checker
    assert not pipeline.config.default_use_system_prompt
    assert pipeline.config.use_native_flow_schedule
    assert pipeline.safety_checker is None

    pipeline.save_config(tmp_path)
    model_index = json.loads((tmp_path / "model_index.json").read_text())
    assert model_index["enable_safety_checker"] is False
    assert model_index["default_use_system_prompt"] is False
    assert model_index["use_native_flow_schedule"] is True


def test_cosmos3_tokenize_prompt_uses_checkpoint_system_prompt_default():
    pipeline = get_dummy_pipeline(default_use_system_prompt=False)

    pipeline.tokenize_prompt("A prompt", num_frames=1, add_resolution_template=False)

    assert all(conversation[0]["role"] == "user" for conversation in pipeline.text_tokenizer.conversations)
