# coding=utf-8
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

from types import SimpleNamespace

import pytest

from diffusers.modular_pipelines import Cosmos3DistilledBlocks
from diffusers.modular_pipelines.cosmos.before_denoise import Cosmos3DistilledSetTimestepsStep
from diffusers.modular_pipelines.cosmos.encoders import Cosmos3DistilledTextEncoderStep
from diffusers.modular_pipelines.modular_pipeline import PipelineState


DISTILLED_WORKFLOW = [
    "prepare_text_segments",
    "prepare_vision_latents",
    "pack_vision_sequence",
    "prepare_vision_denoiser_inputs",
    "set_timesteps",
    "denoise",
]


def _fake_distilled_components(t_list=(1.0, 0.75, 0.5, 0.25)):
    scheduler = SimpleNamespace(config={"fixed_step_sampler_config": {"t_list": list(t_list)}})
    return SimpleNamespace(_execution_device="cpu", scheduler=scheduler)


def test_cosmos3_distilled_blocks_workflow_ordering():
    blocks = Cosmos3DistilledBlocks()
    assert blocks.block_names == ["text_encoder", "vae_encoder", "denoise", "decode"]
    assert type(blocks.sub_blocks["text_encoder"]).__name__ == "Cosmos3DistilledTextEncoderStep"

    denoise = blocks.sub_blocks["denoise"]
    assert denoise.block_names == DISTILLED_WORKFLOW
    assert type(denoise.sub_blocks["set_timesteps"]).__name__ == "Cosmos3DistilledSetTimestepsStep"

    inner_loop = denoise.sub_blocks["denoise"]
    assert type(inner_loop).__name__ == "Cosmos3DistilledVisionDenoiseStep"
    assert inner_loop.block_names == ["prepare_vision", "denoiser", "update_vision"]
    assert type(inner_loop.sub_blocks["update_vision"]).__name__ == "Cosmos3DistilledVisionLoopSchedulerStep"


def test_cosmos3_distilled_supported_workflows():
    blocks = Cosmos3DistilledBlocks()
    assert set(blocks._workflow_map) == {"text2image", "text2video", "image2video", "video2video"}


def test_cosmos3_distilled_vae_encoder_select_block():
    vae_encoder = Cosmos3DistilledBlocks().sub_blocks["vae_encoder"]
    assert vae_encoder.select_block(image=None, video=None) is None
    assert vae_encoder.select_block(image=object(), video=None) == "image_conditioning"
    assert vae_encoder.select_block(image=None, video=object()) == "video_conditioning"
    with pytest.raises(ValueError, match="either image or video"):
        vae_encoder.select_block(image=object(), video=object())


def test_cosmos3_distilled_set_timesteps_declares_is_distilled_config():
    configs = {spec.name: spec.default for spec in Cosmos3DistilledSetTimestepsStep().expected_configs}
    assert configs == {"is_distilled": True}


def test_cosmos3_distilled_rejects_negative_prompt():
    Cosmos3DistilledTextEncoderStep._check_inputs(SimpleNamespace(prompt="a robot", negative_prompt=None))
    with pytest.raises(ValueError, match="`negative_prompt` is not supported"):
        Cosmos3DistilledTextEncoderStep._check_inputs(SimpleNamespace(prompt="a robot", negative_prompt="blurry"))


def test_cosmos3_distilled_set_timesteps_rejects_step_count_override():
    step = Cosmos3DistilledSetTimestepsStep()
    state = PipelineState()
    state.set("num_inference_steps", 10)
    with pytest.raises(ValueError, match="must be 4 or left unset"):
        step(_fake_distilled_components(), state)


def test_cosmos3_distilled_set_timesteps_rejects_guidance_override():
    step = Cosmos3DistilledSetTimestepsStep()
    state = PipelineState()
    state.set("guidance_scale", 3.0)
    with pytest.raises(ValueError, match="`guidance_scale` must be 1.0"):
        step(_fake_distilled_components(), state)


def test_cosmos3_distilled_set_timesteps_requires_fixed_step_config():
    step = Cosmos3DistilledSetTimestepsStep()
    components = SimpleNamespace(_execution_device="cpu", scheduler=SimpleNamespace(config={}))
    with pytest.raises(ValueError, match="fixed_step_sampler_config"):
        step(components, PipelineState())
