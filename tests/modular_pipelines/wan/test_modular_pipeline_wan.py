# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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
import torch

from diffusers.modular_pipelines import WanBlocks, WanModularPipeline
from diffusers.modular_pipelines.modular_pipeline import BlockState, PipelineState
from diffusers.modular_pipelines.wan.before_denoise import WanTextInputStep
from diffusers.modular_pipelines.wan.denoise import Wan22LoopDenoiser, WanLoopDenoiser

from ..test_modular_pipelines_common import ModularPipelineTesterMixin


class _FakeGuider:
    def set_state(self, step, num_inference_steps, timestep):
        self.step = step
        self.num_inference_steps = num_inference_steps
        self.timestep = timestep

    def prepare_inputs_from_block_state(self, block_state, guider_input_fields):
        batch = {}
        for model_input_name, block_state_input_names in guider_input_fields.items():
            if isinstance(block_state_input_names, tuple):
                block_state_input_name = block_state_input_names[0]
            else:
                block_state_input_name = block_state_input_names
            batch[model_input_name] = getattr(block_state, block_state_input_name)
        return [BlockState(**batch)]

    def prepare_models(self, model):
        pass

    def cleanup_models(self, model):
        pass

    def __call__(self, guider_state):
        return [guider_state[0].noise_pred]


class _FakeTransformer:
    def __init__(self, dtype):
        self.dtype = dtype
        self.calls = []

    def __call__(self, hidden_states, timestep, attention_kwargs, return_dict, **kwargs):
        self.calls.append(
            {
                "hidden_states": hidden_states,
                "timestep": timestep,
                "attention_kwargs": attention_kwargs,
                "return_dict": return_dict,
                **kwargs,
            }
        )
        return (torch.zeros_like(hidden_states),)


class TestWanModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = WanModularPipeline
    pipeline_blocks_class = WanBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-wan-modular-pipe"

    params = frozenset(["prompt", "height", "width", "num_frames"])
    batch_params = frozenset(["prompt"])
    optional_params = frozenset(["num_inference_steps", "num_videos_per_prompt", "latents"])
    output_name = "videos"

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 16,
            "width": 16,
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs

    @pytest.mark.skip(reason="num_videos_per_prompt")
    def test_num_images_per_prompt(self):
        pass

    def test_vae_scale_factors_use_config_values(self):
        pipe = WanModularPipeline.__new__(WanModularPipeline)

        assert pipe.vae_scale_factor_spatial == 8
        assert pipe.vae_scale_factor_temporal == 4

        pipe.vae = SimpleNamespace(
            config=SimpleNamespace(scale_factor_spatial=16, scale_factor_temporal=2),
            temperal_downsample=[True, True, False],
        )

        assert pipe.vae_scale_factor_spatial == 16
        assert pipe.vae_scale_factor_temporal == 2
        assert pipe.default_height == 960
        assert pipe.default_width == 1664
        assert pipe.default_num_frames == 41

    def test_text_input_step_uses_transformer_dtype_and_repeat_interleave(self):
        step = WanTextInputStep()
        components = SimpleNamespace(transformer=SimpleNamespace(dtype=torch.bfloat16))
        prompt_embeds = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        negative_prompt_embeds = -prompt_embeds
        state = PipelineState()
        state.set("num_videos_per_prompt", 2)
        state.set("prompt_embeds", prompt_embeds)
        state.set("negative_prompt_embeds", negative_prompt_embeds)

        _, state = step(components, state)

        assert state.batch_size == 2
        assert state.dtype == torch.bfloat16
        torch.testing.assert_close(state.prompt_embeds, prompt_embeds.repeat_interleave(2, dim=0).to(torch.bfloat16))
        torch.testing.assert_close(
            state.negative_prompt_embeds, negative_prompt_embeds.repeat_interleave(2, dim=0).to(torch.bfloat16)
        )

    def test_loop_denoiser_preserves_timestep_dtype(self):
        transformer = _FakeTransformer(dtype=torch.bfloat16)
        components = SimpleNamespace(transformer=transformer, guider=_FakeGuider())
        block_state = BlockState(
            attention_kwargs=None,
            dtype=torch.bfloat16,
            latent_model_input=torch.ones(2, 4, dtype=torch.float32),
            num_inference_steps=1,
            prompt_embeds=torch.ones(2, 3, dtype=torch.float32),
            negative_prompt_embeds=torch.zeros(2, 3, dtype=torch.float32),
        )
        timestep = torch.tensor(999.1234, dtype=torch.float32)

        WanLoopDenoiser()(components, block_state, i=0, t=timestep)

        call = transformer.calls[0]
        assert call["hidden_states"].dtype == torch.bfloat16
        assert call["encoder_hidden_states"].dtype == torch.bfloat16
        assert call["timestep"].dtype == torch.float32
        torch.testing.assert_close(call["timestep"], timestep.expand(2))

    def test_wan22_loop_denoiser_uses_selected_transformer_dtype_and_preserves_timestep_dtype(self):
        high_noise_transformer = _FakeTransformer(dtype=torch.bfloat16)
        low_noise_transformer = _FakeTransformer(dtype=torch.float16)
        components = SimpleNamespace(
            config=SimpleNamespace(boundary_ratio=0.875),
            num_train_timesteps=1000,
            transformer=high_noise_transformer,
            transformer_2=low_noise_transformer,
            guider=_FakeGuider(),
            guider_2=_FakeGuider(),
        )
        block_state = BlockState(
            attention_kwargs=None,
            dtype=torch.bfloat16,
            latent_model_input=torch.ones(2, 4, dtype=torch.float32),
            num_inference_steps=1,
            prompt_embeds=torch.ones(2, 3, dtype=torch.float32),
            negative_prompt_embeds=torch.zeros(2, 3, dtype=torch.float32),
        )
        timestep = torch.tensor(10.25, dtype=torch.float32)

        Wan22LoopDenoiser()(components, block_state, i=0, t=timestep)

        assert len(high_noise_transformer.calls) == 0
        call = low_noise_transformer.calls[0]
        assert call["hidden_states"].dtype == torch.float16
        assert call["encoder_hidden_states"].dtype == torch.float16
        assert call["timestep"].dtype == torch.float32
        torch.testing.assert_close(call["timestep"], timestep.expand(2))
