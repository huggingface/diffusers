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

import pytest
import torch

from diffusers import ModularPipeline, ModularPipelineBlocks
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from diffusers.modular_pipelines.wan.denoise import WanDenoiseLoopWrapper
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class CallbackSetup(ModularPipelineBlocks):
    @property
    def expected_components(self):
        return [
            ComponentSpec(
                "scheduler", FlowMatchEulerDiscreteScheduler, config={}, default_creation_method="from_config"
            )
        ]

    @property
    def inputs(self):
        return [
            InputParam("latents", required=True),
            InputParam("prompt_embeds", required=True),
            InputParam("num_inference_steps", default=3),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("timesteps"), OutputParam("prompt_embeds", kwargs_type="denoiser_input_fields")]

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        components.scheduler.set_timesteps(block_state.num_inference_steps)
        block_state.timesteps = components.scheduler.timesteps
        self.set_block_state(state, block_state)
        return components, state


class CallbackSchedulerStep(ModularPipelineBlocks):
    @property
    def inputs(self):
        return [
            InputParam("latents", required=True),
            InputParam("prompt_embeds", required=True),
            InputParam.template("denoiser_input_fields"),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("latents")]

    def __call__(self, components, block_state, i, t):
        assert block_state.prompt_embeds is block_state.denoiser_input_fields["prompt_embeds"]
        block_state.latents = components.scheduler.step(
            block_state.prompt_embeds, t, block_state.latents, return_dict=False
        )[0]
        return components, block_state


class CallbackLoop(WanDenoiseLoopWrapper):
    block_classes = [CallbackSchedulerStep]
    block_names = ["scheduler"]

    @property
    def loop_expected_components(self):
        return CallbackSetup().expected_components


class ZeroLatentsCallback(PipelineCallback):
    @property
    def tensor_inputs(self):
        return ["latents"]

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs):
        return {"latents": torch.zeros_like(callback_kwargs["latents"])}


class TestModularCallbacks:
    def get_pipeline(self, loops=1):
        blocks = {"setup": CallbackSetup()}
        for i in range(loops):
            if i:
                blocks[f"setup_{i}"] = CallbackSetup()
            blocks[f"denoise_{i}"] = CallbackLoop()
        return ModularPipeline(blocks=SequentialPipelineBlocks.from_blocks_dict(blocks))

    def run(self, pipe, **kwargs):
        return pipe(latents=torch.zeros(1, 2), prompt_embeds=torch.ones(1, 2), output="latents", **kwargs)

    def test_read_only_callback_and_global_steps(self):
        pipe = self.get_pipeline(loops=2)
        baseline = self.run(pipe)
        steps = []

        def callback(pipeline, step, timestep, tensors):
            steps.append(step)
            return tensors

        output = self.run(pipe, callback_on_step_end=callback)
        torch.testing.assert_close(output, baseline, rtol=0, atol=0)
        assert steps == list(range(6))
        assert not pipe.interrupt
        assert pipe._callback_on_step_end is None

    def test_latents_and_conditioning_updates(self):
        pipe = self.get_pipeline()
        observed = []

        def callback(pipeline, step, timestep, tensors):
            observed.append(tensors["latents"].clone())
            return {
                "latents": torch.zeros_like(tensors["latents"]),
                "prompt_embeds": torch.zeros_like(tensors["latents"]),
            }

        output = self.run(pipe, callback_on_step_end=callback)
        assert observed[0].abs().sum() > 0
        assert observed[1].count_nonzero() == 0
        assert output.count_nonzero() == 0

    def test_callback_objects(self):
        pipe = self.get_pipeline()
        for callback in [
            ZeroLatentsCallback(),
            MultiPipelineCallbacks([ZeroLatentsCallback(), ZeroLatentsCallback()]),
        ]:
            output = self.run(pipe, callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["invalid"])
            assert output.count_nonzero() == 0

    def test_interrupt_and_next_call(self):
        pipe = self.get_pipeline(loops=2)
        steps = []

        def callback(pipeline, step, timestep, tensors):
            steps.append(step)
            pipeline.interrupt = True
            return tensors

        output = self.run(pipe, callback_on_step_end=callback)
        assert steps == [0]
        assert torch.isfinite(output).all()
        assert pipe.interrupt
        assert not torch.equal(self.run(pipe), output)
        assert not pipe.interrupt

    @pytest.mark.parametrize(
        "callback,inputs,error",
        [
            (lambda *args: None, None, TypeError),
            (lambda *args: {"invalid": 1}, None, ValueError),
            (lambda *args: {}, ["invalid"], ValueError),
            (lambda *args: {}, ["negative_prompt_embeds"], ValueError),
            (0, None, TypeError),
            (None, ["invalid"], ValueError),
            (lambda *args: {}, "latents", TypeError),
        ],
    )
    def test_invalid_callbacks_and_cleanup(self, callback, inputs, error):
        pipe = self.get_pipeline()
        pipe.interrupt = True
        with pytest.raises(error):
            self.run(pipe, callback_on_step_end=callback, callback_on_step_end_tensor_inputs=inputs)
        assert not pipe.interrupt
        assert getattr(pipe, "_callback_on_step_end", None) is None
        assert torch.isfinite(self.run(pipe)).all()
