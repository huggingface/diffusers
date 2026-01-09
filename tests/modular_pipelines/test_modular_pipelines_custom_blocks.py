# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import os
import tempfile
from collections import deque
from typing import List

import numpy as np
import torch

from diffusers import FluxTransformer2DModel
from diffusers.modular_pipelines import (
    ComponentSpec,
    InputParam,
    ModularPipelineBlocks,
    OutputParam,
    PipelineState,
    WanModularPipeline,
)

from ..testing_utils import nightly, require_torch, slow


class DummyCustomBlockSimple(ModularPipelineBlocks):
    def __init__(self, use_dummy_model_component=False):
        self.use_dummy_model_component = use_dummy_model_component
        super().__init__()

    @property
    def expected_components(self):
        if self.use_dummy_model_component:
            return [ComponentSpec("transformer", FluxTransformer2DModel)]
        else:
            return []

    @property
    def inputs(self) -> List[InputParam]:
        return [InputParam("prompt", type_hint=str, required=True, description="Prompt to use")]

    @property
    def intermediate_inputs(self) -> List[InputParam]:
        return []

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "output_prompt",
                type_hint=str,
                description="Modified prompt",
            )
        ]

    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        old_prompt = block_state.prompt
        block_state.output_prompt = "Modular diffusers + " + old_prompt
        self.set_block_state(state, block_state)

        return components, state


CODE_STR = """
from diffusers.modular_pipelines import (
    ComponentSpec,
    InputParam,
    ModularPipelineBlocks,
    OutputParam,
    PipelineState,
    WanModularPipeline,
)
from typing import List

class DummyCustomBlockSimple(ModularPipelineBlocks):
    def __init__(self, use_dummy_model_component=False):
        self.use_dummy_model_component = use_dummy_model_component
        super().__init__()

    @property
    def expected_components(self):
        if self.use_dummy_model_component:
            return [ComponentSpec("transformer", FluxTransformer2DModel)]
        else:
            return []

    @property
    def inputs(self) -> List[InputParam]:
        return [InputParam("prompt", type_hint=str, required=True, description="Prompt to use")]

    @property
    def intermediate_inputs(self) -> List[InputParam]:
        return []

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "output_prompt",
                type_hint=str,
                description="Modified prompt",
            )
        ]

    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        old_prompt = block_state.prompt
        block_state.output_prompt = "Modular diffusers + " + old_prompt
        self.set_block_state(state, block_state)

        return components, state
"""


class TestModularCustomBlocks:
    def _test_block_properties(self, block):
        assert not block.expected_components
        assert not block.intermediate_inputs

        actual_inputs = [inp.name for inp in block.inputs]
        actual_intermediate_outputs = [out.name for out in block.intermediate_outputs]
        assert actual_inputs == ["prompt"]
        assert actual_intermediate_outputs == ["output_prompt"]

    def test_custom_block_properties(self):
        custom_block = DummyCustomBlockSimple()
        self._test_block_properties(custom_block)

    def test_custom_block_output(self):
        custom_block = DummyCustomBlockSimple()
        pipe = custom_block.init_pipeline()
        prompt = "Diffusers is nice"
        output = pipe(prompt=prompt)

        actual_inputs = [inp.name for inp in custom_block.inputs]
        actual_intermediate_outputs = [out.name for out in custom_block.intermediate_outputs]
        assert sorted(output.values) == sorted(actual_inputs + actual_intermediate_outputs)

        output_prompt = output.values["output_prompt"]
        assert output_prompt.startswith("Modular diffusers + ")

    def test_custom_block_saving_loading(self):
        custom_block = DummyCustomBlockSimple()

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_block.save_pretrained(tmpdir)
            assert any("modular_config.json" in k for k in os.listdir(tmpdir))

            with open(os.path.join(tmpdir, "modular_config.json"), "r") as f:
                config = json.load(f)
            auto_map = config["auto_map"]
            assert auto_map == {"ModularPipelineBlocks": "test_modular_pipelines_custom_blocks.DummyCustomBlockSimple"}

            # For now, the Python script that implements the custom block has to be manually pushed to the Hub.
            # This is why, we have to separately save the Python script here.
            code_path = os.path.join(tmpdir, "test_modular_pipelines_custom_blocks.py")
            with open(code_path, "w") as f:
                f.write(CODE_STR)

            loaded_custom_block = ModularPipelineBlocks.from_pretrained(tmpdir, trust_remote_code=True)

        pipe = loaded_custom_block.init_pipeline()
        prompt = "Diffusers is nice"
        output = pipe(prompt=prompt)

        actual_inputs = [inp.name for inp in loaded_custom_block.inputs]
        actual_intermediate_outputs = [out.name for out in loaded_custom_block.intermediate_outputs]
        assert sorted(output.values) == sorted(actual_inputs + actual_intermediate_outputs)

        output_prompt = output.values["output_prompt"]
        assert output_prompt.startswith("Modular diffusers + ")

    def test_custom_block_supported_components(self):
        custom_block = DummyCustomBlockSimple(use_dummy_model_component=True)
        pipe = custom_block.init_pipeline("hf-internal-testing/tiny-flux-kontext-pipe")
        pipe.load_components()

        assert len(pipe.components) == 1
        assert pipe.component_names[0] == "transformer"

    def test_custom_block_loads_from_hub(self):
        repo_id = "hf-internal-testing/tiny-modular-diffusers-block"
        block = ModularPipelineBlocks.from_pretrained(repo_id, trust_remote_code=True)
        self._test_block_properties(block)

        pipe = block.init_pipeline()

        prompt = "Diffusers is nice"
        output = pipe(prompt=prompt)
        output_prompt = output.values["output_prompt"]
        assert output_prompt.startswith("Modular diffusers + ")


@slow
@nightly
@require_torch
class TestKreaCustomBlocksIntegration:
    repo_id = "krea/krea-realtime-video"

    def test_loading_from_hub(self):
        blocks = ModularPipelineBlocks.from_pretrained(self.repo_id, trust_remote_code=True)
        block_names = sorted(blocks.sub_blocks)

        assert block_names == sorted(["text_encoder", "before_denoise", "denoise", "decode"])

        pipe = WanModularPipeline(blocks, self.repo_id)
        pipe.load_components(
            trust_remote_code=True,
            device_map="cuda",
            torch_dtype={"default": torch.bfloat16, "vae": torch.float16},
        )
        assert len(pipe.components) == 7
        assert sorted(pipe.components) == sorted(
            ["text_encoder", "tokenizer", "guider", "scheduler", "vae", "transformer", "video_processor"]
        )

    def test_forward(self):
        blocks = ModularPipelineBlocks.from_pretrained(self.repo_id, trust_remote_code=True)
        pipe = WanModularPipeline(blocks, self.repo_id)
        pipe.load_components(
            trust_remote_code=True,
            device_map="cuda",
            torch_dtype={"default": torch.bfloat16, "vae": torch.float16},
        )

        num_frames_per_block = 2
        num_blocks = 2

        state = PipelineState()
        state.set("frame_cache_context", deque(maxlen=pipe.config.frame_cache_len))

        prompt = ["a cat sitting on a boat"]

        for block in pipe.transformer.blocks:
            block.self_attn.fuse_projections()

        for block_idx in range(num_blocks):
            state = pipe(
                state,
                prompt=prompt,
                num_inference_steps=2,
                num_blocks=num_blocks,
                num_frames_per_block=num_frames_per_block,
                block_idx=block_idx,
                generator=torch.manual_seed(42),
            )
            current_frames = np.array(state.values["videos"][0])
            current_frames_flat = current_frames.flatten()
            actual_slices = np.concatenate([current_frames_flat[:4], current_frames_flat[-4:]]).tolist()

            if block_idx == 0:
                assert current_frames.shape == (5, 480, 832, 3)
                expected_slices = np.array([211, 229, 238, 208, 195, 180, 188, 193])
            else:
                assert current_frames.shape == (8, 480, 832, 3)
                expected_slices = np.array([179, 203, 214, 176, 194, 181, 187, 191])

            assert np.allclose(actual_slices, expected_slices)
