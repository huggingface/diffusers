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


from diffusers.modular_pipelines import InputParam, ModularPipelineBlocks, SequentialPipelineBlocks

from .test_conditional_pipeline_blocks import ImageToImageBlock


class NoDefaultStrengthBlock(ModularPipelineBlocks):
    @property
    def inputs(self):
        return [InputParam(name="strength", type_hint=float)]

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        self.set_block_state(state, block_state)
        return components, state


class NoDefaultThenImg2Img(SequentialPipelineBlocks):
    block_classes = [NoDefaultStrengthBlock, ImageToImageBlock]
    block_names = ["no_default", "img2img"]


class TestSequenceDefaults:
    def test_first_reader_fills_and_records_the_default(self):
        # in a sequence the input is one shared value: the first block that reads `strength` fills in its own default
        # (None) and records it, so the img2img block reads None instead of swapping in its 0.3
        pipe = NoDefaultThenImg2Img().init_pipeline()
        state = pipe(prompt="p", image="i")
        assert state.get("workflow") == "img2img"
        assert "strength" in state.values
        assert state.get("strength") is None

        # a passed value reaches every block
        state = pipe(prompt="p", image="i", strength=0.7)
        assert state.get("strength") == 0.7
