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


from diffusers.modular_pipelines import (
    AutoPipelineBlocks,
    ConditionalPipelineBlocks,
    InputParam,
    ModularPipelineBlocks,
)


class TextToImageBlock(ModularPipelineBlocks):
    model_name = "text2img"

    @property
    def inputs(self):
        return [InputParam(name="prompt")]

    @property
    def intermediate_outputs(self):
        return []

    @property
    def description(self):
        return "text-to-image workflow"

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        block_state.workflow = "text2img"
        self.set_block_state(state, block_state)
        return components, state


class ImageToImageBlock(ModularPipelineBlocks):
    model_name = "img2img"

    @property
    def inputs(self):
        return [InputParam(name="prompt"), InputParam(name="image")]

    @property
    def intermediate_outputs(self):
        return []

    @property
    def description(self):
        return "image-to-image workflow"

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        block_state.workflow = "img2img"
        self.set_block_state(state, block_state)
        return components, state


class InpaintBlock(ModularPipelineBlocks):
    model_name = "inpaint"

    @property
    def inputs(self):
        return [InputParam(name="prompt"), InputParam(name="image"), InputParam(name="mask")]

    @property
    def intermediate_outputs(self):
        return []

    @property
    def description(self):
        return "inpaint workflow"

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        block_state.workflow = "inpaint"
        self.set_block_state(state, block_state)
        return components, state


class ConditionalImageBlocks(ConditionalPipelineBlocks):
    block_classes = [InpaintBlock, ImageToImageBlock, TextToImageBlock]
    block_names = ["inpaint", "img2img", "text2img"]
    block_trigger_inputs = ["mask", "image"]
    default_block_name = "text2img"

    @property
    def description(self):
        return "Conditional image blocks for testing"

    def select_block(self, mask=None, image=None) -> str | None:
        if mask is not None:
            return "inpaint"
        if image is not None:
            return "img2img"
        return None  # falls back to default_block_name


class OptionalConditionalBlocks(ConditionalPipelineBlocks):
    block_classes = [InpaintBlock, ImageToImageBlock]
    block_names = ["inpaint", "img2img"]
    block_trigger_inputs = ["mask", "image"]
    default_block_name = None  # no default; block can be skipped

    @property
    def description(self):
        return "Optional conditional blocks (skippable)"

    def select_block(self, mask=None, image=None) -> str | None:
        if mask is not None:
            return "inpaint"
        if image is not None:
            return "img2img"
        return None


class AutoImageBlocks(AutoPipelineBlocks):
    block_classes = [InpaintBlock, ImageToImageBlock, TextToImageBlock]
    block_names = ["inpaint", "img2img", "text2img"]
    block_trigger_inputs = ["mask", "image", None]

    @property
    def description(self):
        return "Auto image blocks for testing"


class TestConditionalPipelineBlocksSelectBlock:
    def test_select_block_with_mask(self):
        blocks = ConditionalImageBlocks()
        assert blocks.select_block(mask="something") == "inpaint"

    def test_select_block_with_image(self):
        blocks = ConditionalImageBlocks()
        assert blocks.select_block(image="something") == "img2img"

    def test_select_block_with_mask_and_image(self):
        blocks = ConditionalImageBlocks()
        assert blocks.select_block(mask="m", image="i") == "inpaint"

    def test_select_block_no_triggers_returns_none(self):
        blocks = ConditionalImageBlocks()
        assert blocks.select_block() is None

    def test_select_block_explicit_none_values(self):
        blocks = ConditionalImageBlocks()
        assert blocks.select_block(mask=None, image=None) is None


class TestConditionalPipelineBlocksWorkflowSelection:
    def test_default_workflow_when_no_triggers(self):
        blocks = ConditionalImageBlocks()
        execution = blocks.get_execution_blocks()
        assert execution is not None
        assert isinstance(execution, TextToImageBlock)

    def test_mask_trigger_selects_inpaint(self):
        blocks = ConditionalImageBlocks()
        execution = blocks.get_execution_blocks(mask=True)
        assert isinstance(execution, InpaintBlock)

    def test_image_trigger_selects_img2img(self):
        blocks = ConditionalImageBlocks()
        execution = blocks.get_execution_blocks(image=True)
        assert isinstance(execution, ImageToImageBlock)

    def test_mask_and_image_selects_inpaint(self):
        blocks = ConditionalImageBlocks()
        execution = blocks.get_execution_blocks(mask=True, image=True)
        assert isinstance(execution, InpaintBlock)

    def test_skippable_block_returns_none(self):
        blocks = OptionalConditionalBlocks()
        execution = blocks.get_execution_blocks()
        assert execution is None

    def test_skippable_block_still_selects_when_triggered(self):
        blocks = OptionalConditionalBlocks()
        execution = blocks.get_execution_blocks(image=True)
        assert isinstance(execution, ImageToImageBlock)


class TestAutoPipelineBlocksSelectBlock:
    def test_auto_select_mask(self):
        blocks = AutoImageBlocks()
        assert blocks.select_block(mask="m") == "inpaint"

    def test_auto_select_image(self):
        blocks = AutoImageBlocks()
        assert blocks.select_block(image="i") == "img2img"

    def test_auto_select_default(self):
        blocks = AutoImageBlocks()
        # No trigger -> returns None -> falls back to default (text2img)
        assert blocks.select_block() is None

    def test_auto_select_priority_order(self):
        blocks = AutoImageBlocks()
        assert blocks.select_block(mask="m", image="i") == "inpaint"


class TestAutoPipelineBlocksWorkflowSelection:
    def test_auto_default_workflow(self):
        blocks = AutoImageBlocks()
        execution = blocks.get_execution_blocks()
        assert isinstance(execution, TextToImageBlock)

    def test_auto_mask_workflow(self):
        blocks = AutoImageBlocks()
        execution = blocks.get_execution_blocks(mask=True)
        assert isinstance(execution, InpaintBlock)

    def test_auto_image_workflow(self):
        blocks = AutoImageBlocks()
        execution = blocks.get_execution_blocks(image=True)
        assert isinstance(execution, ImageToImageBlock)


class TestConditionalPipelineBlocksStructure:
    def test_block_names_accessible(self):
        blocks = ConditionalImageBlocks()
        sub = dict(blocks.sub_blocks)
        assert set(sub.keys()) == {"inpaint", "img2img", "text2img"}

    def test_sub_block_types(self):
        blocks = ConditionalImageBlocks()
        sub = dict(blocks.sub_blocks)
        assert isinstance(sub["inpaint"], InpaintBlock)
        assert isinstance(sub["img2img"], ImageToImageBlock)
        assert isinstance(sub["text2img"], TextToImageBlock)

    def test_description(self):
        blocks = ConditionalImageBlocks()
        assert "Conditional" in blocks.description
