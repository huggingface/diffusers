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


from diffusers.modular_pipelines import (
    AutoPipelineBlocks,
    ConditionalPipelineBlocks,
    InputParam,
    ModularPipelineBlocks,
)
from diffusers.modular_pipelines.modular_pipeline_utils import combine_inputs


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
        return [
            InputParam(name="prompt"),
            InputParam(name="image"),
            InputParam(name="strength", type_hint=float, default=0.3),
        ]

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
        return [
            InputParam(name="prompt"),
            InputParam(name="image"),
            InputParam(name="mask"),
            InputParam(name="strength", type_hint=float, default=0.9999),
        ]

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


class NestedImageBlocks(ConditionalPipelineBlocks):
    block_classes = [InpaintBlock, AutoImageBlocks]
    block_names = ["refine", "image"]
    block_trigger_inputs = ["mask"]
    default_block_name = "image"

    @property
    def description(self):
        return "Nested conditional blocks: refine when `mask` is provided, auto image blocks otherwise"

    def select_block(self, mask=None) -> str | None:
        if mask is not None:
            return "refine"
        return None


class TestConditionalBlocksBranchDefaults:
    def test_conflicting_defaults_merge_to_none(self):
        merged = {p.name: p for p in AutoImageBlocks().inputs}["strength"]
        assert merged.default is None
        assert merged.defaults_by_block == {"inpaint": 0.9999, "img2img": 0.3}

    def test_agreeing_defaults_stay_untouched(self):
        combined = combine_inputs(
            ("a", [InputParam(name="x", default=5)]),
            ("b", [InputParam(name="x", default=5)]),
        )
        assert combined[0].default == 5
        assert combined[0].defaults_by_block is None

    def test_none_default_counts_as_disagreement(self):
        # a None default is a sentinel ("user didn't pass this"), it must not be overridden by a sibling's default
        combined = combine_inputs(
            ("a", [InputParam(name="x", default=None)]),
            ("b", [InputParam(name="x", default=5)]),
        )
        assert combined[0].default is None
        assert combined[0].defaults_by_block == {"a": None, "b": 5}

    def test_branch_resolves_own_default(self):
        pipe = AutoImageBlocks().init_pipeline()
        state = pipe(prompt="p", image="i")
        assert state.get("strength") == 0.3

        state = pipe(prompt="p", image="i", mask="m")
        assert state.get("strength") == 0.9999

    def test_explicit_value_overrides_branch_default(self):
        pipe = AutoImageBlocks().init_pipeline()
        state = pipe(prompt="p", image="i", strength=0.7)
        assert state.get("strength") == 0.7

    def test_standalone_branch_keeps_default(self):
        pipe = ImageToImageBlock().init_pipeline()
        state = pipe(prompt="p", image="i")
        assert state.get("strength") == 0.3

    def test_doc_renders_per_block_defaults(self):
        doc = " ".join(AutoImageBlocks().doc.split())
        assert "strength (`float`, *optional*, defaults to 0.9999 or 0.3, depending on the workflow):" in doc

    def test_nested_defaults_prefixed_with_sub_block_name(self):
        merged = {p.name: p for p in NestedImageBlocks().inputs}["strength"]
        assert merged.default is None
        assert merged.defaults_by_block == {
            "refine": 0.9999,
            "image.inpaint": 0.9999,
            "image.img2img": 0.3,
        }
