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

import contextlib

import pytest

from diffusers.modular_pipelines import (
    AutoPipelineBlocks,
    ComponentSpec,
    ConditionalPipelineBlocks,
    ConfigSpec,
    InputParam,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    OutputParam,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.modular_pipeline_utils import REQUIRED
from diffusers.utils import logging
from diffusers.utils.testing_utils import CaptureLogger


logger = logging.get_logger("diffusers.modular_pipelines.modular_pipeline")


@contextlib.contextmanager
def quiet():
    """The body must log nothing."""
    with CaptureLogger(logger) as logs:
        yield
    assert logs.out == "", logs.out


class EncodeImageBlock(ModularPipelineBlocks):
    model_name = "test"

    @property
    def inputs(self):
        return [InputParam("image", required=True)]

    @property
    def intermediate_outputs(self):
        return [OutputParam("image_latents")]

    def __call__(self, components, state):
        return components, state


class PrepareLatentsBlock(ModularPipelineBlocks):
    """Takes `latents` as an optional user input and always produces it."""

    model_name = "test"

    @property
    def inputs(self):
        return [InputParam("latents"), InputParam("batch_size", required=True)]

    @property
    def intermediate_outputs(self):
        return [OutputParam("latents")]

    def __call__(self, components, state):
        return components, state


class DenoiseBlock(ModularPipelineBlocks):
    """Requires `latents`, which an earlier block normally produces."""

    model_name = "test"

    @property
    def inputs(self):
        return [InputParam("latents", required=True), InputParam("prompt_embeds", required=True)]

    @property
    def intermediate_outputs(self):
        return [OutputParam("latents")]

    def __call__(self, components, state):
        return components, state


class SkippableEncodeBlocks(ConditionalPipelineBlocks):
    block_classes = [EncodeImageBlock]
    block_names = ["encode"]
    block_trigger_inputs = ["image"]
    default_block_name = None

    def select_block(self, image=None):
        return "encode" if image is not None else None


class NeedsImageLatentsBlock(ModularPipelineBlocks):
    model_name = "test"

    @property
    def inputs(self):
        return [InputParam("image_latents", required=True)]

    def __call__(self, components, state):
        return components, state


def strength_auto_blocks(img2img_default, inpaint_default):
    """
    An auto block whose two branches declare `strength` with their own default: `None` for no default, `REQUIRED` for
    a branch that requires it, `...` for a branch that does not declare it.
    """

    def strength(default):
        if default is ...:
            return []
        if default is REQUIRED:
            return [InputParam("strength", required=True)]
        return [InputParam("strength", default=default)]

    class Img2ImgBlock(ModularPipelineBlocks):
        model_name = "test"

        @property
        def inputs(self):
            return [InputParam("image", required=True), *strength(img2img_default)]

        def __call__(self, components, state):
            return components, state

    class InpaintBlock(ModularPipelineBlocks):
        model_name = "test"

        @property
        def inputs(self):
            return [InputParam("mask", required=True), *strength(inpaint_default)]

        def __call__(self, components, state):
            return components, state

    class Auto(AutoPipelineBlocks):
        block_classes = [InpaintBlock, Img2ImgBlock]
        block_names = ["inpaint", "img2img"]
        block_trigger_inputs = ["mask", "image"]

    return Auto


class RequiresStrengthBlock(ModularPipelineBlocks):
    model_name = "test"

    @property
    def inputs(self):
        return [InputParam("strength", required=True)]

    def __call__(self, components, state):
        return components, state


def required_flags(blocks, *names):
    return {inp.name: inp.required for inp in blocks.inputs if inp.name in names}


class TestSequentialPipelineBlocksRequiredInputs:
    def test_input_produced_by_earlier_block_is_not_required(self):
        blocks = SequentialPipelineBlocks.from_blocks_dict(
            {"prepare_latents": PrepareLatentsBlock(), "denoise": DenoiseBlock()}
        )

        assert set(blocks.required_inputs) == {"batch_size", "prompt_embeds"}
        assert required_flags(blocks, "latents", "batch_size", "prompt_embeds") == {
            "latents": False,
            "batch_size": True,
            "prompt_embeds": True,
        }

    def test_input_consumed_before_it_is_produced_stays_required(self):
        blocks = SequentialPipelineBlocks.from_blocks_dict(
            {"denoise": DenoiseBlock(), "prepare_latents": PrepareLatentsBlock()}
        )

        assert set(blocks.required_inputs) == {"latents", "batch_size", "prompt_embeds"}

    def test_later_block_requiring_an_optional_input_makes_it_required(self):
        class OptionalPromptBlock(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [InputParam("prompt")]

            def __call__(self, components, state):
                return components, state

        class RequiredPromptBlock(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [InputParam("prompt", required=True)]

            def __call__(self, components, state):
                return components, state

        blocks = SequentialPipelineBlocks.from_blocks_dict(
            {"optional": OptionalPromptBlock(), "required": RequiredPromptBlock()}
        )

        assert blocks.required_inputs == ["prompt"]
        assert required_flags(blocks, "prompt") == {"prompt": True}

    def test_default_from_earlier_block_satisfies_later_requirement(self):
        class DefaultStepsBlock(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [InputParam("num_inference_steps", default=50)]

            def __call__(self, components, state):
                return components, state

        class RequiredStepsBlock(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [InputParam("num_inference_steps", required=True)]

            def __call__(self, components, state):
                return components, state

        blocks = SequentialPipelineBlocks.from_blocks_dict(
            {"default": DefaultStepsBlock(), "required": RequiredStepsBlock()}
        )

        assert blocks.required_inputs == []
        steps = next(inp for inp in blocks.inputs if inp.name == "num_inference_steps")
        assert steps.required is False
        assert steps.default == 50

    def test_composite_flags_never_reach_the_sub_block_declarations(self):
        class OptionalBatchBlock(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [InputParam("batch_size")]

            def __call__(self, components, state):
                return components, state

        class NeedsImageLatentsBlock(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [InputParam("image_latents", required=True)]

            def __call__(self, components, state):
                return components, state

        # a later requirement makes the composite's entry required, the first sub-block still says optional
        blocks = SequentialPipelineBlocks.from_blocks_dict(
            {"optional_batch": OptionalBatchBlock(), "prepare_latents": PrepareLatentsBlock()}
        )
        assert required_flags(blocks, "batch_size") == {"batch_size": True}
        assert required_flags(blocks.sub_blocks["optional_batch"], "batch_size") == {"batch_size": False}

        # a skippable producer makes the composite's entry optional, the leaf still requires it at runtime
        blocks = SequentialPipelineBlocks.from_blocks_dict(
            {"encode": SkippableEncodeBlocks(), "needs_latents": NeedsImageLatentsBlock()}
        )
        assert required_flags(blocks, "image_latents") == {"image_latents": False}
        assert required_flags(blocks.sub_blocks["needs_latents"], "image_latents") == {"image_latents": True}

        # a conditional block only requires what every branch requires, each branch keeps its own flag
        class Auto(AutoPipelineBlocks):
            block_classes = [NeedsImageLatentsBlock, EncodeImageBlock]
            block_names = ["needs_latents", "encode"]
            block_trigger_inputs = ["image_latents", None]

        auto = Auto()
        assert required_flags(auto, "image_latents", "image") == {"image_latents": False, "image": False}
        assert required_flags(auto.sub_blocks["needs_latents"], "image_latents") == {"image_latents": True}
        assert required_flags(auto.sub_blocks["encode"], "image") == {"image": True}

    def test_output_of_skippable_block_keeps_required_input_listed_but_optional(self):
        class NeedsImageLatentsBlock(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [InputParam("image_latents", required=True)]

            def __call__(self, components, state):
                return components, state

        blocks = SequentialPipelineBlocks.from_blocks_dict(
            {"encode": SkippableEncodeBlocks(), "consume": NeedsImageLatentsBlock()}
        )

        # the caller can pass `image` and let the encoder produce `image_latents`, or pass `image_latents` directly,
        # so neither is required on its own and both stay listed
        assert required_flags(blocks, "image", "image_latents") == {"image": False, "image_latents": False}
        assert blocks.required_inputs == []

    def test_output_of_some_branches_only_keeps_required_input_listed_but_optional(self):
        class EncodeImageAndCountBlock(EncodeImageBlock):
            @property
            def intermediate_outputs(self):
                return [OutputParam("image_latents"), OutputParam("batch_size")]

        class CountBlock(ModularPipelineBlocks):
            model_name = "test"

            @property
            def intermediate_outputs(self):
                return [OutputParam("batch_size")]

            def __call__(self, components, state):
                return components, state

        class EncodeOrCountBlocks(ConditionalPipelineBlocks):
            block_classes = [EncodeImageAndCountBlock, CountBlock]
            block_names = ["encode", "count"]
            block_trigger_inputs = ["image"]
            default_block_name = "count"

            def select_block(self, image=None):
                return "encode" if image is not None else "count"

        class NeedsBothBlock(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [InputParam("image_latents", required=True), InputParam("batch_size", required=True)]

            def __call__(self, components, state):
                return components, state

        blocks = SequentialPipelineBlocks.from_blocks_dict(
            {"encode": EncodeOrCountBlocks(), "consume": NeedsBothBlock()}
        )

        # the block always runs: `batch_size` comes out of every branch, so it is produced whatever runs and is not a
        # pipeline input. `image_latents` only comes out of `encode`, so it stays listed, optional: pass `image` or
        # pass `image_latents`
        assert [inp.name for inp in blocks.inputs] == ["image", "image_latents"]
        assert required_flags(blocks, "image", "image_latents") == {"image": False, "image_latents": False}
        assert blocks.required_inputs == []

    def test_skippable_block_nested_in_a_sequence_is_still_optional(self):
        class Encoders(SequentialPipelineBlocks):
            block_classes = [PrepareLatentsBlock, SkippableEncodeBlocks]
            block_names = ["prepare_latents", "encode"]

        # the sequence reports what `encode` reports: `image_latents` is only sometimes produced
        assert [(out.name, out._maybe) for out in Encoders().intermediate_outputs] == [
            ("latents", False),
            ("image_latents", True),
        ]

        blocks = SequentialPipelineBlocks.from_blocks_dict(
            {"encoders": Encoders(), "consume": NeedsImageLatentsBlock()}
        )
        # so wrapping `encode` in a sequence does not hide it: `image_latents` stays listed, optional. `latents` is
        # produced on every run and is not listed
        assert [inp.name for inp in blocks.inputs] == ["latents", "batch_size", "image", "image_latents"]
        assert required_flags(blocks, "image", "image_latents") == {"image": False, "image_latents": False}

    def test_branch_with_a_skippable_block_inside_does_not_always_produce(self):
        class EncodersWithOptionalImage(SequentialPipelineBlocks):
            block_classes = [PrepareLatentsBlock, SkippableEncodeBlocks]
            block_names = ["prepare_latents", "encode"]

        class EncodersOrEncode(ConditionalPipelineBlocks):
            block_classes = [EncodersWithOptionalImage, EncodeImageBlock]
            block_names = ["optional", "always"]
            block_trigger_inputs = ["mask"]
            default_block_name = "optional"

            def select_block(self, mask=None):
                return "always" if mask is not None else "optional"

        # `always` produces `image_latents` every time, `optional` only when `image` is passed, so the block as a whole
        # only sometimes does; `latents` comes out of one branch only
        assert {out.name: out._maybe for out in EncodersOrEncode().intermediate_outputs} == {
            "latents": True,
            "image_latents": True,
        }
        blocks = SequentialPipelineBlocks.from_blocks_dict(
            {"encode": EncodersOrEncode(), "consume": NeedsImageLatentsBlock()}
        )
        assert required_flags(blocks, "image_latents") == {"image_latents": False}

    def test_record_of_a_conditional_sub_block_is_prefixed_with_its_name(self):
        # `denoise` is a conditional block whose img2img branch requires `image_latents`, recorded as
        # {"img2img": REQUIRED}; the sequence keeps the record, keyed by `denoise.img2img`, and the input optional:
        # `encode` may produce it, so whether the caller has to pass it depends on the workflow
        class Img2ImgBlock(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [InputParam("image_latents", required=True), InputParam("strength", default=0.3)]

            def __call__(self, components, state):
                return components, state

        class Text2ImgBlock(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [InputParam("strength", default=0.9999)]

            def __call__(self, components, state):
                return components, state

        class DenoiseBlocks(ConditionalPipelineBlocks):
            block_classes = [Img2ImgBlock, Text2ImgBlock]
            block_names = ["img2img", "text2img"]
            block_trigger_inputs = ["image_latents"]
            default_block_name = "text2img"

            def select_block(self, image_latents=None):
                return "img2img" if image_latents is not None else "text2img"

        class Blocks(SequentialPipelineBlocks):
            block_classes = [SkippableEncodeBlocks, DenoiseBlocks]
            block_names = ["encode", "denoise"]

        inputs = {inp.name: inp for inp in Blocks().inputs}
        assert (inputs["image_latents"].required, inputs["image_latents"].defaults_by_block) == (
            False,
            {"denoise.img2img": REQUIRED},
        )
        assert inputs["strength"].defaults_by_block == {"denoise.img2img": 0.3, "denoise.text2img": 0.9999}
        # the conditional block on its own keeps the branch names
        inputs = {inp.name: inp for inp in DenoiseBlocks().inputs}
        assert inputs["image_latents"].defaults_by_block == {"img2img": REQUIRED}


class TestSequentialPipelineBlocksKwargsInputs:
    def test_kwargs_type_inputs_are_kept_apart(self):
        class TwoKwargsBlock(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [InputParam(kwargs_type="denoiser_input_fields"), InputParam(kwargs_type="controlnet_kwargs")]

            def __call__(self, components, state):
                return components, state

        blocks = SequentialPipelineBlocks.from_blocks_dict(
            {"first": PrepareLatentsBlock(), "second": TwoKwargsBlock()}
        )

        assert [inp.kwargs_type for inp in blocks.inputs if inp.name is None] == [
            "denoiser_input_fields",
            "controlnet_kwargs",
        ]


class TestAutoPipelineBlocksRequiredInputs:
    def test_inputs_produced_inside_each_workflow_are_not_required(self):
        class Text2ImageBlocks(SequentialPipelineBlocks):
            block_classes = [PrepareLatentsBlock, DenoiseBlock]
            block_names = ["prepare_latents", "denoise"]

        class Image2ImageBlocks(SequentialPipelineBlocks):
            block_classes = [EncodeImageBlock, PrepareLatentsBlock, DenoiseBlock]
            block_names = ["encode", "prepare_latents", "denoise"]

        class AutoBlocks(AutoPipelineBlocks):
            block_classes = [Image2ImageBlocks, Text2ImageBlocks]
            block_names = ["img2img", "text2img"]
            block_trigger_inputs = ["image", None]

        blocks = AutoBlocks()

        assert set(blocks.required_inputs) == {"batch_size", "prompt_embeds"}
        assert required_flags(blocks, "latents", "image", "batch_size", "prompt_embeds") == {
            "latents": False,
            "image": False,
            "batch_size": True,
            "prompt_embeds": True,
        }


class TestLoopSequentialPipelineBlocksRequiredInputs:
    def test_loop_inputs_and_unsatisfied_step_inputs_are_required(self):
        class DenoiserStep(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [InputParam("latents", required=True), InputParam(kwargs_type="denoiser_input_fields")]

            @property
            def intermediate_outputs(self):
                return [OutputParam("noise_pred")]

            def __call__(self, components, state, i, t):
                return components, state

        class SchedulerStep(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [
                    InputParam("noise_pred", required=True),
                    InputParam("latents", required=True),
                    InputParam(kwargs_type="scheduler_kwargs"),
                ]

            @property
            def intermediate_outputs(self):
                return [OutputParam("latents")]

            def __call__(self, components, state, i, t):
                return components, state

        class DenoiseLoop(LoopSequentialPipelineBlocks):
            model_name = "test"
            block_classes = [DenoiserStep, SchedulerStep]
            block_names = ["denoiser", "scheduler"]

            @property
            def loop_inputs(self):
                return [InputParam("timesteps", required=True), InputParam("num_inference_steps")]

        blocks = DenoiseLoop()

        assert set(blocks.required_inputs) == {"timesteps", "latents"}
        assert "noise_pred" not in [inp.name for inp in blocks.inputs]
        assert required_flags(blocks, "timesteps", "num_inference_steps", "latents") == {
            "timesteps": True,
            "num_inference_steps": False,
            "latents": True,
        }
        assert [inp.kwargs_type for inp in blocks.inputs if inp.name is None] == [
            "denoiser_input_fields",
            "scheduler_kwargs",
        ]

    def test_later_step_requirement_makes_an_optional_loop_input_required(self):
        class Step(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [InputParam("num_inference_steps", required=True), InputParam("guidance_scale", required=True)]

            def __call__(self, components, state, i, t):
                return components, state

        class Loop(LoopSequentialPipelineBlocks):
            model_name = "test"
            block_classes = [Step]
            block_names = ["step"]

            @property
            def loop_inputs(self):
                return [InputParam("num_inference_steps"), InputParam("guidance_scale", default=7.5)]

        # no default on the loop declaration, so the step's requirement wins; a default satisfies it
        assert required_flags(Loop(), "num_inference_steps", "guidance_scale") == {
            "num_inference_steps": True,
            "guidance_scale": False,
        }


class TestDefaultConflicts:
    def _blocks(self, first_default, second_default):
        class FirstBlock(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [InputParam("strength", default=first_default)]

            def __call__(self, components, state):
                return components, state

        class SecondBlock(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [InputParam("strength", default=second_default)]

            def __call__(self, components, state):
                return components, state

        return FirstBlock, SecondBlock

    def test_two_explicit_values_first_wins(self):
        # two explicit values for one shared input: the sequence keeps the first (and logs the other at debug level)
        FirstBlock, SecondBlock = self._blocks(0.3, 0.9999)

        class Blocks(SequentialPipelineBlocks):
            block_classes = [FirstBlock, SecondBlock]
            block_names = ["first", "second"]

        with quiet():
            inputs = Blocks().inputs
        assert [inp.default for inp in inputs if inp.name == "strength"] == [0.3]

    def test_a_value_and_an_open_declaration_are_not_a_conflict_either_way(self):
        # a later block with no default takes whatever is set; a later block with a default after an earlier block
        # that leaves the input open has a fallback the sequence does not use. Neither is a conflict
        FirstBlock, SecondBlock = self._blocks(0.3, None)

        class Blocks(SequentialPipelineBlocks):
            block_classes = [FirstBlock, SecondBlock]
            block_names = ["first", "second"]

        with quiet():
            inputs = Blocks().inputs
        assert [inp.default for inp in inputs if inp.name == "strength"] == [0.3]

        FirstBlock, SecondBlock = self._blocks(None, 0.3)

        class Blocks(SequentialPipelineBlocks):
            block_classes = [FirstBlock, SecondBlock]
            block_names = ["first", "second"]

        with quiet():
            inputs = Blocks().inputs
        # the sequence keeps the input open
        assert [inp.default for inp in inputs if inp.name == "strength"] == [None]

    def test_new_on_a_sub_block_settles_the_conflict(self):
        FirstBlock, SecondBlock = self._blocks(0.3, 0.9999)

        class Blocks(SequentialPipelineBlocks):
            # the shared block is right elsewhere with 0.3; this sequence wants 0.9999
            block_classes = [FirstBlock().new(inputs=[InputParam("strength", default=0.9999)]), SecondBlock]
            block_names = ["first", "second"]

        with quiet():
            inputs = Blocks().inputs
        assert [inp.default for inp in inputs if inp.name == "strength"] == [0.9999]
        # the sub-block fills in the settled default at run time, it is its own declaration
        assert [inp.default for inp in Blocks().sub_blocks["first"].inputs] == [0.9999]

    def test_loop_input_without_default_is_kept_open(self):
        FirstBlock, _ = self._blocks(0.9999, None)

        class Loop(LoopSequentialPipelineBlocks):
            model_name = "test"
            block_classes = [FirstBlock]
            block_names = ["first"]

            @property
            def loop_inputs(self):
                return [InputParam("timesteps", required=True), InputParam("strength")]

        with quiet():
            inputs = Loop().inputs
        # the loop declaration comes first and wins; the sub-block's default is a fallback, not a conflict
        assert [inp.default for inp in inputs if inp.name == "strength"] == [None]

    def test_loop_blocks_keep_the_earlier_declaration(self):
        FirstBlock, SecondBlock = self._blocks(0.3, 0.9999)

        class Loop(LoopSequentialPipelineBlocks):
            model_name = "test"
            block_classes = [FirstBlock, SecondBlock]
            block_names = ["first", "second"]

            @property
            def loop_inputs(self):
                return [InputParam("timesteps", required=True)]

        with quiet():
            inputs = Loop().inputs
        assert [inp.default for inp in inputs if inp.name == "strength"] == [0.3]

        with quiet():
            inputs = Loop.from_blocks_dict(
                {"first": FirstBlock().new(inputs=[InputParam("strength", default=0.9999)]), "second": SecondBlock()}
            ).inputs
        assert [inp.default for inp in inputs if inp.name == "strength"] == [0.9999]

    def test_value_after_branch_defaults_keeps_the_branch_defaults(self):
        # the running branch fills in its default first, so a later leaf default is never the one in use and the
        # per-branch record stays. A conditional block after a value declares fallbacks for when the input is still
        # unset: the value stays
        FirstBlock, _ = self._blocks(0.5, None)
        Auto = strength_auto_blocks(0.3, 0.9999)

        class AutoThenValue(SequentialPipelineBlocks):
            block_classes = [Auto, FirstBlock]
            block_names = ["auto", "first"]

        with quiet():
            inputs = AutoThenValue().inputs
        assert [(inp.default, inp.defaults_by_block) for inp in inputs if inp.name == "strength"] == [
            (None, {"auto.inpaint": 0.9999, "auto.img2img": 0.3})
        ]

        class ValueThenAuto(SequentialPipelineBlocks):
            block_classes = [FirstBlock, Auto]
            block_names = ["first", "auto"]

        with quiet():
            inputs = ValueThenAuto().inputs
        assert [(inp.default, inp.defaults_by_block) for inp in inputs if inp.name == "strength"] == [(0.5, None)]

    def test_declarations_without_a_value_never_conflict(self):
        # `None` and per-branch defaults both carry `default=None`: nothing to pre-fill, nothing to disagree about
        Auto = strength_auto_blocks(0.3, 0.9999)
        NoneBlock, _ = self._blocks(None, None)

        class NoneThenAuto(SequentialPipelineBlocks):
            block_classes = [NoneBlock, Auto]
            block_names = ["first", "auto"]

        with quiet():
            (strength,) = [inp for inp in NoneThenAuto().inputs if inp.name == "strength"]
        assert (strength.default, strength.defaults_by_block) == (None, None)

        class AutoThenNone(SequentialPipelineBlocks):
            block_classes = [Auto, NoneBlock]
            block_names = ["auto", "first"]

        with quiet():
            AutoThenNone().inputs

        class AutoThenAuto(SequentialPipelineBlocks):
            block_classes = [Auto, strength_auto_blocks(0.9999, 0.3)]
            block_names = ["a", "b"]

        with quiet():
            AutoThenAuto().inputs

    def test_value_after_a_required_branch_keeps_the_record(self):
        # one branch requires the input and the other does not declare it, so the auto block records
        # {"img2img": REQUIRED} with default `None`; a later 0.5 does not change that
        FirstBlock, _ = self._blocks(0.5, None)
        Auto = strength_auto_blocks(REQUIRED, ...)

        class Blocks(SequentialPipelineBlocks):
            block_classes = [Auto, FirstBlock]
            block_names = ["auto", "first"]

        with quiet():
            (strength,) = [inp for inp in Blocks().inputs if inp.name == "strength"]
        assert (strength.required, strength.default, strength.defaults_by_block) == (
            False,
            None,
            {"auto.img2img": REQUIRED},
        )

    def test_required_branch_does_not_satisfy_a_later_requirement(self):
        # the auto block's map holds only a requirement and a `None`, so a later block requiring the input makes it
        # a required pipeline input
        Auto = strength_auto_blocks(REQUIRED, ...)

        class Blocks(SequentialPipelineBlocks):
            block_classes = [Auto, RequiresStrengthBlock]
            block_names = ["auto", "requires"]

        assert required_flags(Blocks(), "strength") == {"strength": True}

    def test_branch_defaults_satisfy_a_later_requirement(self):
        class Blocks(SequentialPipelineBlocks):
            block_classes = [strength_auto_blocks(0.3, 0.9999), RequiresStrengthBlock]
            block_names = ["auto", "requires"]

        # the branch that runs fills in its default before `requires` reads it
        assert required_flags(Blocks(), "strength") == {"strength": False}


class UNetBlock(ModularPipelineBlocks):
    model_name = "test"

    @property
    def expected_components(self):
        return [ComponentSpec("unet", description="the denoiser")]

    @property
    def expected_configs(self):
        return [ConfigSpec("scale", default=8)]

    @property
    def inputs(self):
        return [InputParam("latents", description="noisy latents"), InputParam("batch_size", required=True)]

    @property
    def intermediate_outputs(self):
        return [OutputParam("latents", description="denoised latents")]

    def __call__(self, components, state):
        return components, state


class TestNew:
    """`ModularPipelineBlocks.new()`: a copy of a block with some of its declarations replaced."""

    def test_replaces_the_listed_declarations_and_leaves_the_block_alone(self):
        block = PrepareLatentsBlock()
        copy_ = block.new(
            description="prepares latents for a fixed batch",
            inputs=[InputParam("batch_size", default=1)],
            intermediate_outputs=[OutputParam("latents", description="initial latents")],
        )

        assert copy_.description == "prepares latents for a fixed batch"
        (batch_size,) = [inp for inp in copy_.inputs if inp.name == "batch_size"]
        assert batch_size.default == 1 and batch_size.required is False
        assert "batch_size" not in copy_.required_inputs
        assert [out.description for out in copy_.intermediate_outputs if out.name == "latents"] == ["initial latents"]
        # the other entries are the block's own
        assert [inp.name for inp in copy_.inputs] == [inp.name for inp in block.inputs]
        # the block itself is untouched, and still the class the copy derives from
        (batch_size,) = [inp for inp in block.inputs if inp.name == "batch_size"]
        assert batch_size.required is True and batch_size.default is None
        assert isinstance(copy_, PrepareLatentsBlock) and type(copy_).__name__ == "PrepareLatentsBlock"

    def test_components_configs_and_outputs(self):
        block = UNetBlock()
        copy_ = block.new(
            components=[ComponentSpec("unet", description="any UNet2DConditionModel")],
            configs=[ConfigSpec("scale", default=16)],
            outputs=[OutputParam("latents", description="the final latents")],
        )
        assert [c.description for c in copy_.expected_components if c.name == "unet"] == ["any UNet2DConditionModel"]
        assert [c.default for c in copy_.expected_configs if c.name == "scale"] == [16]
        assert [out.description for out in copy_.outputs if out.name == "latents"] == ["the final latents"]
        assert [c.default for c in block.expected_configs if c.name == "scale"] == [8]

    def test_keeps_what_the_block_was_created_with(self):
        class RenamingBlock(ModularPipelineBlocks):
            model_name = "test"

            def __init__(self, latents_name="latents"):
                super().__init__()
                self.latents_name = latents_name

            @property
            def inputs(self):
                return [InputParam(self.latents_name), InputParam("strength", default=0.3)]

            def __call__(self, components, state):
                return components, state

        copy_ = RenamingBlock(latents_name="image_latents").new(inputs=[InputParam("strength", default=0.9)])
        assert [(inp.name, inp.default) for inp in copy_.inputs] == [("image_latents", None), ("strength", 0.9)]

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="does not declare an input 'strenght'"):
            PrepareLatentsBlock().new(inputs=[InputParam("strenght", default=0.5)])
        with pytest.raises(ValueError, match="does not declare a component 'vae'"):
            UNetBlock().new(components=[ComponentSpec("vae", description="x")])

    def test_a_block_with_sub_blocks_raises(self):
        class Blocks(SequentialPipelineBlocks):
            block_classes = [PrepareLatentsBlock, UNetBlock]
            block_names = ["prepare_latents", "unet"]

        with pytest.raises(TypeError, match="derives its declarations from its sub-blocks"):
            Blocks().new(inputs=[InputParam("batch_size", default=1)])

        class Loop(LoopSequentialPipelineBlocks):
            model_name = "test"
            block_classes = [UNetBlock]
            block_names = ["unet"]

        with pytest.raises(TypeError, match="derives its declarations from its sub-blocks"):
            Loop().new(configs=[ConfigSpec("scale", default=16)])


class TestFromBlocksDict:
    def test_description_and_outputs(self):
        blocks = SequentialPipelineBlocks.from_blocks_dict(
            {"prepare_latents": PrepareLatentsBlock(), "unet": UNetBlock()},
            description="prepares and denoises",
            outputs=[OutputParam("latents", description="denoised latents")],
        )
        assert blocks.description == "prepares and denoises"
        assert [out.name for out in blocks.outputs] == ["latents"]
        # without them: no description, every intermediate output
        blocks = SequentialPipelineBlocks.from_blocks_dict(
            {"prepare_latents": PrepareLatentsBlock(), "unet": UNetBlock()}
        )
        assert blocks.description == ""
        assert [out.name for out in blocks.outputs] == [out.name for out in blocks.intermediate_outputs]

    def test_an_output_no_block_produces_raises(self):
        with pytest.raises(ValueError, match="\\['images'\\] are not intermediate outputs"):
            SequentialPipelineBlocks.from_blocks_dict({"unet": UNetBlock()}, outputs=[OutputParam("images")])

    def test_execution_blocks_keep_the_outputs_of_the_full_blocks(self):
        class DecodeBlock(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [InputParam("latents", required=True)]

            @property
            def intermediate_outputs(self):
                return [OutputParam("images")]

            def __call__(self, components, state):
                return components, state

        class Blocks(SequentialPipelineBlocks):
            block_classes = [strength_auto_blocks(0.3, 0.9999), UNetBlock, DecodeBlock]
            block_names = ["auto", "unet", "decode"]

            @property
            def outputs(self):
                # what the pipeline is for, plus one that only the img2img branch produces
                return [OutputParam("images"), OutputParam("image_latents")]

        workflow = Blocks().get_execution_blocks(mask=True)
        assert [out.name for out in workflow.outputs] == ["images"]
        assert len(workflow.intermediate_outputs) > 1
