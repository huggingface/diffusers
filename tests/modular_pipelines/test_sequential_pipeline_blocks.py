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

import warnings

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


def strength_auto_blocks(img2img_default, inpaint_default):
    """An auto block whose two branches declare `strength` with their own default (`None` for no default)."""

    class Img2ImgBlock(ModularPipelineBlocks):
        model_name = "test"

        @property
        def inputs(self):
            return [InputParam("image", required=True), InputParam("strength", default=img2img_default)]

        def __call__(self, components, state):
            return components, state

    class InpaintBlock(ModularPipelineBlocks):
        model_name = "test"

        @property
        def inputs(self):
            return [InputParam("mask", required=True), InputParam("strength", default=inpaint_default)]

        def __call__(self, components, state):
            return components, state

    class Auto(AutoPipelineBlocks):
        block_classes = [InpaintBlock, Img2ImgBlock]
        block_names = ["inpaint", "img2img"]
        block_trigger_inputs = ["mask", "image"]

    return Auto


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

    def test_conflicting_defaults_are_deprecated_first_wins(self):
        FirstBlock, SecondBlock = self._blocks(0.3, 0.9999)

        class Blocks(SequentialPipelineBlocks):
            block_classes = [FirstBlock, SecondBlock]
            block_names = ["first", "second"]

        with pytest.warns(FutureWarning, match="'second' declares default 0.9999 for input 'strength'"):
            inputs = Blocks().inputs
        # the first declaration wins until the deprecation turns into an error
        assert [inp.default for inp in inputs if inp.name == "strength"] == [0.3]

    def test_later_none_default_is_not_a_conflict(self):
        FirstBlock, SecondBlock = self._blocks(0.3, None)

        class Blocks(SequentialPipelineBlocks):
            block_classes = [FirstBlock, SecondBlock]
            block_names = ["first", "second"]

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            Blocks().inputs

    def test_default_override_settles_the_conflict(self):
        FirstBlock, SecondBlock = self._blocks(0.3, 0.9999)

        class Blocks(SequentialPipelineBlocks):
            block_classes = [FirstBlock, SecondBlock]
            block_names = ["first", "second"]
            input_overrides = {"strength": InputParam("strength", default=0.9999)}

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            inputs = Blocks().inputs
        assert [inp.default for inp in inputs if inp.name == "strength"] == [0.9999]

    def test_loop_input_without_default_is_not_a_conflict(self):
        FirstBlock, _ = self._blocks(0.9999, None)

        class Loop(LoopSequentialPipelineBlocks):
            model_name = "test"
            block_classes = [FirstBlock]
            block_names = ["first"]

            @property
            def loop_inputs(self):
                return [InputParam("timesteps", required=True), InputParam("strength")]

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            inputs = Loop().inputs
        # the loop declaration has no opinion, so the step's default is the one the loop reports
        assert [inp.default for inp in inputs if inp.name == "strength"] == [None]

    def test_loop_blocks_deprecate_and_accept_the_override(self):
        FirstBlock, SecondBlock = self._blocks(0.3, 0.9999)

        class Loop(LoopSequentialPipelineBlocks):
            model_name = "test"
            block_classes = [FirstBlock, SecondBlock]
            block_names = ["first", "second"]

            @property
            def loop_inputs(self):
                return [InputParam("timesteps", required=True)]

        with pytest.warns(FutureWarning, match="'strength'"):
            Loop().inputs

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            inputs = Loop.from_blocks_dict(
                {"first": FirstBlock(), "second": SecondBlock()},
                input_overrides={"strength": InputParam("strength", default=0.9999)},
            ).inputs
        assert [inp.default for inp in inputs if inp.name == "strength"] == [0.9999]

    def test_branch_defaults_against_a_value_are_a_conflict(self):
        FirstBlock, _ = self._blocks(0.5, None)
        Auto = strength_auto_blocks(0.3, 0.9999)

        class ValueThenAuto(SequentialPipelineBlocks):
            block_classes = [FirstBlock, Auto]
            block_names = ["first", "auto"]

        with pytest.warns(FutureWarning, match="'auto' declares per-branch defaults .* declares default 0.5"):
            inputs = ValueThenAuto().inputs
        assert [(inp.default, inp.defaults_by_block) for inp in inputs if inp.name == "strength"] == [(0.5, None)]

        class AutoThenValue(SequentialPipelineBlocks):
            block_classes = [Auto, FirstBlock]
            block_names = ["auto", "first"]

        with pytest.warns(FutureWarning, match="'first' declares default 0.5 .* declares per-branch defaults"):
            AutoThenValue().inputs

    def test_branch_without_default_has_no_opinion(self):
        # one branch agrees with the earlier value and the other declares no default: nothing to settle
        FirstBlock, _ = self._blocks(0.3, None)

        class Blocks(SequentialPipelineBlocks):
            block_classes = [FirstBlock, strength_auto_blocks(0.3, None)]
            block_names = ["first", "auto"]

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            Blocks().inputs

    def test_branch_defaults_are_carried_onto_an_entry_without_default(self):
        FirstBlock, _ = self._blocks(None, None)

        class Blocks(SequentialPipelineBlocks):
            block_classes = [FirstBlock, strength_auto_blocks(0.3, 0.9999)]
            block_names = ["first", "auto"]

        blocks = Blocks()
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            (strength,) = [inp for inp in blocks.inputs if inp.name == "strength"]
        assert strength.default is None
        assert strength.defaults_by_block == {"auto.inpaint": 0.9999, "auto.img2img": 0.3}
        (first,) = [inp for inp in blocks.sub_blocks["first"].inputs if inp.name == "strength"]
        assert first.defaults_by_block is None

    def test_branch_defaults_satisfy_a_later_requirement(self):
        class RequiresStrengthBlock(ModularPipelineBlocks):
            model_name = "test"

            @property
            def inputs(self):
                return [InputParam("strength", required=True)]

            def __call__(self, components, state):
                return components, state

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


class TestOverrides:
    def _sequential(self, **overrides):
        class Blocks(SequentialPipelineBlocks):
            block_classes = [PrepareLatentsBlock, UNetBlock]
            block_names = ["prepare_latents", "unet"]

        for name, value in overrides.items():
            setattr(Blocks, name, value)
        return Blocks()

    def test_input_override_replaces_the_derived_declaration(self):
        blocks = self._sequential(input_overrides={"batch_size": InputParam("batch_size", default=1)})
        (batch_size,) = [inp for inp in blocks.inputs if inp.name == "batch_size"]
        assert batch_size.default == 1
        assert batch_size.required is False
        assert "batch_size" not in blocks.required_inputs

    def test_override_does_not_mutate_the_sub_block_declaration(self):
        blocks = self._sequential(input_overrides={"batch_size": InputParam("batch_size", default=1)})
        _ = blocks.inputs
        (batch_size,) = [inp for inp in blocks.sub_blocks["prepare_latents"].inputs if inp.name == "batch_size"]
        assert batch_size.required is True
        assert batch_size.default is None

    def test_output_component_and_config_overrides(self):
        blocks = self._sequential(
            intermediate_output_overrides={"latents": OutputParam("latents", description="final latents")},
            component_overrides={"unet": ComponentSpec("unet", description="any UNet2DConditionModel")},
            config_overrides={"scale": ConfigSpec("scale", default=16)},
        )
        assert [out.description for out in blocks.intermediate_outputs if out.name == "latents"] == ["final latents"]
        assert [c.description for c in blocks.expected_components if c.name == "unet"] == ["any UNet2DConditionModel"]
        assert [c.default for c in blocks.expected_configs if c.name == "scale"] == [16]

    def test_unknown_name_raises(self):
        blocks = self._sequential(input_overrides={"strenght": InputParam("strenght", default=0.5)})
        with pytest.raises(ValueError, match="strenght"):
            blocks.inputs
        blocks = self._sequential(component_overrides={"vae": ComponentSpec("vae", description="x")})
        with pytest.raises(ValueError, match="vae"):
            blocks.expected_components

    def test_partial_dict_override_raises(self):
        blocks = self._sequential(input_overrides={"batch_size": {"default": 1}})
        with pytest.raises(TypeError, match="must be of type InputParam"):
            blocks.inputs

    def test_from_blocks_dict_accepts_overrides(self):
        blocks = SequentialPipelineBlocks.from_blocks_dict(
            {"prepare_latents": PrepareLatentsBlock(), "unet": UNetBlock()},
            input_overrides={"batch_size": InputParam("batch_size", default=2)},
            intermediate_output_overrides={"latents": OutputParam("latents", description="final latents")},
            config_overrides={"scale": ConfigSpec("scale", default=16)},
        )
        assert [inp.default for inp in blocks.inputs if inp.name == "batch_size"] == [2]
        assert [out.description for out in blocks.intermediate_outputs if out.name == "latents"] == ["final latents"]
        assert [c.default for c in blocks.expected_configs if c.name == "scale"] == [16]

        loop = LoopSequentialPipelineBlocks.from_blocks_dict(
            {"unet": UNetBlock()},
            component_overrides={"unet": ComponentSpec("unet", description="any UNet2DConditionModel")},
        )
        assert [c.description for c in loop.expected_components if c.name == "unet"] == ["any UNet2DConditionModel"]

    def test_auto_blocks_override_replaces_the_per_block_defaults(self):
        Auto = strength_auto_blocks(0.3, 0.9999)

        (strength,) = [inp for inp in Auto().inputs if inp.name == "strength"]
        assert strength.default is None
        assert strength.defaults_by_block == {"inpaint": 0.9999, "img2img": 0.3}

        class AutoWithDefault(Auto):
            input_overrides = {"strength": InputParam("strength", default=0.5)}

        (strength,) = [inp for inp in AutoWithDefault().inputs if inp.name == "strength"]
        assert strength.default == 0.5
        assert strength.defaults_by_block is None
