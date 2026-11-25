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

import inspect

from diffusers import KolorsPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline

from ...testing_utils import (
    assert_tensors_close,
    require_accelerate_version_greater,
    require_accelerator,
    torch_device,
)
from .common import BasePipelineOutputMixin


# The repo and constructor kwargs used to build the "original" pipeline a variant is derived from, keyed by that
# original pipeline class.
ORIGINAL_PIPELINE_REPOS = {
    StableDiffusionPipeline: ("hf-internal-testing/tiny-stable-diffusion-torch", {"requires_safety_checker": False}),
    StableDiffusionXLPipeline: (
        "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
        {"requires_aesthetics_score": True, "force_zeros_for_empty_prompt": False},
    ),
    KolorsPipeline: ("hf-internal-testing/tiny-kolors-pipe", {"force_zeros_for_empty_prompt": False}),
}


class FromPipeTesterMixin(BasePipelineOutputMixin):
    """`DiffusionPipeline.from_pipe` tests for pipelines that are variants of an existing one.

    Composed with `BasePipelineTesterConfig`, which supplies `pipeline_class`, `get_dummy_components()` and
    `get_dummy_inputs()`.
    """

    # Set on the test class to pull the original pipeline from a repo other than the default for its class.
    original_pipeline_repo = None

    @property
    def original_pipeline_class(self):
        """The pipeline this one is a variant of — the source `from_pipe` is expected to round-trip through."""
        name = self.pipeline_class.__name__.lower()
        if "xl" in name:
            return StableDiffusionXLPipeline
        elif "kolors" in name:
            return KolorsPipeline
        return StableDiffusionPipeline

    def get_dummy_inputs_pipe(self):
        inputs = self.get_dummy_inputs()
        inputs["return_dict"] = False
        return inputs

    def get_dummy_inputs_for_pipe_original(self):
        """The dummy inputs, restricted to the parameters the original pipeline's `__call__` accepts."""
        original_call_params = set(inspect.signature(self.original_pipeline_class.__call__).parameters.keys())
        return {k: v for k, v in self.get_dummy_inputs_pipe().items() if k in original_call_params}

    def _split_components(self, components):
        """Split the given components into what the original pipeline expects and what only this one does."""
        original_expected_modules, _ = self.original_pipeline_class._get_signature_keys(self.original_pipeline_class)

        # components of this pipeline that the original one also expects
        original_pipe_components = {}
        # components this pipeline doesn't have, but the original one expects
        original_pipe_additional_components = {}
        # components this pipeline has, but the original one doesn't expect
        current_pipe_additional_components = {}

        for name, component in components.items():
            if name in original_expected_modules:
                original_pipe_components[name] = component
            else:
                current_pipe_additional_components[name] = component

        for name in original_expected_modules:
            if name not in original_pipe_components:
                if name in self.original_pipeline_class._optional_components:
                    original_pipe_additional_components[name] = None
                else:
                    raise ValueError(f"missing required module for {self.original_pipeline_class.__name__}: {name}")

        return (
            {**original_pipe_components, **original_pipe_additional_components},
            current_pipe_additional_components,
        )

    def _build_original_pipeline(self, components):
        """Build the original pipeline out of `components`, plus the components only this pipeline has.

        Both pipelines are built from the *same* component instances — rebuilding them per pipeline would make the
        comparison depend on `get_dummy_components()` being bit-identical across calls.
        """
        original_components, current_pipe_additional_components = self._split_components(components)
        pipe_original = self.original_pipeline_class(**original_components)
        pipe_original.set_progress_bar_config(disable=None)
        return pipe_original, current_pipe_additional_components

    def test_from_pipe_consistent_config(self):
        original_repo, original_kwargs = ORIGINAL_PIPELINE_REPOS[self.original_pipeline_class]
        original_repo = self.original_pipeline_repo or original_repo

        # create original_pipeline_class(sd/sdxl/kolors)
        pipe_original = self.original_pipeline_class.from_pretrained(original_repo, **original_kwargs)

        # original_pipeline_class -> pipeline_class
        pipe_additional_components = {
            name: component
            for name, component in self.get_dummy_components().items()
            if name not in pipe_original.components
        }
        pipe = self.pipeline_class.from_pipe(pipe_original, **pipe_additional_components)

        # pipeline_class -> original_pipeline_class
        original_pipe_additional_components = {}
        for name, component in pipe_original.components.items():
            if name not in pipe.components or not isinstance(component, pipe.components[name].__class__):
                original_pipe_additional_components[name] = component

        pipe_original_2 = self.original_pipeline_class.from_pipe(pipe, **original_pipe_additional_components)

        # compare the config
        original_config = {k: v for k, v in pipe_original.config.items() if not k.startswith("_")}
        original_config_2 = {k: v for k, v in pipe_original_2.config.items() if not k.startswith("_")}
        assert original_config_2 == original_config

    def test_from_pipe_consistent_forward_pass(self, expected_max_diff=1e-3):
        components = self.get_dummy_components()
        pipe_original, current_pipe_additional_components = self._build_original_pipeline(components)
        pipe_original.to(torch_device)

        output_original = pipe_original(**self.get_dummy_inputs_for_pipe_original())[0]

        # `from_pipe` must not repurpose the original pipeline's attention processors — PAG and friends install
        # their own on the derived pipeline only.
        original_attn_processor_types = {
            name: {k: type(v) for k, v in component.attn_processors.items()}
            for name, component in pipe_original.components.items()
            if hasattr(component, "attn_processors")
        }

        pipe = self.get_pipeline(**components).to(torch_device)
        output = pipe(**self.get_dummy_inputs_pipe())[0]

        pipe_from_original = self.pipeline_class.from_pipe(pipe_original, **current_pipe_additional_components)
        pipe_from_original.to(torch_device)
        pipe_from_original.set_progress_bar_config(disable=None)
        output_from_original = pipe_from_original(**self.get_dummy_inputs_pipe())[0]

        assert_tensors_close(
            output_from_original,
            output,
            atol=expected_max_diff,
            msg="The outputs of the pipelines created with `from_pipe` and `__init__` are different.",
        )

        output_original_2 = pipe_original(**self.get_dummy_inputs_for_pipe_original())[0]
        assert_tensors_close(
            output_original_2,
            output_original,
            atol=expected_max_diff,
            msg="`from_pipe` should not change the output of original pipeline.",
        )

        for name, expected_types in original_attn_processor_types.items():
            component = pipe_original.components[name]
            assert {k: type(v) for k, v in component.attn_processors.items()} == expected_types, (
                f"`from_pipe` changed the attention processors of `{name}` in the original pipeline."
            )

    @require_accelerator
    @require_accelerate_version_greater("0.14.0")
    def test_from_pipe_consistent_forward_pass_cpu_offload(self, expected_max_diff=1e-3):
        components = self.get_dummy_components()

        # Build the original pipeline before running anything. Both pipelines share one scheduler object, and some
        # `__init__`s edit that object in place: `StableDiffusionPipeline`, for one, rewrites a `steps_offset` of 0
        # to 1. Building it later would put the two forward passes below on different schedules.
        pipe_original, current_pipe_additional_components = self._build_original_pipeline(components)

        pipe = self.get_pipeline(**components)
        pipe.enable_model_cpu_offload(device=torch_device)
        output = pipe(**self.get_dummy_inputs_pipe())[0]

        pipe_from_original = self.pipeline_class.from_pipe(pipe_original, **current_pipe_additional_components)
        pipe_from_original.set_progress_bar_config(disable=None)
        pipe_from_original.enable_model_cpu_offload(device=torch_device)
        output_from_original = pipe_from_original(**self.get_dummy_inputs_pipe())[0]

        assert_tensors_close(
            output_from_original,
            output,
            atol=expected_max_diff,
            msg="The outputs of the pipelines created with `from_pipe` and `__init__` are different.",
        )
