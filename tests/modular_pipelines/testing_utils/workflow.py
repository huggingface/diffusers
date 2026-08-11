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

from diffusers import ModularPipeline


class ModularWorkflowTesterMixin:
    """
    Tests for the workflows a blocks class declares through `_workflow_map`: which blocks each workflow resolves to,
    the components/configs/inputs it ends up expecting, and the two entry points that prune by workflow name
    (`ModularPipeline.from_pretrained(..., workflow=...)` and `load_components(workflow=...)`).

    Every test skips when the blocks class declares no `_workflow_map`, so this mixin is safe to compose onto any
    modular pipeline test.
    """

    def test_workflow_map(self):
        blocks = self.pipeline_blocks_class()
        if blocks._workflow_map is None:
            pytest.skip("Skipping test as _workflow_map is not set")

        assert hasattr(self, "expected_workflow_blocks") and self.expected_workflow_blocks, (
            "expected_workflow_blocks must be defined in the test class"
        )

        for workflow_name, expected_blocks in self.expected_workflow_blocks.items():
            workflow_blocks = blocks.get_workflow(workflow_name)
            actual_blocks = list(workflow_blocks.sub_blocks.items())

            # Check that the number of blocks matches
            assert len(actual_blocks) == len(expected_blocks), (
                f"Workflow '{workflow_name}' has {len(actual_blocks)} blocks, expected {len(expected_blocks)}"
            )

            # Check that each block name and type matches
            for i, ((actual_name, actual_block), (expected_name, expected_class_name)) in enumerate(
                zip(actual_blocks, expected_blocks)
            ):
                assert actual_name == expected_name
                assert actual_block.__class__.__name__ == expected_class_name, (
                    f"Workflow '{workflow_name}': block '{actual_name}' has type "
                    f"{actual_block.__class__.__name__}, expected {expected_class_name}"
                )

        # a `_workflow_map` value is either one trigger dict or a tuple of trigger dicts — alternative spellings
        # of the same workflow, which must all resolve the same blocks since `get_workflow` prunes with the first
        for workflow_name, trigger_inputs in blocks._workflow_map.items():
            if isinstance(trigger_inputs, dict):
                continue
            assert isinstance(trigger_inputs, tuple) and all(
                isinstance(alternative, dict) for alternative in trigger_inputs
            ), (
                f"Workflow '{workflow_name}': a `_workflow_map` value must be a trigger dict or a tuple of "
                f"trigger dicts, got {trigger_inputs!r}"
            )
            resolved = [list(blocks.get_execution_blocks(**alternative).sub_blocks) for alternative in trigger_inputs]
            for alternative_blocks in resolved[1:]:
                assert alternative_blocks == resolved[0], (
                    f"Workflow '{workflow_name}': its trigger spellings resolve different blocks: "
                    f"{resolved[0]} vs {alternative_blocks}"
                )

    def test_workflow_defaults(self):
        if not getattr(self, "expected_workflow_defaults", None):
            pytest.skip("Skipping test as expected_workflow_defaults is not set")

        blocks = self.pipeline_blocks_class()
        for workflow_name, expected_defaults in self.expected_workflow_defaults.items():
            workflow_blocks = blocks.get_workflow(workflow_name)

            # components: every one of the workflow is named with its class, so one appearing, disappearing or
            # changing type fails loudly
            component_specs = {spec.name: spec for spec in workflow_blocks.expected_components}
            expected_components = expected_defaults["components"]
            assert set(component_specs) == set(expected_components), (
                f"Workflow '{workflow_name}' expects components {sorted(component_specs)}, "
                f"the test expects {sorted(expected_components)}"
            )
            for component_name, expected_class_name in expected_components.items():
                actual_class_name = component_specs[component_name].type_hint.__name__
                assert actual_class_name == expected_class_name, (
                    f"Workflow '{workflow_name}': component '{component_name}' is a {actual_class_name}, "
                    f"expected {expected_class_name}"
                )

            # configs: the pipeline-level configs the workflow declares, with their defaults
            config_specs = {spec.name: spec.default for spec in workflow_blocks.expected_configs}
            expected_configs = expected_defaults.get("configs", {})
            assert config_specs == expected_configs, (
                f"Workflow '{workflow_name}' declares configs {config_specs}, the test expects {expected_configs}"
            )

            # inputs: `inputs` names every optional input with its default and `required_inputs` the required ones,
            # and together they are exactly the workflow's inputs. kwargs-style inputs (e.g.
            # **denoiser_input_fields) have no name and no default to pin
            input_params = {param.name: param for param in workflow_blocks.inputs if param.name is not None}
            expected_inputs = expected_defaults["inputs"]
            expected_required = expected_defaults.get("required_inputs", [])
            assert set(input_params) == set(expected_inputs) | set(expected_required), (
                f"Workflow '{workflow_name}' takes inputs {sorted(input_params)}, "
                f"the test expects {sorted(set(expected_inputs) | set(expected_required))}"
            )
            for input_name in expected_required:
                assert input_params[input_name].required, (
                    f"Workflow '{workflow_name}': input '{input_name}' should be required"
                )
            for input_name, expected_default in expected_inputs.items():
                param = input_params[input_name]
                assert not param.required and param.default == expected_default, (
                    f"Workflow '{workflow_name}': input '{input_name}' default is "
                    f"{param.default!r} (required={param.required}), expected {expected_default!r}"
                )

    def test_from_pretrained_workflow(self):
        blocks = self.pipeline_blocks_class()
        if blocks._workflow_map is None:
            pytest.skip("Skipping test as _workflow_map is not set")

        for workflow_name in blocks.available_workflows:
            # the workflow argument should be equivalent to pruning the blocks by hand
            pipe = ModularPipeline.from_pretrained(self.pretrained_model_name_or_path, workflow=workflow_name)
            ref_pipe = blocks.get_workflow(workflow_name).init_pipeline(self.pretrained_model_name_or_path)
            assert set(pipe.component_names) == set(ref_pipe.component_names), (
                f"Workflow '{workflow_name}': pipeline expects components {sorted(pipe.component_names)}, "
                f"the workflow blocks expect {sorted(ref_pipe.component_names)}"
            )
            for name in pipe.pretrained_component_names:
                assert pipe.get_component_spec(name) == ref_pipe.get_component_spec(name), (
                    f"Workflow '{workflow_name}': component '{name}' has a different spec than the one "
                    f"created from the workflow blocks"
                )

        with pytest.raises(ValueError, match="Available workflows"):
            ModularPipeline.from_pretrained(self.pretrained_model_name_or_path, workflow="not_a_workflow")

    def test_load_components_workflow(self):
        blocks = self.pipeline_blocks_class()
        if blocks._workflow_map is None:
            pytest.skip("Skipping test as _workflow_map is not set")

        workflow_name = blocks.available_workflows[0]

        # a full pipeline restricted at load time should load the same components as a pipeline
        # created from the pruned workflow blocks
        pipe = ModularPipeline.from_pretrained(self.pretrained_model_name_or_path)
        pipe.load_components(workflow=workflow_name)
        ref_pipe = blocks.get_workflow(workflow_name).init_pipeline(self.pretrained_model_name_or_path)
        ref_pipe.load_components()

        loaded = {name for name in pipe.pretrained_component_names if pipe.components[name] is not None}
        ref_loaded = {name for name in ref_pipe.pretrained_component_names if ref_pipe.components[name] is not None}
        assert loaded == ref_loaded, (
            f"Workflow '{workflow_name}': load_components(workflow=...) loaded {sorted(loaded)}, "
            f"the pipeline created from the workflow blocks loaded {sorted(ref_loaded)}"
        )

        with pytest.raises(ValueError, match="not both"):
            pipe.load_components(names="unet", workflow=workflow_name)
