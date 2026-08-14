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

import json
import weakref

import pytest
import torch

import diffusers
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils import logging

from ...testing_utils import CaptureLogger, torch_device
from .common import BaseModularPipelineOutputMixin
from .utils import get_specified_components


class ModularLoadingTesterMixin(BaseModularPipelineOutputMixin):
    """
    Serialization and component lifecycle tests for a modular pipeline: `save_pretrained`/`from_pretrained`
    round-trips, what `modular_model_index.json` records, and `load_components`/`unload_components`.

    Tests that assert on *device memory* after unloading live in `memory.py` instead.
    """

    def test_save_from_pretrained(self, tmp_path, base_pipe_output):
        base_pipe = self.get_pipeline().to(torch_device)
        base_pipe.save_pretrained(str(tmp_path))

        pipe = ModularPipeline.from_pretrained(tmp_path)
        pipe.load_components(dtype=torch.float32)
        pipe.to(torch_device)

        image = pipe(**self.get_dummy_inputs(), output=self.output_name)

        expected_slice = base_pipe_output[0, -3:, -3:, -1].flatten()
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert torch.abs(expected_slice - image_slice).max() < 1e-3

    def test_load_expected_components_from_pretrained(self, tmp_path):
        pipe = self.get_pipeline()
        expected = get_specified_components(self.pretrained_model_name_or_path, cache_dir=tmp_path)
        if not expected:
            pytest.skip("Skipping test as we couldn't fetch the expected components.")

        actual = {
            name
            for name in pipe.components
            if getattr(pipe, name, None) is not None
            and getattr(getattr(pipe, name), "_diffusers_load_id", None) not in (None, "null")
        }
        assert expected == actual, f"Component mismatch: missing={expected - actual}, unexpected={actual - expected}"

    def test_load_expected_components_from_save_pretrained(self, tmp_path):
        pipe = self.get_pipeline()
        save_dir = str(tmp_path / "saved-pipeline")
        pipe.save_pretrained(save_dir)

        expected = get_specified_components(save_dir)
        loaded_pipe = ModularPipeline.from_pretrained(save_dir)
        loaded_pipe.load_components(dtype=torch.float32)

        actual = {
            name
            for name in loaded_pipe.components
            if getattr(loaded_pipe, name, None) is not None
            and getattr(getattr(loaded_pipe, name), "_diffusers_load_id", None) not in (None, "null")
        }
        assert expected == actual, (
            f"Component mismatch after save/load: missing={expected - actual}, unexpected={actual - expected}"
        )

    def test_modular_index_consistency(self, tmp_path):
        pipe = self.get_pipeline()
        components_spec = pipe._component_specs
        components = sorted(components_spec.keys())

        pipe.save_pretrained(str(tmp_path))
        index_file = tmp_path / "modular_model_index.json"
        assert index_file.exists()

        with open(index_file) as f:
            index_contents = json.load(f)

        compulsory_keys = {"_blocks_class_name", "_class_name", "_diffusers_version"}
        for k in compulsory_keys:
            assert k in index_contents

        to_check_attrs = {"pretrained_model_name_or_path", "revision", "subfolder"}
        for component in components:
            spec = components_spec[component]
            for attr in to_check_attrs:
                if getattr(spec, "pretrained_model_name_or_path", None) is not None:
                    for attr in to_check_attrs:
                        assert component in index_contents, f"{component} should be present in index but isn't."
                        attr_value_from_index = index_contents[component][2][attr]
                        assert getattr(spec, attr) == attr_value_from_index

    def test_unload_components(self):
        pipe = ModularPipeline.from_pretrained(self.pretrained_model_name_or_path)
        pipe.load_components()
        name = next(name for name in pipe.pretrained_component_names if pipe.components.get(name) is not None)
        spec_before = pipe._component_specs[name]

        pipe.unload_components(name)
        assert getattr(pipe, name) is None
        # `components` is the mapping most callers iterate over: the entry stays, its value becomes None
        assert pipe.components[name] is None
        # unloading an already unloaded component is a no-op, not an error
        pipe.unload_components(name)
        assert pipe.components[name] is None
        # the spec survives, so the component can be loaded again
        assert pipe._component_specs[name] is spec_before
        pipe.load_components(names=name)
        assert getattr(pipe, name) is not None

        # with a ComponentsManager attached, unloading also removes the component from the manager
        manager = ComponentsManager()
        pipe = ModularPipeline.from_pretrained(self.pretrained_model_name_or_path, components_manager=manager)
        pipe.load_components(names=name)
        assert len(manager._lookup_ids(name=name)) == 1
        pipe.unload_components(name)
        assert getattr(pipe, name) is None
        assert len(manager._lookup_ids(name=name)) == 0

    def test_unload_components_multiple_names(self):
        pipe = ModularPipeline.from_pretrained(self.pretrained_model_name_or_path)
        pipe.load_components()
        names = [name for name in pipe.pretrained_component_names if pipe.components.get(name) is not None]
        if len(names) < 2:
            pytest.skip("Skipping test as the pipeline has fewer than two loaded pretrained components.")

        pipe.unload_components(names)
        assert all(pipe.components[name] is None for name in names)

        pipe.load_components(names=names)
        assert all(pipe.components[name] is not None for name in names)

    def test_unload_components_invalid_names(self):
        pipe = ModularPipeline.from_pretrained(self.pretrained_model_name_or_path)
        pipe.load_components()
        name = next(name for name in pipe.pretrained_component_names if pipe.components.get(name) is not None)

        with pytest.raises(ValueError, match="Invalid type for names"):
            pipe.unload_components((name,))
        assert pipe.components[name] is not None

        # an unknown name is warned about and skipped; the known names are still unloaded
        logger = logging.get_logger("diffusers.modular_pipelines.modular_pipeline")
        logger.setLevel(diffusers.logging.WARNING)
        with CaptureLogger(logger) as cap_logger:
            pipe.unload_components([name, "not_a_component"])

        assert "not_a_component" in cap_logger.out
        assert pipe.components[name] is None

    def test_unload_components_releases_component(self):
        pipe = ModularPipeline.from_pretrained(self.pretrained_model_name_or_path)
        pipe.load_components()
        name = next(
            name for name in pipe.pretrained_component_names if isinstance(pipe.components.get(name), torch.nn.Module)
        )

        # a weakref keeps no strong reference, so it goes dead only if nothing in the pipeline holds the
        # component anymore — which is what makes the memory actually reclaimable
        component_ref = weakref.ref(pipe.components[name])
        pipe.unload_components(name)

        assert component_ref() is None
