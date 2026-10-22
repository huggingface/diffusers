# Copyright 2026 The HuggingFace Team.
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

import pytest
import torch

from .cosmos_guardrail import DummyCosmosSafetyChecker


class CosmosSafetyCheckerTesterMixin:
    """Overrides of the shared pipeline tests that the Cosmos `safety_checker` component gets in the way of.

    Every Cosmos pipeline takes a `safety_checker`, and the fast tests substitute `DummyCosmosSafetyChecker` for the
    real Cosmos Guardrail, which is far too large to build on CI. The dummy is not serialized like the other model
    components and a pipeline constructed without one falls back to the real guardrail, so the tests that save,
    reload or enumerate components need the component handled explicitly.

    Compose it *before* `PipelineTesterMixin` so these overrides win, e.g.
    `class TestCosmosXPipeline(CosmosXPipelineTesterConfig, CosmosSafetyCheckerTesterMixin, PipelineTesterMixin)`.
    """

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=1e-4):
        # `safety_checker` is listed as optional, but a pipeline built without one falls back to the real Cosmos
        # Guardrail — so keep it out of the components the base test nulls.
        self.pipeline_class._optional_components.remove("safety_checker")
        try:
            super().test_save_load_optional_components(tmp_path, expected_max_difference=expected_max_difference)
        finally:
            self.pipeline_class._optional_components.append("safety_checker")

    def test_serialization_with_variants(self, tmp_path):
        # Same as the base test, except `safety_checker` is not serialized like the other model components.
        pipe = self.get_pipeline()
        model_components = [
            component_name
            for component_name, component in pipe.components.items()
            if isinstance(component, torch.nn.Module)
        ]
        model_components.remove("safety_checker")
        variant = "fp16"

        pipe.save_pretrained(tmp_path, variant=variant, safe_serialization=False)

        with open(f"{tmp_path}/model_index.json", "r") as f:
            config = json.load(f)

        for subfolder in os.listdir(tmp_path):
            if not os.path.isfile(subfolder) and subfolder in model_components:
                folder_path = os.path.join(tmp_path, subfolder)
                is_folder = os.path.isdir(folder_path) and subfolder in config
                assert is_folder and any(p.split(".")[1].startswith(variant) for p in os.listdir(folder_path))

    def test_torch_dtype_dict(self, tmp_path):
        # Same as the base test, except the safety checker has to be passed back in on load and is left out of the
        # dtype check (the dummy tracks its dtype through a non-persistent buffer).
        components = self.get_dummy_components()
        pipe = self.get_pipeline(**components)
        specified_key = next(iter(components.keys()))

        pipe.save_pretrained(str(tmp_path), safe_serialization=False)
        torch_dtype_dict = {specified_key: torch.bfloat16, "default": torch.float16}
        loaded_pipe = self.pipeline_class.from_pretrained(
            str(tmp_path), safety_checker=DummyCosmosSafetyChecker(), dtype=torch_dtype_dict
        )

        for name, component in loaded_pipe.components.items():
            if name == "safety_checker":
                continue
            if isinstance(component, torch.nn.Module) and hasattr(component, "dtype"):
                expected_dtype = torch_dtype_dict.get(name, torch_dtype_dict.get("default", torch.float32))
                assert component.dtype == expected_dtype, (
                    f"Component '{name}' has dtype {component.dtype} but expected {expected_dtype}"
                )

    @pytest.mark.skip(
        "The pipeline should not be runnable without a safety checker. The test creates a pipeline without passing in "
        "a safety checker, which makes the pipeline default to the actual Cosmos Guardrail. The Cosmos Guardrail is "
        "too large and slow to run on CI."
    )
    def test_encode_prompt_works_in_isolation(self):
        pass
