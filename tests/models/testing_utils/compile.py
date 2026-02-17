# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

import gc
import os

import pytest
import torch

from ...testing_utils import (
    backend_empty_cache,
    is_torch_compile,
    require_accelerator,
    require_torch_version_greater,
    torch_device,
)


@is_torch_compile
@require_accelerator
@require_torch_version_greater("2.7.1")
class TorchCompileTesterMixin:
    """
    Mixin class for testing torch.compile functionality on models.

    Expected from config mixin:
        - model_class: The model class to test

    Optional properties:
        - different_shapes_for_compilation: List of (height, width) tuples for dynamic shape testing (default: None)

    Expected methods from config mixin:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Pytest mark: compile
        Use `pytest -m "not compile"` to skip these tests
    """

    @property
    def different_shapes_for_compilation(self) -> list[tuple[int, int]] | None:
        """Optional list of (height, width) tuples for dynamic shape testing."""
        return None

    def setup_method(self):
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)

    @torch.no_grad()
    def test_torch_compile_recompilation_and_graph_break(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()

        model = self.model_class(**init_dict).to(torch_device)
        model.eval()
        model = torch.compile(model, fullgraph=True)

        with (
            torch._inductor.utils.fresh_inductor_cache(),
            torch._dynamo.config.patch(error_on_recompile=True),
        ):
            _ = model(**inputs_dict)
            _ = model(**inputs_dict)

    @torch.no_grad()
    def test_torch_compile_repeated_blocks(self, recompile_limit=1):
        if self.model_class._repeated_blocks is None:
            pytest.skip("Skipping test as the model class doesn't have `_repeated_blocks` set.")

        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()

        model = self.model_class(**init_dict).to(torch_device)
        model.eval()
        model.compile_repeated_blocks(fullgraph=True)

        if self.model_class.__name__ == "UNet2DConditionModel":
            recompile_limit = 2

        with (
            torch._inductor.utils.fresh_inductor_cache(),
            torch._dynamo.config.patch(recompile_limit=recompile_limit),
        ):
            _ = model(**inputs_dict)
            _ = model(**inputs_dict)

    @torch.no_grad()
    def test_compile_with_group_offloading(self):
        if not self.model_class._supports_group_offloading:
            pytest.skip("Model does not support group offloading.")

        torch._dynamo.config.cache_size_limit = 10000

        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict)
        model.eval()

        group_offload_kwargs = {
            "onload_device": torch_device,
            "offload_device": "cpu",
            "offload_type": "block_level",
            "num_blocks_per_group": 1,
            "use_stream": True,
            "non_blocking": True,
        }
        model.enable_group_offload(**group_offload_kwargs)
        model.compile()

        _ = model(**inputs_dict)
        _ = model(**inputs_dict)

    @torch.no_grad()
    def test_compile_on_different_shapes(self):
        if self.different_shapes_for_compilation is None:
            pytest.skip(f"Skipping as `different_shapes_for_compilation` is not set for {self.__class__.__name__}.")
        torch.fx.experimental._config.use_duck_shape = False

        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)
        model.eval()
        model = torch.compile(model, fullgraph=True, dynamic=True)

        for height, width in self.different_shapes_for_compilation:
            with torch._dynamo.config.patch(error_on_recompile=True):
                inputs_dict = self.get_dummy_inputs(height=height, width=width)
                _ = model(**inputs_dict)

    @torch.no_grad()
    def test_compile_works_with_aot(self, tmp_path):
        from torch._inductor.package import load_package

        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()

        model = self.model_class(**init_dict).to(torch_device)
        exported_model = torch.export.export(model, args=(), kwargs=inputs_dict)

        package_path = os.path.join(str(tmp_path), f"{self.model_class.__name__}.pt2")
        _ = torch._inductor.aoti_compile_and_package(exported_model, package_path=package_path)
        assert os.path.exists(package_path), f"Package file not created at {package_path}"
        loaded_binary = load_package(package_path, run_single_threaded=True)

        model.forward = loaded_binary

        _ = model(**inputs_dict)
        _ = model(**inputs_dict)
