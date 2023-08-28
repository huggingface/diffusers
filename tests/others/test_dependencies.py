# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import unittest
from importlib import import_module


class DependencyTester(unittest.TestCase):
    def test_diffusers_import(self):
        try:
            import diffusers  # noqa: F401
        except ImportError:
            assert False

    def test_backend_registration(self):
        import diffusers
        from diffusers.dependency_versions_table import deps

        all_classes = inspect.getmembers(diffusers, inspect.isclass)

        for cls_name, cls_module in all_classes:
            if "dummy_" in cls_module.__module__:
                for backend in cls_module._backends:
                    if backend == "k_diffusion":
                        backend = "k-diffusion"
                    elif backend == "invisible_watermark":
                        backend = "invisible-watermark"
                    assert backend in deps, f"{backend} is not in the deps table!"

    def test_pipeline_imports(self):
        import diffusers
        import diffusers.pipelines

        all_classes = inspect.getmembers(diffusers, inspect.isclass)
        for cls_name, cls_module in all_classes:
            if hasattr(diffusers.pipelines, cls_name):
                pipeline_folder_module = ".".join(str(cls_module.__module__).split(".")[:3])
                _ = import_module(pipeline_folder_module, str(cls_name))
