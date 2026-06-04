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

import inspect
from importlib import import_module

import pytest


class TestDependencies:
    def test_diffusers_import(self):
        import diffusers  # noqa: F401

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
                    elif backend == "opencv":
                        backend = "opencv-python"
                    elif backend == "nvidia_modelopt":
                        backend = "nvidia_modelopt[hf]"
                    assert backend in deps, f"{backend} is not in the deps table!"

    def test_pipeline_imports(self):
        import diffusers
        import diffusers.pipelines

        all_classes = inspect.getmembers(diffusers, inspect.isclass)
        for cls_name, cls_module in all_classes:
            if hasattr(diffusers.pipelines, cls_name):
                pipeline_folder_module = ".".join(str(cls_module.__module__).split(".")[:3])
                _ = import_module(pipeline_folder_module, str(cls_name))

    def test_pipeline_module_imports(self):
        """Import every pipeline submodule whose dependencies are satisfied,
        to catch unguarded optional-dep imports (e.g., torchvision).

        Uses inspect.getmembers to discover classes that the lazy loader can
        actually resolve (same self-filtering as test_pipeline_imports), then
        imports the full module path instead of truncating to the folder level.
        """
        import diffusers
        import diffusers.pipelines

        failures = []
        all_classes = inspect.getmembers(diffusers, inspect.isclass)

        for cls_name, cls_module in all_classes:
            if not hasattr(diffusers.pipelines, cls_name):
                continue
            if "dummy_" in cls_module.__module__:
                continue

            full_module_path = cls_module.__module__
            try:
                import_module(full_module_path)
            except ImportError as e:
                failures.append(f"{full_module_path}: {e}")
            except Exception:
                # Non-import errors (e.g., missing config) are fine; we only
                # care about unguarded import statements.
                pass

        if failures:
            pytest.fail("Unguarded optional-dependency imports found:\n" + "\n".join(failures))
