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
from pathlib import Path

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
        """Import every pipeline submodule whose folder-level dependency guards
        are satisfied, to catch unguarded optional-dep imports.

        Each pipeline folder's __init__.py evaluates guards like
        is_torch_available() and populates _import_structure only for submodules
        whose deps are met. We use _import_structure as the source of truth:
        if a submodule is listed there, its declared deps are installed, so any
        ImportError from importing it is a real bug (e.g., unguarded torchvision).
        """
        import diffusers.pipelines

        pipelines_dir = Path(diffusers.pipelines.__file__).parent
        failures = []

        for subdir in sorted(pipelines_dir.iterdir()):
            if not subdir.is_dir() or not (subdir / "__init__.py").exists():
                continue

            # Import the pipeline package to trigger its guard evaluation
            package_module_path = f"diffusers.pipelines.{subdir.name}"
            try:
                package_module = import_module(package_module_path)
            except Exception:
                continue

            # _import_structure keys are the submodules whose deps are satisfied
            import_structure = getattr(package_module, "_import_structure", {})

            for submodule_name in import_structure:
                full_module_path = f"{package_module_path}.{submodule_name}"
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
