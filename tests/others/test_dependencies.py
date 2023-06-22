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

import unittest


class DependencyTester(unittest.TestCase):
    # We include all the soft dependencies listed in setup.py except `pytest`.
    # We need `pytest` to test this utility. `urllib3` is also excluded because it's needed
    # by `requests`.
    soft_dependencies = [
        "tensorboard",
        "parameterized",
        "requests-mock",
        "omegaconf",
        "torchvision",
        "datasets",
        "black",
        "ruff",
        "flax",
        "protobuf",
        "pytest-timeout",
        "k-diffusion",
        "sentencepiece",
        "hf-doc-builder",
        "scipy",
        "pytest-xdist",
        "transformers",
        "jax",
        "accelerate",
        "torch",
        "safetensors",
        "jaxlib",
        "librosa",
        "compel",
        "isort",
        "Jinja2",
    ]

    def test_soft_dependencies_no_installed(self):
        for soft_dep in self.soft_dependencies:
            with self.subTest(dependency=soft_dep):
                try:
                    __import__(soft_dep)
                    self.fail(f"Imported {soft_dep} successfully")
                except ImportError:
                    assert True
