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

import warnings

import pytest

import diffusers


TINY_PIPELINE_CLASSES = [
    "LTXLatentUpsamplePipeline",
    "LTX2Pipeline",
    "AudioLDM2Pipeline",
    "StableAudioPipeline",
    "ShapEPipeline",
    "AceStepPipeline",
    "IFPipeline",
]


@pytest.mark.parametrize("pipeline_class_name", TINY_PIPELINE_CLASSES)
def test_tiny_pipeline_loads_without_relocation_warning(pipeline_class_name):
    """
    Loading a pipeline from a pretrained checkpoint must not trigger any relocation-shim deprecation.
    """
    pipeline_class = getattr(diffusers, pipeline_class_name)
    repo_id = f"hf-internal-testing/tiny-{pipeline_class_name}"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline_class.from_pretrained(repo_id)

    relocation_warnings = [
        w
        for w in caught
        if issubclass(w.category, FutureWarning)
        and "Importing " in str(w.message)
        and "diffusers.pipelines." in str(w.message)
        and "is deprecated" in str(w.message)
    ]
    assert not relocation_warnings, (
        f"Loading {pipeline_class_name} from {repo_id} triggered relocation shim FutureWarning(s):\n"
        + "\n".join(f"  - {w.message}" for w in relocation_warnings)
    )
