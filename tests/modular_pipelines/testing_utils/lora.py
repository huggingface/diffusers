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

from ...pipelines.testing_utils.lora import LoraMemoryTesterMixin, LoraTesterMixin
from .common import BaseModularPipelineOutputMixin


class ModularLoraTesterMixin(BaseModularPipelineOutputMixin, LoraTesterMixin):
    """
    The pipeline-level LoRA tests, run against a modular pipeline.

    A modular pipeline inherits the very same `LoraBaseMixin` subclass a standard one does
    (`MiniMaxH3ModularPipeline` is a `MiniMaxH3LoraLoaderMixin`, and so on), so the tests are the same. Only building
    and calling the pipeline differs, and that is `BaseModularPipelineOutputMixin`, listed first so its
    `get_pipeline`/`run_pipe`/`base_pipe_output` win over the non-modular ones.

    Compose with a `BaseModularPipelineTesterConfig` subclass, and override `denoiser_target_modules` when the
    denoiser components are not named `transformer`.
    """

    @pytest.mark.skip(
        reason="`ModularPipeline.save_pretrained` writes the component index, not the weights, so there is no "
        "pipeline directory to reload an attached adapter from."
    )
    def test_simple_inference_save_pretrained_with_text_lora(self):
        pass


class ModularLoraMemoryTesterMixin(BaseModularPipelineOutputMixin, LoraMemoryTesterMixin):
    """LoRA x offloading tests for modular pipelines: group offloading composed with `load_lora_weights`."""

    @pytest.mark.skip(
        reason="`ModularPipeline` has no `enable_model_cpu_offload`; a modular pipeline offloads through "
        "`ComponentsManager.enable_auto_cpu_offload`, which gets a LoRA test of its own."
    )
    def test_lora_loading_model_cpu_offload(self):
        pass
