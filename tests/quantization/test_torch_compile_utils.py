# coding=utf-8
# Copyright 2025 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc

import pytest
import torch

from diffusers import DiffusionPipeline

from ..testing_utils import backend_empty_cache, require_torch_accelerator, slow, torch_device


@require_torch_accelerator
@slow
class QuantCompileTests:
    @property
    def quantization_config(self):
        raise NotImplementedError(
            "This property should be implemented in the subclass to return the appropriate quantization config."
        )

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        torch.compiler.reset()
        yield
        gc.collect()
        backend_empty_cache(torch_device)
        torch.compiler.reset()

    def _init_pipeline(self, quantization_config, torch_dtype):
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
        )
        return pipe

    def _test_torch_compile_with_cpu_offload(self, torch_dtype=torch.bfloat16):
        pipe = self._init_pipeline(self.quantization_config, torch_dtype)
        pipe.enable_model_cpu_offload()
        # regional compilation is better for offloading.
        # see: https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/
        if getattr(pipe.transformer, "_repeated_blocks"):
            pipe.transformer.compile_repeated_blocks(fullgraph=True)
        else:
            pipe.transformer.compile()

        # small resolutions to ensure speedy execution.
        pipe("a dog", num_inference_steps=2, max_sequence_length=16, height=256, width=256)

    def test_torch_compile_with_cpu_offload(self):
        self._test_torch_compile_with_cpu_offload()
