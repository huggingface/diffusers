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
import inspect

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

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)
        torch.compiler.reset()

    def tearDown(self):
        super().tearDown()
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

    def _test_torch_compile(self, torch_dtype=torch.bfloat16):
        pipe = self._init_pipeline(self.quantization_config, torch_dtype).to(torch_device)
        # `fullgraph=True` ensures no graph breaks
        pipe.transformer.compile(fullgraph=True)

        # small resolutions to ensure speedy execution.
        with torch._dynamo.config.patch(error_on_recompile=True):
            pipe("a dog", num_inference_steps=2, max_sequence_length=16, height=256, width=256)

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

    def _test_torch_compile_with_group_offload_leaf(self, torch_dtype=torch.bfloat16, *, use_stream: bool = False):
        torch._dynamo.config.cache_size_limit = 1000

        pipe = self._init_pipeline(self.quantization_config, torch_dtype)
        group_offload_kwargs = {
            "onload_device": torch.device(torch_device),
            "offload_device": torch.device("cpu"),
            "offload_type": "leaf_level",
            "use_stream": use_stream,
        }
        pipe.transformer.enable_group_offload(**group_offload_kwargs)
        pipe.transformer.compile()
        for name, component in pipe.components.items():
            if name != "transformer" and isinstance(component, torch.nn.Module):
                if torch.device(component.device).type == "cpu":
                    component.to(torch_device)

        # small resolutions to ensure speedy execution.
        pipe("a dog", num_inference_steps=2, max_sequence_length=16, height=256, width=256)

    def test_torch_compile(self):
        self._test_torch_compile()

    def test_torch_compile_with_cpu_offload(self):
        self._test_torch_compile_with_cpu_offload()

    def test_torch_compile_with_group_offload_leaf(self, use_stream=False):
        for cls in inspect.getmro(self.__class__):
            if "test_torch_compile_with_group_offload_leaf" in cls.__dict__ and cls is not QuantCompileTests:
                return
        self._test_torch_compile_with_group_offload_leaf(use_stream=use_stream)
