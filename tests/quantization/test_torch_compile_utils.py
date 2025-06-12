# coding=utf-8
# Copyright 2024 The HuggingFace Team Inc.
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
import unittest

import torch

from diffusers import DiffusionPipeline
from diffusers.utils.testing_utils import backend_empty_cache, require_torch_gpu, slow, torch_device


@require_torch_gpu
@slow
class QuantCompileTests(unittest.TestCase):
    quantization_config = None

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

    def _test_torch_compile(self, quantization_config, torch_dtype=torch.bfloat16):
        pipe = self._init_pipeline(quantization_config, torch_dtype).to("cuda")
        # import to ensure fullgraph True
        pipe.transformer.compile(fullgraph=True)

        for _ in range(2):
            # small resolutions to ensure speedy execution.
            pipe("a dog", num_inference_steps=3, max_sequence_length=16, height=256, width=256)

    def _test_torch_compile_with_cpu_offload(self, quantization_config, torch_dtype=torch.bfloat16):
        pipe = self._init_pipeline(quantization_config, torch_dtype)
        pipe.enable_model_cpu_offload()
        pipe.transformer.compile()

        for _ in range(2):
            # small resolutions to ensure speedy execution.
            pipe("a dog", num_inference_steps=3, max_sequence_length=16, height=256, width=256)

    def _test_torch_compile_with_group_offload(self, quantization_config, torch_dtype=torch.bfloat16):
        torch._dynamo.config.cache_size_limit = 10000

        pipe = self._init_pipeline(quantization_config, torch_dtype)
        group_offload_kwargs = {
            "onload_device": torch.device("cuda"),
            "offload_device": torch.device("cpu"),
            "offload_type": "leaf_level",
            "use_stream": True,
            "non_blocking": True,
        }
        pipe.transformer.enable_group_offload(**group_offload_kwargs)
        pipe.transformer.compile()
        for name, component in pipe.components.items():
            if name != "transformer" and isinstance(component, torch.nn.Module):
                if torch.device(component.device).type == "cpu":
                    component.to("cuda")

        for _ in range(2):
            # small resolutions to ensure speedy execution.
            pipe("a dog", num_inference_steps=3, max_sequence_length=16, height=256, width=256)
