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
import gc
import threading
import unittest

import torch

from diffusers import UNet2DConditionModel

from ..testing_utils import require_peft_backend


def _build_lora_state_dict(unet, rank=4):
    """Build a minimal LoRA state dict for the attention linear layers of the given UNet."""
    state_dict = {}
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Linear) and (
            name.endswith("to_q") or name.endswith("to_k") or name.endswith("to_v") or name.endswith("to_out.0")
        ):
            state_dict[f"{name}.lora_A.weight"] = torch.randn(rank, module.in_features)
            state_dict[f"{name}.lora_B.weight"] = torch.randn(module.out_features, rank)
    return state_dict


@require_peft_backend
class LowCpuMemUsageThreadSafetyTests(unittest.TestCase):
    unet_kwargs = {
        "block_out_channels": (32, 64),
        "layers_per_block": 2,
        "sample_size": 32,
        "in_channels": 4,
        "out_channels": 4,
        "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D"),
        "up_block_types": ("CrossAttnUpBlock2D", "UpBlock2D"),
        "cross_attention_dim": 32,
    }

    num_threads = 4
    num_rounds = 10

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torch.manual_seed(0)
        probe_unet = UNet2DConditionModel(**cls.unet_kwargs)
        cls.lora_state_dict = _build_lora_state_dict(probe_unet)
        del probe_unet
        gc.collect()

    def _run_concurrent_injections(self, low_cpu_mem_usage):
        original = torch.nn.Module.register_parameter
        errors = []
        try:
            barrier = threading.Barrier(self.num_threads)

            def worker(thread_id):
                for _ in range(self.num_rounds):
                    try:
                        barrier.wait()
                    except threading.BrokenBarrierError:
                        return
                    try:
                        unet = UNet2DConditionModel(**self.unet_kwargs)
                        unet.load_lora_adapter(
                            self.lora_state_dict,
                            adapter_name=f"adapter-{thread_id}",
                            low_cpu_mem_usage=low_cpu_mem_usage,
                            prefix=None,
                        )
                    except Exception as e:
                        errors.append((thread_id, type(e).__name__, str(e)))
                        return

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(self.num_threads)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            return {
                "register_parameter_patched": torch.nn.Module.register_parameter is not original,
                "fresh_linear_device": torch.nn.Linear(2, 2).weight.device.type,
                "thread_errors": errors,
            }
        finally:
            torch.nn.Module.register_parameter = original

    def test_concurrent_low_cpu_mem_usage_injection_does_not_leak_register_parameter(self):
        """Concurrent `low_cpu_mem_usage=True` LoRA injection must not leak the global patch."""
        observed = self._run_concurrent_injections(low_cpu_mem_usage=True)
        self.assertFalse(
            observed["register_parameter_patched"],
            f"Concurrent `low_cpu_mem_usage=True` injection leaked the global "
            f"`torch.nn.Module.register_parameter` monkey patch. Observed: {observed}",
        )
        self.assertNotEqual(
            observed["fresh_linear_device"],
            "meta",
            f"Concurrent `low_cpu_mem_usage=True` injection left newly created modules on the meta "
            f"device. Observed: {observed}",
        )
        self.assertEqual(
            observed["thread_errors"],
            [],
            f"Worker threads raised during concurrent `low_cpu_mem_usage=True` injection: {observed['thread_errors']}",
        )

    def test_concurrent_injection_without_low_cpu_mem_usage_is_safe(self):
        """Sanity control: concurrent `low_cpu_mem_usage=False` injection must be safe."""
        observed = self._run_concurrent_injections(low_cpu_mem_usage=False)
        self.assertFalse(
            observed["register_parameter_patched"],
            f"Concurrent `low_cpu_mem_usage=False` injection leaked the global "
            f"`torch.nn.Module.register_parameter` monkey patch. Observed: {observed}",
        )
        self.assertNotEqual(
            observed["fresh_linear_device"],
            "meta",
            f"Concurrent `low_cpu_mem_usage=False` injection left newly created modules on the meta "
            f"device. Observed: {observed}",
        )
        self.assertEqual(
            observed["thread_errors"],
            [],
            f"Worker threads raised during concurrent `low_cpu_mem_usage=False` injection: "
            f"{observed['thread_errors']}",
        )
