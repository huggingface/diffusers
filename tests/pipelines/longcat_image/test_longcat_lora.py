# Copyright 2025 The HuggingFace Team.
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

import numpy as np
import torch

from diffusers import LongCatImagePipeline

from ...testing_utils import enable_full_determinism, require_accelerate, require_torch_gpu, slow


enable_full_determinism()


def _pil_to_np01(img):
    """PIL -> float32 in [0, 1], shape (H, W, 3)."""
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    return arr


class LongCatImagePipelineLoRATests(unittest.TestCase):
    @slow
    @require_torch_gpu
    @require_accelerate
    def test_lora_load_changes_output_and_unload_restores(self):
        """
        1) Generate baseline image
        2) Load LoRA -> output should change
        3) Unload LoRA -> output should return close to baseline
        """
        model_id = "meituan-longcat/LongCat-Image"
        lora_repo = "lrzjason/LongCatEmojiTest"
        weight_name = "longcat_image-9-450.safetensors"
        adapter_name = "emoji"

        pipe = LongCatImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)

        prompt = "a 3d anime character, cute emoji style, studio lighting"

        common_kwargs = {
            "height": 768,
            "width": 1344,
            "guidance_scale": 4.0,
            "num_inference_steps": 8,
            "num_images_per_prompt": 1,
            "output_type": "pil",
        }

        # 1) Baseline (no LoRA)
        g0 = torch.Generator(device="cpu").manual_seed(123)
        base_img = pipe(prompt, generator=g0, **common_kwargs).images[0]

        # 2) Load LoRA
        pipe.load_lora_weights(
            lora_repo,
            weight_name=weight_name,
            adapter_name=adapter_name,
        )

        g1 = torch.Generator(device="cpu").manual_seed(123)
        lora_img = pipe(prompt, generator=g1, **common_kwargs).images[0]

        # 3) Unload LoRA
        pipe.unload_lora_weights()

        g2 = torch.Generator(device="cpu").manual_seed(123)
        after_img = pipe(prompt, generator=g2, **common_kwargs).images[0]

        base = _pil_to_np01(base_img)
        lora = _pil_to_np01(lora_img)
        after = _pil_to_np01(after_img)

        diff_lora = float(np.mean(np.abs(base - lora)))
        diff_after = float(np.mean(np.abs(base - after)))

        self.assertGreater(
            diff_lora,
            1e-4,
            msg=f"LoRA didn't change output enough (mean|base-lora|={diff_lora}).",
        )

        # After unload, output should be substantially closer to base than the LoRA output.
        self.assertLess(
            diff_after,
            diff_lora * 0.5,
            msg=(
                "Unloading LoRA didn't restore base behavior enough "
                f"(mean|base-after|={diff_after}, mean|base-lora|={diff_lora})."
            ),
        )
