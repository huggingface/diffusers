# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
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
"""Tests for ``DreamLiteMobilePipeline``.

The mobile pipeline is a distilled, no-CFG sibling of ``DreamLitePipeline``.
It runs a single UNet forward per step (no 3-way concat) and ignores
``guidance_scale`` / ``image_guidance_scale``. Test layout mirrors
``test_pipeline_dreamlite.py``; the shared tiny Qwen3-VL test fixture lives in
``testing_utils.py``.
"""

import gc
import os

import numpy as np
import pytest
import torch
from PIL import Image

from diffusers import DreamLiteMobilePipeline

from ...testing_utils import (
    enable_full_determinism,
    nightly,
    require_torch_gpu,
    torch_device,
)
from ..testing_utils import MemoryTesterMixin, PipelineTesterMixin
from .testing_utils import DreamLiteBaseTesterConfig


enable_full_determinism()


class DreamLiteMobilePipelineTesterConfig(DreamLiteBaseTesterConfig):
    pipeline_class = DreamLiteMobilePipeline
    required_input_params_in_call_signature = frozenset(
        [
            "prompt",
            "height",
            "width",
            "num_inference_steps",
        ]
    )
    batch_input_params = frozenset(["prompt"])

    def get_dummy_inputs(self):
        return {
            "prompt": "a small dog",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "height": 64,
            "width": 64,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }

    def get_dummy_i2i_inputs(self, seed=0):
        inputs = self.get_dummy_inputs()
        inputs["image"] = Image.fromarray((np.random.RandomState(seed).rand(64, 64, 3) * 255).astype(np.uint8))
        return inputs


class TestDreamLiteMobilePipeline(DreamLiteMobilePipelineTesterConfig, PipelineTesterMixin):
    # ---- skips for mixin tests that genuinely don't apply ----------------
    # The remaining skips are intrinsic to the mobile pipeline's design:
    #   * ``encode_prompt`` returns ``(prompt_embeds, prompt_embeds_mask)``;
    #   * the pipeline forces ``batch_size = 1`` internally.
    @pytest.mark.skip(
        "DreamLiteMobile encode_prompt returns (embeds, mask) tuple, not a single tensor; "
        "the mixin's test_encode_prompt_works_in_isolation assumes single tensor return."
    )
    def test_encode_prompt_works_in_isolation(self):
        pass

    @pytest.mark.skip(
        "DreamLiteMobile intentionally limits ``batch_size`` to 1; only ``num_images_per_prompt > 1`` is supported."
    )
    def test_num_images_per_prompt(self):
        pass

    @pytest.mark.skip("DreamLiteMobile forces batch_size=1 internally.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip("DreamLiteMobile forces batch_size=1 internally.")
    def test_inference_batch_single_identical(self):
        pass

    # ---- actual tests ------------------------------------------------------
    def test_mobile_t2i_default_case(self):
        pipe = self.get_pipeline().to(torch_device)

        out = pipe(**self.get_dummy_inputs()).images

        assert out.shape == (1, *self.output_shape)
        assert not torch.isnan(out).any()

    def test_mobile_i2i_default_case(self):
        pipe = self.get_pipeline().to(torch_device)

        out = pipe(**self.get_dummy_i2i_inputs()).images

        assert out.shape == (1, *self.output_shape)
        assert not torch.isnan(out).any()

    def test_mobile_single_forward_per_step(self):
        """Mobile pipeline must run exactly ONE UNet forward per step (no CFG concat)."""
        pipe = self.get_pipeline().to(torch_device)

        original_forward = pipe.unet.forward
        seen_batches = []

        def spy_forward(*args, **kwargs):
            x = args[0] if args else kwargs["sample"]
            seen_batches.append(x.shape[0])
            return original_forward(*args, **kwargs)

        pipe.unet.forward = spy_forward
        inputs = self.get_dummy_i2i_inputs()
        inputs["num_inference_steps"] = 2
        pipe(**inputs)

        assert all(b == 1 for b in seen_batches), f"expected all 1-way, got {seen_batches}"
        assert len(seen_batches) == 2, "expected exactly 2 unet calls (1 per step)"

    def test_mobile_guidance_scale_ignored(self):
        """Passing guidance_scale to the mobile pipeline should be accepted but ignored (with warning)."""
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        inputs["guidance_scale"] = 7.5  # should not raise
        inputs["image_guidance_scale"] = 1.5  # should not raise
        out = pipe(**inputs).images

        assert out.shape == (1, *self.output_shape)


class TestDreamLiteMobilePipelineMemory(DreamLiteMobilePipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the mobile pipeline."""


@nightly
@require_torch_gpu
class TestDreamLiteMobilePipelineIntegration:
    """End-to-end test against the real DreamLite-mobile checkpoint on the Hub.

    By default this loads ``carlofkl/DreamLite-mobile`` (``diffusers`` branch)
    from the HF Hub. To run against a local copy during development, set the
    ``DREAMLITE_MOBILE_PATH`` env var to that path.
    """

    repo_id = "carlofkl/DreamLite-mobile"
    revision = "diffusers"

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        torch.cuda.empty_cache()
        yield
        gc.collect()
        torch.cuda.empty_cache()

    def _from_pretrained_kwargs(self):
        local = os.getenv("DREAMLITE_MOBILE_PATH")
        if local:
            return {"pretrained_model_name_or_path": local}
        return {"pretrained_model_name_or_path": self.repo_id, "revision": self.revision}

    def test_mobile_t2i_real_checkpoint(self):
        pipe = DreamLiteMobilePipeline.from_pretrained(
            **self._from_pretrained_kwargs(), torch_dtype=torch.bfloat16
        ).to("cuda")
        out = pipe(
            prompt="a dog running on the grass",
            num_inference_steps=4,
            height=1024,
            width=1024,
            generator=torch.Generator("cpu").manual_seed(0),
            output_type="np",
        ).images

        assert out.shape == (1, 1024, 1024, 3)
        assert not np.isnan(out).any()

    def test_mobile_i2i_real_checkpoint(self):
        pipe = DreamLiteMobilePipeline.from_pretrained(
            **self._from_pretrained_kwargs(), torch_dtype=torch.bfloat16
        ).to("cuda")

        src = Image.fromarray((np.random.RandomState(0).rand(1024, 1024, 3) * 255).astype(np.uint8))
        out = pipe(
            prompt="make it look like a painting",
            image=src,
            num_inference_steps=4,
            height=1024,
            width=1024,
            generator=torch.Generator("cpu").manual_seed(0),
            output_type="np",
        ).images

        assert out.shape == (1, 1024, 1024, 3)
        assert not np.isnan(out).any()
