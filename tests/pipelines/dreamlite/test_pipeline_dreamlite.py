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
"""Tests for ``DreamLitePipeline``.

Test design
-----------
``DreamLitePipeline`` depends on Qwen3-VL as its text/image encoder. For the
fast tests we instantiate a real ``Qwen3VLForConditionalGeneration`` from a
tiny config (mirroring the NucleusMoE-Image fast tests), and load the matching
processor / tokenizer from the public ``hf-internal-testing`` mirror, so that
the standard ``PipelineTesterMixin`` save/load and dtype/device tests work
out of the box. The shared component set lives in ``testing_utils.py``.

For end-to-end verification against the original repo, see the
``parity_run_*.py`` scripts shipped with the integration.
"""

import gc
import os

import numpy as np
import pytest
import torch
from PIL import Image

from diffusers import DreamLitePipeline, DreamLiteUNetModel

from ...testing_utils import (
    enable_full_determinism,
    nightly,
    require_torch_gpu,
    torch_device,
)
from ..testing_utils import MemoryTesterMixin, PipelineTesterMixin
from .testing_utils import CROSS_ATTN_DIM, DreamLiteBaseTesterConfig


enable_full_determinism()


class DreamLitePipelineTesterConfig(DreamLiteBaseTesterConfig):
    pipeline_class = DreamLitePipeline
    required_input_params_in_call_signature = frozenset(
        [
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "num_inference_steps",
        ]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])

    def get_dummy_inputs(self):
        return {
            "prompt": "a small dog",
            "negative_prompt": "",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 3.5,
            "height": 64,
            "width": 64,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }

    def get_dummy_i2i_inputs(self, seed=0):
        inputs = self.get_dummy_inputs()
        # 64x64 RGB image -- will be processed by VaeImageProcessor.
        inputs["image"] = Image.fromarray((np.random.RandomState(seed).rand(64, 64, 3) * 255).astype(np.uint8))
        inputs["image_guidance_scale"] = 1.5
        return inputs


class TestDreamLitePipeline(DreamLitePipelineTesterConfig, PipelineTesterMixin):
    # ---- skips for mixin tests that genuinely don't apply ----------------
    # The remaining skips reflect intrinsic design choices of the DreamLite pipeline:
    #   * ``encode_prompt`` returns a ``(prompt_embeds, prompt_embeds_mask)``
    #     tuple, while the mixin's ``test_encode_prompt_works_in_isolation``
    #     assumes a single tensor return value;
    #   * the pipeline forces ``batch_size = 1`` internally, so the mixin's
    #     batch sweep cannot apply.
    @pytest.mark.skip(
        "DreamLite intentionally limits ``batch_size`` to 1 (CFG memory blow-up); "
        "only ``num_images_per_prompt > 1`` is supported."
    )
    def test_num_images_per_prompt(self):
        pass

    @pytest.mark.skip(
        "DreamLite encode_prompt returns (embeds, mask) tuple, not a single tensor; "
        "the mixin's test_encode_prompt_works_in_isolation assumes single tensor return."
    )
    def test_encode_prompt_works_in_isolation(self):
        pass

    @pytest.mark.skip("DreamLite forces batch_size=1 internally.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip("DreamLite forces batch_size=1 internally.")
    def test_inference_batch_single_identical(self):
        pass

    # ---- actual tests ------------------------------------------------------
    def test_legacy_block_type_aliases(self):
        unet = DreamLiteUNetModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            down_block_types=(
                "CrossAttnDownRemoveSelfAttnBlock2D",
                "CrossAttnDownRemoveSelfAttnBlock2D",
                "CrossAttnDownBlock2D",
            ),
            mid_block_type="UNetMidBlock2DCrossAttn",
            up_block_types=(
                "CrossAttnUpBlock2D",
                "CrossAttnUpRemoveSelfAttnBlock2DV1",
                "UpBlock2D",
            ),
            block_out_channels=(16, 32, 64),
            cross_attention_dim=CROSS_ATTN_DIM,
            attention_head_dim=8,
            layers_per_block=1,
            norm_num_groups=8,
            transformer_layers_per_block=1,
        )

        assert [block.__class__.__name__ for block in unet.down_blocks] == [
            "DreamLiteCrossAttnNoSelfAttnDownBlock2D",
            "DreamLiteCrossAttnNoSelfAttnDownBlock2D",
            "DreamLiteCrossAttnDownBlock2D",
        ]
        assert unet.mid_block.__class__.__name__ == "DreamLiteUNetMidBlock2DCrossAttn"
        assert [block.__class__.__name__ for block in unet.up_blocks] == [
            "DreamLiteCrossAttnUpBlock2D",
            "DreamLiteCrossAttnNoSelfAttnUpBlock2D",
            "DreamLiteUpBlock2D",
        ]

        unet_with_non_v1_up_alias = DreamLiteUNetModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            down_block_types=(
                "CrossAttnDownRemoveSelfAttnBlock2D",
                "CrossAttnDownRemoveSelfAttnBlock2D",
                "CrossAttnDownBlock2D",
            ),
            mid_block_type="UNetMidBlock2DCrossAttn",
            up_block_types=(
                "CrossAttnUpBlock2D",
                "CrossAttnUpRemoveSelfAttnBlock2D",
                "UpBlock2D",
            ),
            block_out_channels=(16, 32, 64),
            cross_attention_dim=CROSS_ATTN_DIM,
            attention_head_dim=8,
            layers_per_block=1,
            norm_num_groups=8,
            transformer_layers_per_block=1,
        )
        assert [block.__class__.__name__ for block in unet_with_non_v1_up_alias.up_blocks] == [
            "DreamLiteCrossAttnUpBlock2D",
            "DreamLiteCrossAttnNoSelfAttnUpBlock2D",
            "DreamLiteUpBlock2D",
        ]

    def test_dreamlite_t2i_default_case(self):
        pipe = self.get_pipeline().to(torch_device)

        out = pipe(**self.get_dummy_inputs()).images

        assert out.shape == (1, *self.output_shape)
        assert not torch.isnan(out).any()

    def test_dreamlite_i2i_default_case(self):
        pipe = self.get_pipeline().to(torch_device)

        out = pipe(**self.get_dummy_i2i_inputs()).images

        assert out.shape == (1, *self.output_shape)
        assert not torch.isnan(out).any()

    def test_dreamlite_cfg_branch_count(self):
        """In edit mode the pipeline must run a 3-way CFG concat (uncond/img/text)."""
        pipe = self.get_pipeline().to(torch_device)

        original_forward = pipe.unet.forward
        seen_batches = []

        def spy_forward(*args, **kwargs):
            x = args[0] if args else kwargs["sample"]
            seen_batches.append(x.shape[0])
            return original_forward(*args, **kwargs)

        pipe.unet.forward = spy_forward
        inputs = self.get_dummy_i2i_inputs()
        inputs["num_inference_steps"] = 1
        pipe(**inputs)

        assert all(b == 3 for b in seen_batches), f"expected all 3-way, got {seen_batches}"


class TestDreamLitePipelineMemory(DreamLitePipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the DreamLite pipeline."""


@nightly
@require_torch_gpu
class TestDreamLitePipelineIntegration:
    """End-to-end test against the real DreamLite-base checkpoint on the Hub.

    By default this loads ``carlofkl/DreamLite-base`` (``diffusers`` branch)
    from the HF Hub. To run against a local copy during development, set the
    ``DREAMLITE_BASE_PATH`` env var to that path.
    """

    repo_id = "carlofkl/DreamLite-base"
    revision = "diffusers"

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        torch.cuda.empty_cache()
        yield
        gc.collect()
        torch.cuda.empty_cache()

    def _from_pretrained_kwargs(self):
        local = os.getenv("DREAMLITE_BASE_PATH")
        if local:
            return {"pretrained_model_name_or_path": local}
        return {"pretrained_model_name_or_path": self.repo_id, "revision": self.revision}

    def test_dreamlite_t2i_real_checkpoint(self):
        pipe = DreamLitePipeline.from_pretrained(**self._from_pretrained_kwargs(), torch_dtype=torch.bfloat16).to(
            "cuda"
        )
        out = pipe(
            prompt="a dog running on the grass",
            num_inference_steps=2,
            guidance_scale=3.5,
            height=1024,
            width=1024,
            generator=torch.Generator("cpu").manual_seed(0),
            output_type="np",
        ).images

        assert out.shape == (1, 1024, 1024, 3)
        assert not np.isnan(out).any()

    def test_dreamlite_i2i_real_checkpoint(self):
        pipe = DreamLitePipeline.from_pretrained(**self._from_pretrained_kwargs(), torch_dtype=torch.bfloat16).to(
            "cuda"
        )

        src = Image.fromarray((np.random.RandomState(0).rand(1024, 1024, 3) * 255).astype(np.uint8))
        out = pipe(
            prompt="make it look like a painting",
            image=src,
            num_inference_steps=2,
            guidance_scale=3.5,
            image_guidance_scale=1.5,
            height=1024,
            width=1024,
            generator=torch.Generator("cpu").manual_seed(0),
            output_type="np",
        ).images

        assert out.shape == (1, 1024, 1024, 3)
        assert not np.isnan(out).any()
