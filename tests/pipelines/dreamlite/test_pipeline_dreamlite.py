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
out of the box.

For end-to-end verification against the original repo, see the
``parity_run_*.py`` scripts shipped with the integration.
"""

import gc
import os
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from diffusers import (
    AutoencoderTiny,
    DreamLitePipeline,
    DreamLiteUNetModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    nightly,
    require_torch_gpu,
    torch_device,
)

from ..test_pipelines_common import (
    PipelineTesterMixin,
    to_np,
)


enable_full_determinism()


# Match the tiny text encoder hidden size below; the UNet's cross-attention
# dimension must match what ``encode_prompt`` returns.
_CROSS_ATTN_DIM = 16


def _build_tiny_text_encoder() -> Qwen3VLForConditionalGeneration:
    """Build a tiny but functional Qwen3-VL model for the fast test fixture.

    Mirrors the recipe used by ``tests/pipelines/nucleusmoe_image``: small text
    + vision configs that still go through the real Qwen3-VL forward path, so
    DreamLite's ``encode_prompt`` (chat template + tokenizer + multimodal
    processor) is exercised for real.
    """
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": _CROSS_ATTN_DIM,
            "intermediate_size": _CROSS_ATTN_DIM,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "rope_scaling": {
                "mrope_section": [1, 1, 2],
                "rope_type": "default",
                "type": "default",
            },
            "rope_theta": 1000000.0,
            "vocab_size": 151936,
            "head_dim": 8,
        },
        vision_config={
            "depth": 2,
            "hidden_size": _CROSS_ATTN_DIM,
            "intermediate_size": _CROSS_ATTN_DIM,
            "num_heads": 2,
            "out_channels": _CROSS_ATTN_DIM,
            # ``out_hidden_size`` is the dim that vision tokens are projected to before
            # being merged into the text stream; it must match ``text_config.hidden_size``.
            "out_hidden_size": _CROSS_ATTN_DIM,
            # Match the cached ``hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration``
            # image processor (``patch_size=14``); otherwise the pixel_values
            # produced by the processor cannot be reshaped to the model's
            # vision patch embed.
            "patch_size": 14,
        },
    )
    return Qwen3VLForConditionalGeneration(config).eval()


class DreamLitePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = DreamLitePipeline
    params = frozenset(
        [
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "num_inference_steps",
        ]
    )
    batch_params = frozenset(["prompt", "negative_prompt"])
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "output_type",
            "return_dict",
        ]
    )
    test_xformers_attention = False
    test_attention_slicing = False
    test_layerwise_casting = False
    test_group_offloading = False

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
            cross_attention_dim=_CROSS_ATTN_DIM,
            attention_head_dim=8,
            layers_per_block=1,
            norm_num_groups=8,
            transformer_layers_per_block=1,
        )

        self.assertEqual(
            [block.__class__.__name__ for block in unet.down_blocks],
            [
                "DreamLiteCrossAttnNoSelfAttnDownBlock2D",
                "DreamLiteCrossAttnNoSelfAttnDownBlock2D",
                "DreamLiteCrossAttnDownBlock2D",
            ],
        )
        self.assertEqual(unet.mid_block.__class__.__name__, "DreamLiteUNetMidBlock2DCrossAttn")
        self.assertEqual(
            [block.__class__.__name__ for block in unet.up_blocks],
            [
                "DreamLiteCrossAttnUpBlock2D",
                "DreamLiteCrossAttnNoSelfAttnUpBlock2D",
                "DreamLiteUpBlock2D",
            ],
        )

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = DreamLiteUNetModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            down_block_types=(
                "DreamLiteCrossAttnNoSelfAttnDownBlock2D",
                "DreamLiteCrossAttnDownBlock2D",
            ),
            up_block_types=("DreamLiteCrossAttnUpBlock2D", "DreamLiteUpBlock2D"),
            block_out_channels=(32, 64),
            cross_attention_dim=_CROSS_ATTN_DIM,
            attention_head_dim=8,
            layers_per_block=1,
            norm_num_groups=8,
            transformer_layers_per_block=1,
        )

        torch.manual_seed(0)
        vae = AutoencoderTiny(
            in_channels=3,
            out_channels=3,
            encoder_block_out_channels=(32, 32),
            decoder_block_out_channels=(32, 32),
            num_encoder_blocks=(1, 1),
            num_decoder_blocks=(1, 1),
            latent_channels=4,
        )

        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

        torch.manual_seed(0)
        text_encoder = _build_tiny_text_encoder()
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")
        processor = Qwen3VLProcessor.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

        return {
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "processor": processor,
            "vae": vae,
            "unet": unet,
            "scheduler": scheduler,
        }

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        return {
            "prompt": "a small dog",
            "negative_prompt": "",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 3.5,
            "height": 64,
            "width": 64,
            "max_sequence_length": 16,
            "output_type": "np",
        }

    def get_dummy_i2i_inputs(self, device, seed=0):
        inputs = self.get_dummy_inputs(device, seed)
        # 64x64 RGB image -- will be processed by VaeImageProcessor.
        inputs["image"] = Image.fromarray((np.random.RandomState(seed).rand(64, 64, 3) * 255).astype(np.uint8))
        inputs["image_guidance_scale"] = 1.5
        return inputs

    # ---- skips for mixin tests that genuinely don't apply ----------------
    # The remaining skips reflect intrinsic design choices of the DreamLite pipeline:
    #   * ``encode_prompt`` returns a ``(prompt_embeds, prompt_embeds_mask)``
    #     tuple, while the mixin's ``test_encode_prompt_works_in_isolation``
    #     assumes a single tensor return value;
    #   * the pipeline forces ``batch_size = 1`` internally, so the mixin's
    #     batch sweep cannot apply.
    @unittest.skip(
        "DreamLite intentionally limits ``batch_size`` to 1 (CFG memory blow-up); "
        "only ``num_images_per_prompt > 1`` is supported."
    )
    def test_num_images_per_prompt(self):
        pass

    @unittest.skip(
        "DreamLite encode_prompt returns (embeds, mask) tuple, not a single tensor; "
        "the mixin's test_encode_prompt_works_in_isolation assumes single tensor return."
    )
    def test_encode_prompt_works_in_isolation(self, extra_required_param_value_dict=None, atol=1e-4, rtol=1e-4):
        pass

    @unittest.skip(
        "Qwen3VLProcessor save_pretrained does not currently round-trip through DDUF "
        "(image_processor sub-config is dropped); orthogonal to DreamLite."
    )
    def test_save_load_dduf(self, atol=1e-4, rtol=1e-4):
        pass

    @unittest.skip("DreamLite forces batch_size=1 internally.")
    def test_inference_batch_consistent(self):
        pass

    @unittest.skip("DreamLite forces batch_size=1 internally.")
    def test_inference_batch_single_identical(self):
        pass

    # ---- actual tests ------------------------------------------------------
    def test_dreamlite_t2i_default_case(self):
        device = torch_device
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        out = pipe(**inputs).images
        out_np = to_np(out)

        # shape: (B=1, H, W, C=3)
        self.assertEqual(out_np.shape, (1, 64, 64, 3))
        self.assertFalse(np.isnan(out_np).any())

    def test_dreamlite_i2i_default_case(self):
        device = torch_device
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_i2i_inputs(device)
        out = pipe(**inputs).images
        out_np = to_np(out)

        self.assertEqual(out_np.shape, (1, 64, 64, 3))
        self.assertFalse(np.isnan(out_np).any())

    def test_dreamlite_cfg_branch_count(self):
        """In edit mode the pipeline must run a 3-way CFG concat (uncond/img/text)."""
        device = torch_device
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(device)
        pipe.set_progress_bar_config(disable=None)

        original_forward = pipe.unet.forward
        seen_batches = []

        def spy_forward(*args, **kwargs):
            x = args[0] if args else kwargs["sample"]
            seen_batches.append(x.shape[0])
            return original_forward(*args, **kwargs)

        pipe.unet.forward = spy_forward
        inputs = self.get_dummy_i2i_inputs(device)
        inputs["num_inference_steps"] = 1
        pipe(**inputs)

        self.assertTrue(all(b == 3 for b in seen_batches), f"expected all 3-way, got {seen_batches}")


@nightly
@require_torch_gpu
class DreamLitePipelineSlowTests(unittest.TestCase):
    """End-to-end test against the real DreamLite-base checkpoint on the Hub.

    By default this loads ``carlofkl/DreamLite-base`` (``diffusers`` branch)
    from the HF Hub. To run against a local copy during development, set the
    ``DREAMLITE_BASE_PATH`` env var to that path.
    """

    repo_id = "carlofkl/DreamLite-base"
    revision = "diffusers"

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
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

        self.assertEqual(out.shape, (1, 1024, 1024, 3))
        self.assertFalse(np.isnan(out).any())

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

        self.assertEqual(out.shape, (1, 1024, 1024, 3))
        self.assertFalse(np.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
