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
``test_pipeline_dreamlite.py``; see that file for the rationale behind
mocking ``encode_prompt``.
"""

import gc
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from diffusers import (
    AutoencoderTiny,
    DreamLiteMobilePipeline,
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


_CROSS_ATTN_DIM = 32
_DUMMY_SEQ_LEN = 8


def _make_fake_encode_prompt(cross_attn_dim: int = _CROSS_ATTN_DIM, seq_len: int = _DUMMY_SEQ_LEN):
    def fake_encode_prompt(
        self,
        mode,
        prompts,
        device,
        dtype,
        image=None,
        max_sequence_length=500,
        text_pad_embedding=None,
    ):
        batch = len(prompts)
        prompt_embeds = torch.randn(batch, seq_len, cross_attn_dim, device=device, dtype=dtype)
        prompt_embeds_mask = torch.ones(batch, seq_len, device=device, dtype=torch.long)
        return prompt_embeds, prompt_embeds_mask

    return fake_encode_prompt


class DreamLiteMobilePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = DreamLiteMobilePipeline
    params = frozenset(
        [
            "prompt",
            "height",
            "width",
            "num_inference_steps",
        ]
    )
    batch_params = frozenset(["prompt"])
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

        text_encoder = MagicMock()
        text_encoder.dtype = torch.float32
        text_encoder.to = MagicMock(return_value=text_encoder)
        text_encoder.eval = MagicMock(return_value=text_encoder)

        tokenizer = MagicMock()
        processor = MagicMock()

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
            "generator": generator,
            "num_inference_steps": 2,
            "height": 64,
            "width": 64,
            "output_type": "np",
        }

    def get_dummy_i2i_inputs(self, device, seed=0):
        inputs = self.get_dummy_inputs(device, seed)
        inputs["image"] = Image.fromarray((np.random.RandomState(seed).rand(64, 64, 3) * 255).astype(np.uint8))
        return inputs

    # ---- mixin compatibility: auto-patch encode_prompt for ALL tests ------
    # PipelineTesterMixin tests (test_cfg, test_inference_batch_*, etc.) do
    # not know about ``_patch_encode_prompt`` and instantiate the pipeline
    # themselves, then call it. Without a patched ``encode_prompt`` they hit
    # the real Qwen3-VL code path (``drop_idx=34`` slice on a MagicMock
    # tokenizer output) and crash inside ``pad_sequence``. Patching at the
    # class level via ``unittest.mock.patch.object`` covers every pipeline
    # instance built during a test method, with automatic teardown.
    def setUp(self):
        super().setUp()
        self._encode_prompt_patcher = patch.object(
            self.pipeline_class,
            "encode_prompt",
            _make_fake_encode_prompt(_CROSS_ATTN_DIM, _DUMMY_SEQ_LEN),
        )
        self._encode_prompt_patcher.start()

    def tearDown(self):
        self._encode_prompt_patcher.stop()
        super().tearDown()

    def _patch_encode_prompt(self, pipe):
        fake = _make_fake_encode_prompt(_CROSS_ATTN_DIM, _DUMMY_SEQ_LEN)
        pipe.encode_prompt = fake.__get__(pipe, type(pipe))

    # ---- override mixin tests that don't apply to DreamLite ---------------
    # The following inherited PipelineTesterMixin tests are skipped because
    # they make assumptions that don't fit DreamLite's design.
    # This mirrors what SD3 / Flux do for the same incompatibilities.
    @unittest.skip("MagicMock text_encoder has no real dtype propagation.")
    def test_to_dtype(self):
        pass

    @unittest.skip("MagicMock text_encoder has no real dtype propagation.")
    def test_torch_dtype_dict(self):
        pass

    @unittest.skip("MagicMock components cannot be serialised via save_pretrained.")
    def test_save_load_dduf(self, atol=1e-4, rtol=1e-4):
        pass

    @unittest.skip("MagicMock components cannot be serialised via save_pretrained.")
    def test_loading_with_variants(self):
        pass

    @unittest.skip("MagicMock components cannot be serialised via save_pretrained.")
    def test_pipeline_with_accelerator_device_map(self):
        pass

    @unittest.skip(
        "DreamLite UNet uses DreamLiteAttnProcessor2_0 which is not in "
        "UNet2DConditionModel's default processor set; set_default_attn_processor raises."
    )
    def test_dict_tuple_outputs_equivalent(self, expected_max_difference=0.0001):
        pass

    @unittest.skip(
        "DreamLite encode_prompt returns (embeds, mask) tuple, not a single tensor; "
        "the mixin's test_encode_prompt_works_in_isolation assumes single tensor return."
    )
    def test_encode_prompt_works_in_isolation(self, extra_required_param_value_dict=None, atol=1e-4, rtol=1e-4):
        pass

    @unittest.skip(
        "DreamLite intentionally limits ``batch_size`` to 1 (CFG memory blow-up); "
        "only ``num_images_per_prompt > 1`` is supported. The mixin sweep over "
        "batch_size=[1, 2] x num_images_per_prompt=[1, 2] would fail on "
        "batch_size=2 cases."
    )
    def test_num_images_per_prompt(self):
        pass

    def test_mobile_t2i_default_case(self):
        device = torch_device
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(device)
        pipe.set_progress_bar_config(disable=None)
        self._patch_encode_prompt(pipe)

        inputs = self.get_dummy_inputs(device)
        out = pipe(**inputs).images
        out_np = to_np(out)

        self.assertEqual(out_np.shape, (1, 64, 64, 3))
        self.assertFalse(np.isnan(out_np).any())

    def test_mobile_i2i_default_case(self):
        device = torch_device
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(device)
        pipe.set_progress_bar_config(disable=None)
        self._patch_encode_prompt(pipe)

        inputs = self.get_dummy_i2i_inputs(device)
        out = pipe(**inputs).images
        out_np = to_np(out)

        self.assertEqual(out_np.shape, (1, 64, 64, 3))
        self.assertFalse(np.isnan(out_np).any())

    def test_mobile_single_forward_per_step(self):
        """Mobile pipeline must run exactly ONE UNet forward per step (no CFG concat)."""
        device = torch_device
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(device)
        pipe.set_progress_bar_config(disable=None)
        self._patch_encode_prompt(pipe)

        original_forward = pipe.unet.forward
        seen_batches = []

        def spy_forward(*args, **kwargs):
            x = args[0] if args else kwargs["sample"]
            seen_batches.append(x.shape[0])
            return original_forward(*args, **kwargs)

        pipe.unet.forward = spy_forward
        inputs = self.get_dummy_i2i_inputs(device)
        inputs["num_inference_steps"] = 2
        pipe(**inputs)

        self.assertTrue(all(b == 1 for b in seen_batches), f"expected all 1-way, got {seen_batches}")
        self.assertEqual(len(seen_batches), 2, "expected exactly 2 unet calls (1 per step)")

    def test_mobile_guidance_scale_ignored(self):
        """Passing guidance_scale to the mobile pipeline should be accepted but ignored (with warning)."""
        device = torch_device
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(device)
        pipe.set_progress_bar_config(disable=None)
        self._patch_encode_prompt(pipe)

        inputs = self.get_dummy_inputs(device)
        inputs["guidance_scale"] = 7.5  # should not raise
        inputs["image_guidance_scale"] = 1.5  # should not raise
        out = pipe(**inputs).images
        self.assertEqual(to_np(out).shape, (1, 64, 64, 3))

    @unittest.skip("DreamLiteMobile uses mocked text_encoder; save/load round-trip is N/A.")
    def test_save_load_local(self):
        pass

    @unittest.skip("DreamLiteMobile uses mocked text_encoder; save/load round-trip is N/A.")
    def test_save_load_optional_components(self):
        pass

    @unittest.skip("DreamLiteMobile uses mocked text_encoder; save/load round-trip is N/A.")
    def test_save_load_float16(self):
        pass

    @unittest.skip("DreamLiteMobile uses mocked text_encoder; from_pretrained round-trip is N/A.")
    def test_from_pipe_consistent_config(self):
        pass

    @unittest.skip("DreamLiteMobile uses mocked text_encoder; serialization is N/A.")
    def test_serialization(self):
        pass

    @unittest.skip("DreamLiteMobile forces batch_size=1 internally.")
    def test_inference_batch_consistent(self):
        pass

    @unittest.skip("DreamLiteMobile forces batch_size=1 internally.")
    def test_inference_batch_single_identical(self):
        pass


@nightly
@require_torch_gpu
class DreamLiteMobilePipelineSlowTests(unittest.TestCase):
    """End-to-end test against the real DreamLite-mobile checkpoint on the Hub.

    By default this loads ``carlofkl/DreamLite-mobile`` (``diffusers`` branch)
    from the HF Hub. To run against a local copy during development, set the
    ``DREAMLITE_MOBILE_PATH`` env var to that path.
    """

    repo_id = "carlofkl/DreamLite-mobile"
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

        self.assertEqual(out.shape, (1, 1024, 1024, 3))
        self.assertFalse(np.isnan(out).any())

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

        self.assertEqual(out.shape, (1, 1024, 1024, 3))
        self.assertFalse(np.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
