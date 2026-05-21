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

import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

import diffusers.models.autoencoders.autoencoder_rae as _rae_module
from diffusers import AutoencoderRAE, FlowMatchEulerDiscreteScheduler, RAEDiTPipeline, RAEDiTPipelineOutput
from diffusers.models.autoencoders.autoencoder_rae import _ENCODER_FORWARD_FNS, _build_encoder
from diffusers.models.transformers.transformer_rae_dit import RAEDiT2DModel

from ...testing_utils import enable_full_determinism, torch_device
from ..pipeline_params import (
    CLASS_CONDITIONED_IMAGE_GENERATION_BATCH_PARAMS,
    CLASS_CONDITIONED_IMAGE_GENERATION_PARAMS,
)
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


def _initialize_non_zero_stage2_head(model: RAEDiT2DModel):
    torch.manual_seed(0)

    for block in model.blocks:
        block.adaLN_modulation[-1].weight.data.normal_(mean=0.0, std=0.02)
        block.adaLN_modulation[-1].bias.data.normal_(mean=0.0, std=0.02)

    model.final_layer.adaLN_modulation[-1].weight.data.normal_(mean=0.0, std=0.02)
    model.final_layer.adaLN_modulation[-1].bias.data.normal_(mean=0.0, std=0.02)
    model.final_layer.linear.weight.data.normal_(mean=0.0, std=0.02)
    model.final_layer.linear.bias.data.normal_(mean=0.0, std=0.02)


class _TinyTestEncoderModule(torch.nn.Module):
    def __init__(self, hidden_size: int = 8, patch_size: int = 4, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(images.mean(dim=1, keepdim=True), kernel_size=self.patch_size, stride=self.patch_size)
        tokens = pooled.flatten(2).transpose(1, 2).contiguous()
        return tokens.repeat(1, 1, self.hidden_size)


def _tiny_test_encoder_forward(model, images):
    return model(images)


def _build_tiny_test_encoder(encoder_type, hidden_size, patch_size, num_hidden_layers):
    return _TinyTestEncoderModule(hidden_size=hidden_size, patch_size=patch_size)


_ENCODER_FORWARD_FNS["tiny_test"] = _tiny_test_encoder_forward
_original_build_encoder = _build_encoder


def _patched_build_encoder(encoder_type, hidden_size, patch_size, num_hidden_layers):
    if encoder_type == "tiny_test":
        return _build_tiny_test_encoder(encoder_type, hidden_size, patch_size, num_hidden_layers)
    return _original_build_encoder(encoder_type, hidden_size, patch_size, num_hidden_layers)


_rae_module._build_encoder = _patched_build_encoder


class RAEDiTPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = RAEDiTPipeline
    params = CLASS_CONDITIONED_IMAGE_GENERATION_PARAMS
    batch_params = CLASS_CONDITIONED_IMAGE_GENERATION_BATCH_PARAMS
    test_attention_slicing = False
    test_xformers_attention = False

    @classmethod
    def tearDownClass(cls):
        _rae_module._build_encoder = _original_build_encoder
        _ENCODER_FORWARD_FNS.pop("tiny_test", None)
        super().tearDownClass()

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = RAEDiT2DModel(
            sample_size=2,
            patch_size=1,
            in_channels=8,
            hidden_size=(16, 16),
            depth=(1, 1),
            num_heads=(2, 2),
            mlp_ratio=2.0,
            class_dropout_prob=0.1,
            num_classes=4,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            use_pos_embed=True,
        )
        _initialize_non_zero_stage2_head(transformer)

        vae = AutoencoderRAE(
            encoder_type="tiny_test",
            encoder_hidden_size=8,
            encoder_patch_size=4,
            encoder_num_hidden_layers=1,
            decoder_hidden_size=16,
            decoder_num_hidden_layers=1,
            decoder_num_attention_heads=2,
            decoder_intermediate_size=32,
            patch_size=2,
            encoder_input_size=8,
            image_size=4,
            num_channels=3,
            encoder_norm_mean=[0.5, 0.5, 0.5],
            encoder_norm_std=[0.5, 0.5, 0.5],
            noise_tau=0.0,
            reshape_to_2d=True,
            scaling_factor=1.0,
        )
        scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)

        return {
            "transformer": transformer.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
        }

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        return {
            "class_labels": [1],
            "generator": generator,
            "guidance_scale": 1.0,
            "num_inference_steps": 2,
            "output_type": "np",
        }

    def test_save_load_local(self, expected_max_difference=5e-4):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(torch_device))[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir, safe_serialization=False)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        output_loaded = pipe_loaded(**self.get_dummy_inputs(torch_device))[0]
        max_diff = np.abs(output_loaded - output).max()
        self.assertLess(max_diff, expected_max_difference)

    def test_inference(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to("cpu")
        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs("cpu"))
        self.assertIsInstance(output, RAEDiTPipelineOutput)
        image = output.images
        image_slice = image[0, -2:, -2:, -1]

        self.assertEqual(image.shape, (1, 4, 4, 3))
        expected_slice = np.array([0.78739226, 0.79371649, 0.56565261, 0.78660309])
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-4)

    def test_inference_casts_latents_to_vae_dtype_before_decode(self):
        components = self.get_dummy_components()
        components["vae"] = components["vae"].to(dtype=torch.float64)
        pipe = self.pipeline_class(**components).to("cpu")
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs("cpu")
        inputs["output_type"] = "pt"

        images = pipe(**inputs).images

        self.assertEqual(images.shape, (1, 3, 4, 4))
        self.assertTrue(torch.isfinite(images).all().item())

    def test_inference_classifier_free_guidance(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to("cpu")
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs("cpu")
        inputs.update({"guidance_scale": 2.0})

        image = pipe(**inputs).images
        self.assertEqual(image.shape, (1, 4, 4, 3))
        self.assertTrue(np.isfinite(image).all())

        no_guidance = pipe(**self.get_dummy_inputs("cpu")).images
        self.assertGreater(np.abs(image - no_guidance).max(), 1e-6)

    def test_guidance_interval_can_disable_cfg(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to("cpu")
        pipe.set_progress_bar_config(disable=None)

        base = pipe(**self.get_dummy_inputs("cpu")).images

        inputs = self.get_dummy_inputs("cpu")
        inputs.pop("guidance_scale")
        cfg_disabled = pipe(
            **inputs,
            guidance_scale=2.0,
            guidance_start=0.25,
            guidance_end=0.75,
        ).images

        self.assertLessEqual(np.abs(base - cfg_disabled).max(), 1e-5)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=1e-4)

    def test_latent_output(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to("cpu")
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs("cpu")
        inputs.pop("output_type")
        latents = pipe(**inputs, output_type="latent").images
        self.assertEqual(latents.shape, (1, 8, 2, 2))
        self.assertTrue(torch.isfinite(latents).all().item())

    def test_get_label_ids(self):
        pipe = self.pipeline_class(
            **self.get_dummy_components(),
            id2label={
                0: "zero",
                1: "one, first",
            },
        )
        self.assertEqual(pipe.get_label_ids("first"), [1])
        self.assertEqual(pipe.get_label_ids(["zero", "one"]), [0, 1])

    def test_save_load_preserves_label_ids(self):
        pipe = self.pipeline_class(
            **self.get_dummy_components(),
            id2label={
                0: "zero",
                1: "one, first",
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir, safe_serialization=False)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)

        self.assertEqual(pipe_loaded.config.id2label, {"0": "zero", "1": "one, first"})
        self.assertEqual(pipe_loaded.get_label_ids("first"), [1])
        self.assertEqual(pipe_loaded.get_label_ids(["zero", "one"]), [0, 1])
