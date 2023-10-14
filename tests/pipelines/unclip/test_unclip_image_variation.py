# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
import random
import unittest

import numpy as np
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    DiffusionPipeline,
    UnCLIPImageVariationPipeline,
    UnCLIPScheduler,
    UNet2DConditionModel,
    UNet2DModel,
)
from diffusers.pipelines.unclip.text_proj import UnCLIPTextProjModel
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    nightly,
    require_torch_gpu,
    skip_mps,
    torch_device,
)

from ..pipeline_params import IMAGE_VARIATION_BATCH_PARAMS, IMAGE_VARIATION_PARAMS
from ..test_pipelines_common import PipelineTesterMixin, assert_mean_pixel_difference


enable_full_determinism()


class UnCLIPImageVariationPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = UnCLIPImageVariationPipeline
    params = IMAGE_VARIATION_PARAMS - {"height", "width", "guidance_scale"}
    batch_params = IMAGE_VARIATION_BATCH_PARAMS

    required_optional_params = [
        "generator",
        "return_dict",
        "decoder_num_inference_steps",
        "super_res_num_inference_steps",
    ]
    test_xformers_attention = False

    @property
    def text_embedder_hidden_size(self):
        return 32

    @property
    def time_input_dim(self):
        return 32

    @property
    def block_out_channels_0(self):
        return self.time_input_dim

    @property
    def time_embed_dim(self):
        return self.time_input_dim * 4

    @property
    def cross_attention_dim(self):
        return 100

    @property
    def dummy_tokenizer(self):
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        return tokenizer

    @property
    def dummy_text_encoder(self):
        torch.manual_seed(0)
        config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=self.text_embedder_hidden_size,
            projection_dim=self.text_embedder_hidden_size,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        return CLIPTextModelWithProjection(config)

    @property
    def dummy_image_encoder(self):
        torch.manual_seed(0)
        config = CLIPVisionConfig(
            hidden_size=self.text_embedder_hidden_size,
            projection_dim=self.text_embedder_hidden_size,
            num_hidden_layers=5,
            num_attention_heads=4,
            image_size=32,
            intermediate_size=37,
            patch_size=1,
        )
        return CLIPVisionModelWithProjection(config)

    @property
    def dummy_text_proj(self):
        torch.manual_seed(0)

        model_kwargs = {
            "clip_embeddings_dim": self.text_embedder_hidden_size,
            "time_embed_dim": self.time_embed_dim,
            "cross_attention_dim": self.cross_attention_dim,
        }

        model = UnCLIPTextProjModel(**model_kwargs)
        return model

    @property
    def dummy_decoder(self):
        torch.manual_seed(0)

        model_kwargs = {
            "sample_size": 32,
            # RGB in channels
            "in_channels": 3,
            # Out channels is double in channels because predicts mean and variance
            "out_channels": 6,
            "down_block_types": ("ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D"),
            "up_block_types": ("SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"),
            "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
            "block_out_channels": (self.block_out_channels_0, self.block_out_channels_0 * 2),
            "layers_per_block": 1,
            "cross_attention_dim": self.cross_attention_dim,
            "attention_head_dim": 4,
            "resnet_time_scale_shift": "scale_shift",
            "class_embed_type": "identity",
        }

        model = UNet2DConditionModel(**model_kwargs)
        return model

    @property
    def dummy_super_res_kwargs(self):
        return {
            "sample_size": 64,
            "layers_per_block": 1,
            "down_block_types": ("ResnetDownsampleBlock2D", "ResnetDownsampleBlock2D"),
            "up_block_types": ("ResnetUpsampleBlock2D", "ResnetUpsampleBlock2D"),
            "block_out_channels": (self.block_out_channels_0, self.block_out_channels_0 * 2),
            "in_channels": 6,
            "out_channels": 3,
        }

    @property
    def dummy_super_res_first(self):
        torch.manual_seed(0)

        model = UNet2DModel(**self.dummy_super_res_kwargs)
        return model

    @property
    def dummy_super_res_last(self):
        # seeded differently to get different unet than `self.dummy_super_res_first`
        torch.manual_seed(1)

        model = UNet2DModel(**self.dummy_super_res_kwargs)
        return model

    def get_dummy_components(self):
        decoder = self.dummy_decoder
        text_proj = self.dummy_text_proj
        text_encoder = self.dummy_text_encoder
        tokenizer = self.dummy_tokenizer
        super_res_first = self.dummy_super_res_first
        super_res_last = self.dummy_super_res_last

        decoder_scheduler = UnCLIPScheduler(
            variance_type="learned_range",
            prediction_type="epsilon",
            num_train_timesteps=1000,
        )

        super_res_scheduler = UnCLIPScheduler(
            variance_type="fixed_small_log",
            prediction_type="epsilon",
            num_train_timesteps=1000,
        )

        feature_extractor = CLIPImageProcessor(crop_size=32, size=32)

        image_encoder = self.dummy_image_encoder

        return {
            "decoder": decoder,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_proj": text_proj,
            "feature_extractor": feature_extractor,
            "image_encoder": image_encoder,
            "super_res_first": super_res_first,
            "super_res_last": super_res_last,
            "decoder_scheduler": decoder_scheduler,
            "super_res_scheduler": super_res_scheduler,
        }

    def get_dummy_inputs(self, device, seed=0, pil_image=True):
        input_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        if pil_image:
            input_image = input_image * 0.5 + 0.5
            input_image = input_image.clamp(0, 1)
            input_image = input_image.cpu().permute(0, 2, 3, 1).float().numpy()
            input_image = DiffusionPipeline.numpy_to_pil(input_image)[0]

        return {
            "image": input_image,
            "generator": generator,
            "decoder_num_inference_steps": 2,
            "super_res_num_inference_steps": 2,
            "output_type": "np",
        }

    def test_unclip_image_variation_input_tensor(self):
        device = "cpu"

        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        pipeline_inputs = self.get_dummy_inputs(device, pil_image=False)

        output = pipe(**pipeline_inputs)
        image = output.images

        tuple_pipeline_inputs = self.get_dummy_inputs(device, pil_image=False)

        image_from_tuple = pipe(
            **tuple_pipeline_inputs,
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array(
            [
                0.9997,
                0.0002,
                0.9997,
                0.9997,
                0.9969,
                0.0023,
                0.9997,
                0.9969,
                0.9970,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_unclip_image_variation_input_image(self):
        device = "cpu"

        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        pipeline_inputs = self.get_dummy_inputs(device, pil_image=True)

        output = pipe(**pipeline_inputs)
        image = output.images

        tuple_pipeline_inputs = self.get_dummy_inputs(device, pil_image=True)

        image_from_tuple = pipe(
            **tuple_pipeline_inputs,
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([0.9997, 0.0003, 0.9997, 0.9997, 0.9970, 0.0024, 0.9997, 0.9971, 0.9971])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_unclip_image_variation_input_list_images(self):
        device = "cpu"

        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        pipeline_inputs = self.get_dummy_inputs(device, pil_image=True)
        pipeline_inputs["image"] = [
            pipeline_inputs["image"],
            pipeline_inputs["image"],
        ]

        output = pipe(**pipeline_inputs)
        image = output.images

        tuple_pipeline_inputs = self.get_dummy_inputs(device, pil_image=True)
        tuple_pipeline_inputs["image"] = [
            tuple_pipeline_inputs["image"],
            tuple_pipeline_inputs["image"],
        ]

        image_from_tuple = pipe(
            **tuple_pipeline_inputs,
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (2, 64, 64, 3)

        expected_slice = np.array(
            [
                0.9997,
                0.9989,
                0.0008,
                0.0021,
                0.9960,
                0.0018,
                0.0014,
                0.0002,
                0.9933,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_unclip_passed_image_embed(self):
        device = torch.device("cpu")

        class DummyScheduler:
            init_noise_sigma = 1

        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(0)
        dtype = pipe.decoder.dtype
        batch_size = 1

        shape = (
            batch_size,
            pipe.decoder.config.in_channels,
            pipe.decoder.config.sample_size,
            pipe.decoder.config.sample_size,
        )
        decoder_latents = pipe.prepare_latents(
            shape, dtype=dtype, device=device, generator=generator, latents=None, scheduler=DummyScheduler()
        )

        shape = (
            batch_size,
            pipe.super_res_first.config.in_channels // 2,
            pipe.super_res_first.config.sample_size,
            pipe.super_res_first.config.sample_size,
        )
        super_res_latents = pipe.prepare_latents(
            shape, dtype=dtype, device=device, generator=generator, latents=None, scheduler=DummyScheduler()
        )

        pipeline_inputs = self.get_dummy_inputs(device, pil_image=False)

        img_out_1 = pipe(
            **pipeline_inputs, decoder_latents=decoder_latents, super_res_latents=super_res_latents
        ).images

        pipeline_inputs = self.get_dummy_inputs(device, pil_image=False)
        # Don't pass image, instead pass embedding
        image = pipeline_inputs.pop("image")
        image_embeddings = pipe.image_encoder(image).image_embeds

        img_out_2 = pipe(
            **pipeline_inputs,
            decoder_latents=decoder_latents,
            super_res_latents=super_res_latents,
            image_embeddings=image_embeddings,
        ).images

        # make sure passing text embeddings manually is identical
        assert np.abs(img_out_1 - img_out_2).max() < 1e-4

    # Overriding PipelineTesterMixin::test_attention_slicing_forward_pass
    # because UnCLIP GPU undeterminism requires a looser check.
    @skip_mps
    def test_attention_slicing_forward_pass(self):
        test_max_difference = torch_device == "cpu"

        # Check is relaxed because there is not a torch 2.0 sliced attention added kv processor
        expected_max_diff = 1e-2

        self._test_attention_slicing_forward_pass(
            test_max_difference=test_max_difference, expected_max_diff=expected_max_diff
        )

    # Overriding PipelineTesterMixin::test_inference_batch_single_identical
    # because UnCLIP undeterminism requires a looser check.
    @unittest.skip("UnCLIP produces very large differences. Test is not useful.")
    @skip_mps
    def test_inference_batch_single_identical(self):
        additional_params_copy_to_batched_inputs = [
            "decoder_num_inference_steps",
            "super_res_num_inference_steps",
        ]
        self._test_inference_batch_single_identical(
            additional_params_copy_to_batched_inputs=additional_params_copy_to_batched_inputs, expected_max_diff=5e-3
        )

    def test_inference_batch_consistent(self):
        additional_params_copy_to_batched_inputs = [
            "decoder_num_inference_steps",
            "super_res_num_inference_steps",
        ]

        if torch_device == "mps":
            # TODO: MPS errors with larger batch sizes
            batch_sizes = [2, 3]
            self._test_inference_batch_consistent(
                batch_sizes=batch_sizes,
                additional_params_copy_to_batched_inputs=additional_params_copy_to_batched_inputs,
            )
        else:
            self._test_inference_batch_consistent(
                additional_params_copy_to_batched_inputs=additional_params_copy_to_batched_inputs
            )

    @skip_mps
    def test_dict_tuple_outputs_equivalent(self):
        return super().test_dict_tuple_outputs_equivalent()

    @unittest.skip("UnCLIP produces very large difference. Test is not useful.")
    @skip_mps
    def test_save_load_local(self):
        return super().test_save_load_local(expected_max_difference=4e-3)

    @skip_mps
    def test_save_load_optional_components(self):
        return super().test_save_load_optional_components()

    @unittest.skip("UnCLIP produces very large difference in fp16 vs fp32. Test is not useful.")
    def test_float16_inference(self):
        super().test_float16_inference(expected_max_diff=1.0)


@nightly
@require_torch_gpu
class UnCLIPImageVariationPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_unclip_image_variation_karlo(self):
        input_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/unclip/cat.png"
        )
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/unclip/karlo_v1_alpha_cat_variation_fp16.npy"
        )

        pipeline = UnCLIPImageVariationPipeline.from_pretrained(
            "kakaobrain/karlo-v1-alpha-image-variations", torch_dtype=torch.float16
        )
        pipeline = pipeline.to(torch_device)
        pipeline.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        output = pipeline(
            input_image,
            generator=generator,
            output_type="np",
        )

        image = output.images[0]

        assert image.shape == (256, 256, 3)

        assert_mean_pixel_difference(image, expected_image, 15)
