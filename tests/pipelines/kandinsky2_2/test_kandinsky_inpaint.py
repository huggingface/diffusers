# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
from PIL import Image

from diffusers import (
    DDIMScheduler,
    KandinskyV22InpaintPipeline,
    KandinskyV22PriorPipeline,
    UNet2DConditionModel,
    VQModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    is_flaky,
    load_image,
    load_numpy,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class Dummies:
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
        return 32

    @property
    def dummy_unet(self):
        torch.manual_seed(0)

        model_kwargs = {
            "in_channels": 9,
            # Out channels is double in channels because predicts mean and variance
            "out_channels": 8,
            "addition_embed_type": "image",
            "down_block_types": ("ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D"),
            "up_block_types": ("SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"),
            "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
            "block_out_channels": (self.block_out_channels_0, self.block_out_channels_0 * 2),
            "layers_per_block": 1,
            "encoder_hid_dim": self.text_embedder_hidden_size,
            "encoder_hid_dim_type": "image_proj",
            "cross_attention_dim": self.cross_attention_dim,
            "attention_head_dim": 4,
            "resnet_time_scale_shift": "scale_shift",
            "class_embed_type": None,
        }

        model = UNet2DConditionModel(**model_kwargs)
        return model

    @property
    def dummy_movq_kwargs(self):
        return {
            "block_out_channels": [32, 64],
            "down_block_types": ["DownEncoderBlock2D", "AttnDownEncoderBlock2D"],
            "in_channels": 3,
            "latent_channels": 4,
            "layers_per_block": 1,
            "norm_num_groups": 8,
            "norm_type": "spatial",
            "num_vq_embeddings": 12,
            "out_channels": 3,
            "up_block_types": [
                "AttnUpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],
            "vq_embed_dim": 4,
        }

    @property
    def dummy_movq(self):
        torch.manual_seed(0)
        model = VQModel(**self.dummy_movq_kwargs)
        return model

    def get_dummy_components(self):
        unet = self.dummy_unet
        movq = self.dummy_movq

        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="epsilon",
            thresholding=False,
        )

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "movq": movq,
        }

        return components

    def get_dummy_inputs(self, device, seed=0):
        image_embeds = floats_tensor((1, self.text_embedder_hidden_size), rng=random.Random(seed)).to(device)
        negative_image_embeds = floats_tensor((1, self.text_embedder_hidden_size), rng=random.Random(seed + 1)).to(
            device
        )
        # create init_image
        image = floats_tensor((1, 3, 64, 64), rng=random.Random(seed)).to(device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((256, 256))
        # create mask
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[:32, :32] = 1

        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "image": init_image,
            "mask_image": mask,
            "image_embeds": image_embeds,
            "negative_image_embeds": negative_image_embeds,
            "generator": generator,
            "height": 64,
            "width": 64,
            "num_inference_steps": 2,
            "guidance_scale": 4.0,
            "output_type": "np",
        }
        return inputs


class KandinskyV22InpaintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KandinskyV22InpaintPipeline
    params = ["image_embeds", "negative_image_embeds", "image", "mask_image"]
    batch_params = [
        "image_embeds",
        "negative_image_embeds",
        "image",
        "mask_image",
    ]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "latents",
        "guidance_scale",
        "num_inference_steps",
        "return_dict",
        "guidance_scale",
        "num_images_per_prompt",
        "output_type",
        "return_dict",
    ]
    test_xformers_attention = False
    callback_cfg_params = ["image_embeds", "masked_image", "mask_image"]

    def get_dummy_components(self):
        dummies = Dummies()
        return dummies.get_dummy_components()

    def get_dummy_inputs(self, device, seed=0):
        dummies = Dummies()
        return dummies.get_dummy_inputs(device=device, seed=seed)

    def test_kandinsky_inpaint(self):
        device = "cpu"

        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(device))
        image = output.images

        image_from_tuple = pipe(
            **self.get_dummy_inputs(device),
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array(
            [0.50775903, 0.49527195, 0.48824543, 0.50192237, 0.48644906, 0.49373814, 0.4780598, 0.47234827, 0.48327848]
        )

        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"
        assert (
            np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2
        ), f" expected_slice {expected_slice}, but got {image_from_tuple_slice.flatten()}"

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    def test_float16_inference(self):
        super().test_float16_inference(expected_max_diff=5e-1)

    @is_flaky()
    def test_model_cpu_offload_forward_pass(self):
        super().test_inference_batch_single_identical(expected_max_diff=8e-4)

    def test_save_load_optional_components(self):
        super().test_save_load_optional_components(expected_max_difference=5e-4)

    def test_sequential_cpu_offload_forward_pass(self):
        super().test_sequential_cpu_offload_forward_pass(expected_max_diff=5e-4)

    # override default test because we need to zero out mask too in order to make sure final latent is all zero
    def test_callback_inputs(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        self.assertTrue(
            hasattr(pipe, "_callback_tensor_inputs"),
            f" {self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables its callback function can use as inputs",
        )

        def callback_inputs_test(pipe, i, t, callback_kwargs):
            missing_callback_inputs = set()
            for v in pipe._callback_tensor_inputs:
                if v not in callback_kwargs:
                    missing_callback_inputs.add(v)
            self.assertTrue(
                len(missing_callback_inputs) == 0, f"Missing callback tensor inputs: {missing_callback_inputs}"
            )
            last_i = pipe.num_timesteps - 1
            if i == last_i:
                callback_kwargs["latents"] = torch.zeros_like(callback_kwargs["latents"])
                callback_kwargs["mask_image"] = torch.zeros_like(callback_kwargs["mask_image"])
            return callback_kwargs

        inputs = self.get_dummy_inputs(torch_device)
        inputs["callback_on_step_end"] = callback_inputs_test
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        inputs["output_type"] = "latent"

        output = pipe(**inputs)[0]
        assert output.abs().sum() == 0


@slow
@require_torch_gpu
class KandinskyV22InpaintPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_kandinsky_inpaint(self):
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/kandinskyv22/kandinskyv22_inpaint_cat_with_hat_fp16.npy"
        )

        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main" "/kandinsky/cat.png"
        )
        mask = np.zeros((768, 768), dtype=np.float32)
        mask[:250, 250:-250] = 1

        prompt = "a hat"

        pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
        )
        pipe_prior.to(torch_device)

        pipeline = KandinskyV22InpaintPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
        )
        pipeline = pipeline.to(torch_device)
        pipeline.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        image_emb, zero_image_emb = pipe_prior(
            prompt,
            generator=generator,
            num_inference_steps=2,
            negative_prompt="",
        ).to_tuple()

        generator = torch.Generator(device="cpu").manual_seed(0)
        output = pipeline(
            image=init_image,
            mask_image=mask,
            image_embeds=image_emb,
            negative_image_embeds=zero_image_emb,
            generator=generator,
            num_inference_steps=2,
            height=768,
            width=768,
            output_type="np",
        )

        image = output.images[0]

        assert image.shape == (768, 768, 3)

        max_diff = numpy_cosine_similarity_distance(expected_image.flatten(), image.flatten())
        assert max_diff < 1e-4
