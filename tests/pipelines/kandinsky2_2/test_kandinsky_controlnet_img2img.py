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
import random

import numpy as np
import pytest
import torch
from PIL import Image

from diffusers import (
    DDIMScheduler,
    KandinskyV22ControlnetImg2ImgPipeline,
    KandinskyV22PriorEmb2EmbPipeline,
    UNet2DConditionModel,
    VQModel,
)

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    nightly,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    torch_device,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


# `UNet2DConditionModel` builds an `ImageProjection` for `encoder_hid_dim_type="image_proj"`, and its `forward`
# aligns the input with `self.image_embeds.weight.dtype`. Under layerwise casting that reads the *storage* dtype
# (fp8), because the weight is only upcast inside `self.image_embeds`'s own hooked forward — so the input is pushed
# down to fp8 and the matmul then fails against the upcast bf16 weight. `TextImageProjection` (Kandinsky 2.1) calls
# the projection without reading its weight dtype and is unaffected.
LAYERWISE_CASTING_XFAIL_REASON = (
    "`ImageProjection.forward` reads `self.image_embeds.weight.dtype`, which is the fp8 storage dtype under "
    "layerwise casting, so the input is cast down to fp8 and the matmul fails."
)


class KandinskyV22ControlnetImg2ImgPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = KandinskyV22ControlnetImg2ImgPipeline
    required_input_params_in_call_signature = frozenset(["image_embeds", "negative_image_embeds", "image", "hint"])
    batch_input_params = frozenset(["image_embeds", "negative_image_embeds", "image", "hint"])
    # The pipeline starts denoising from the encoded `image`, so it takes no `latents` argument.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_images_per_prompt", "generator", "output_type", "return_dict"]
    )
    output_shape = (3, 64, 64)

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
    def dummy_unet(self):
        torch.manual_seed(0)

        model_kwargs = {
            "in_channels": 8,
            # Out channels is double in channels because predicts mean and variance
            "out_channels": 8,
            "addition_embed_type": "image_hint",
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
            "block_out_channels": [32, 32, 64, 64],
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "AttnDownEncoderBlock2D",
            ],
            "in_channels": 3,
            "latent_channels": 4,
            "layers_per_block": 1,
            "norm_num_groups": 8,
            "norm_type": "spatial",
            "num_vq_embeddings": 12,
            "out_channels": 3,
            "up_block_types": ["AttnUpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
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

        ddim_config = {
            "num_train_timesteps": 1000,
            "beta_schedule": "linear",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "clip_sample": False,
            "set_alpha_to_one": False,
            "steps_offset": 0,
            "prediction_type": "epsilon",
            "thresholding": False,
        }

        scheduler = DDIMScheduler(**ddim_config)

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "movq": movq,
        }

        return components

    def get_dummy_inputs(self):
        image_embeds = torch.randn((1, self.text_embedder_hidden_size), generator=self.get_generator(0)).to(
            torch_device
        )
        negative_image_embeds = torch.randn((1, self.text_embedder_hidden_size), generator=self.get_generator(1)).to(
            torch_device
        )
        # create init_image
        image = floats_tensor((1, 3, 64, 64), rng=random.Random(0))
        image = image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((256, 256))
        # create hint
        hint = floats_tensor((1, 3, 64, 64), rng=random.Random(0)).to(torch_device)

        return {
            "image": init_image,
            "image_embeds": image_embeds,
            "negative_image_embeds": negative_image_embeds,
            "hint": hint,
            "generator": self.get_generator(0),
            "height": 64,
            "width": 64,
            "num_inference_steps": 10,
            "guidance_scale": 7.0,
            "strength": 0.2,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestKandinskyV22ControlnetImg2ImgPipeline(
    KandinskyV22ControlnetImg2ImgPipelineTesterConfig, PipelineTesterMixin
):
    def test_kandinsky_controlnet_img2img(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        image_from_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]

        assert image.shape == (1, *self.output_shape)

        # fmt: off
        expected_slice = torch.tensor([0.5381, 0.5271, 0.4858, 0.5367, 0.5259, 0.4887, 0.4877, 0.4901, 0.4835])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)
        assert_tensors_close(image_from_tuple[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=5e-2):
        # Batched inference is only approximately equal to single inference here: the tiny 2-step denoising loop
        # amplifies the numerical differences of the batched forward. Tolerance set from the measured drift.
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)


class TestKandinskyV22ControlnetImg2ImgPipelineMemory(
    KandinskyV22ControlnetImg2ImgPipelineTesterConfig, MemoryTesterMixin
):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Kandinsky 2.2 ControlNet
    img2img pipeline."""

    @pytest.mark.xfail(condition=True, reason=LAYERWISE_CASTING_XFAIL_REASON, strict=True)
    def test_layerwise_casting_inference(self):
        super().test_layerwise_casting_inference()


@nightly
@require_torch_accelerator
class TestKandinskyV22ControlnetImg2ImgPipelineIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        # clean up the VRAM before and after each test
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_kandinsky_controlnet_img2img(self):
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/kandinskyv22/kandinskyv22_controlnet_img2img_robotcat_fp16.npy"
        )

        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png"
        )
        init_image = init_image.resize((512, 512))

        hint = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/kandinskyv22/hint_image_cat.png"
        )
        hint = torch.from_numpy(np.array(hint)).float() / 255.0
        hint = hint.permute(2, 0, 1).unsqueeze(0)

        prompt = "A robot, 4k photo"

        pipe_prior = KandinskyV22PriorEmb2EmbPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
        )
        pipe_prior.enable_model_cpu_offload()

        pipeline = KandinskyV22ControlnetImg2ImgPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-controlnet-depth", torch_dtype=torch.float16
        )
        pipeline.enable_model_cpu_offload()

        pipeline.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)

        image_emb, zero_image_emb = pipe_prior(
            prompt,
            image=init_image,
            strength=0.85,
            generator=generator,
            negative_prompt="",
            num_inference_steps=5,
        ).to_tuple()

        generator = torch.Generator(device="cpu").manual_seed(0)
        output = pipeline(
            image=init_image,
            image_embeds=image_emb,
            negative_image_embeds=zero_image_emb,
            hint=hint,
            generator=generator,
            num_inference_steps=5,
            height=512,
            width=512,
            strength=0.5,
            output_type="np",
        )

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        max_diff = numpy_cosine_similarity_distance(expected_image.flatten(), image.flatten())
        assert max_diff < 5e-4
