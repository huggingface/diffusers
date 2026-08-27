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

# This model implementation is heavily inspired by https://github.com/haofanwang/ControlNet-for-Diffusers/

import gc
import random

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.controlnet.pipeline_controlnet import MultiControlNetModel
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    load_numpy,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ..stable_diffusion.ip_adapter_tester import IPAdapterTesterMixin
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class ControlNetImg2ImgPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionControlNetImg2ImgPipeline
    required_input_params_in_call_signature = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width"}
    batch_input_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    output_shape = (3, 64, 64)

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            norm_num_groups=1,
        )
        torch.manual_seed(0)
        controlnet = ControlNetModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
            norm_num_groups=1,
        )
        torch.manual_seed(0)
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[4, 8],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=2,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        return {
            "unet": unet,
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
        }

    def get_dummy_inputs(self):
        # The control image is drawn from the same generator that is handed to the pipeline, so the pipeline sees an
        # already-advanced generator state — keep the order to stay comparable across runs.
        generator = self.get_generator(0)
        controlnet_embedder_scale_factor = 2
        control_image = randn_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            generator=generator,
            device=torch.device(torch_device),
        )
        image = floats_tensor(control_image.shape, rng=random.Random(0)).to(torch_device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "image": image,
            "control_image": control_image,
        }


class TestControlNetImg2ImgPipeline(ControlNetImg2ImgPipelineTesterConfig, PipelineTesterMixin):
    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=2e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_encode_prompt_works_in_isolation(self):
        extra_required_param_value_dict = {
            "device": torch.device(torch_device).type,
            "do_classifier_free_guidance": self.get_dummy_inputs().get("guidance_scale", 1.0) > 1.0,
        }
        super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict)


class TestControlNetImg2ImgPipelineIPAdapter(ControlNetImg2ImgPipelineTesterConfig, IPAdapterTesterMixin):
    """IP-Adapter tests for the Stable Diffusion ControlNet img2img pipeline."""


class TestControlNetImg2ImgPipelineMemory(ControlNetImg2ImgPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the ControlNet img2img pipeline."""


class StableDiffusionMultiControlNetImg2ImgPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionControlNetImg2ImgPipeline
    required_input_params_in_call_signature = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width"}
    batch_input_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    output_shape = (3, 64, 64)

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            norm_num_groups=1,
        )
        torch.manual_seed(0)

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight)
                m.bias.data.fill_(1.0)

        controlnet1 = ControlNetModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
            norm_num_groups=1,
        )
        controlnet1.controlnet_down_blocks.apply(init_weights)

        torch.manual_seed(0)
        controlnet2 = ControlNetModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
            norm_num_groups=1,
        )
        controlnet2.controlnet_down_blocks.apply(init_weights)

        torch.manual_seed(0)
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[4, 8],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=2,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        controlnet = MultiControlNetModel([controlnet1, controlnet2])

        return {
            "unet": unet,
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
        }

    def get_dummy_inputs(self):
        # The control images are drawn from the same generator that is handed to the pipeline, so the pipeline sees
        # an already-advanced generator state — keep the order to stay comparable across runs.
        generator = self.get_generator(0)
        controlnet_embedder_scale_factor = 2
        control_image = [
            randn_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                generator=generator,
                device=torch.device(torch_device),
            ),
            randn_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                generator=generator,
                device=torch.device(torch_device),
            ),
        ]

        image = floats_tensor(control_image[0].shape, rng=random.Random(0)).to(torch_device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "image": image,
            "control_image": control_image,
        }


class TestStableDiffusionMultiControlNetImg2ImgPipeline(
    StableDiffusionMultiControlNetImg2ImgPipelineTesterConfig, PipelineTesterMixin
):
    def test_control_guidance_switch(self):
        pipe = self.get_pipeline().to(torch_device)

        scale = 10.0
        steps = 4

        def run(**extra):
            inputs = self.get_dummy_inputs()
            inputs["num_inference_steps"] = steps
            inputs["controlnet_conditioning_scale"] = scale
            return pipe(**inputs, **extra)[0]

        output_1 = run()
        output_2 = run(control_guidance_start=0.1, control_guidance_end=0.2)
        output_3 = run(control_guidance_start=[0.1, 0.3], control_guidance_end=[0.2, 0.7])
        output_4 = run(control_guidance_start=0.4, control_guidance_end=[0.5, 0.8])

        # make sure that all outputs are different
        assert (output_1 - output_2).abs().sum() > 1e-3
        assert (output_1 - output_3).abs().sum() > 1e-3
        assert (output_1 - output_4).abs().sum() > 1e-3

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=2e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_save_pretrained_raise_not_implemented_exception(self, tmp_path):
        pipe = self.get_pipeline().to(torch_device)
        try:
            # save_pretrained is not implemented for Multi-ControlNet
            pipe.save_pretrained(tmp_path)
        except NotImplementedError:
            pass

    def test_encode_prompt_works_in_isolation(self):
        extra_required_param_value_dict = {
            "device": torch.device(torch_device).type,
            "do_classifier_free_guidance": self.get_dummy_inputs().get("guidance_scale", 1.0) > 1.0,
        }
        super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict)


class TestStableDiffusionMultiControlNetImg2ImgPipelineIPAdapter(
    StableDiffusionMultiControlNetImg2ImgPipelineTesterConfig, IPAdapterTesterMixin
):
    """IP-Adapter tests for the Stable Diffusion Multi-ControlNet img2img pipeline."""


@slow
@require_torch_accelerator
class TestControlNetImg2ImgPipelineSlow:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_canny(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "evil space-punk bird"
        control_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        ).resize((512, 512))
        image = load_image(
            "https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/images/bird.png"
        ).resize((512, 512))

        output = pipe(
            prompt,
            image,
            control_image=control_image,
            generator=generator,
            output_type="np",
            num_inference_steps=50,
            strength=0.6,
        )

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/img2img.npy"
        )

        assert np.abs(expected_image - image).max() < 9e-2
