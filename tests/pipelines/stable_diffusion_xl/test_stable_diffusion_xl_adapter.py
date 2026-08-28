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
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    LCMScheduler,
    MultiAdapter,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
    UNet2DConditionModel,
)

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    floats_tensor,
    load_image,
    nightly,
    numpy_cosine_similarity_distance,
    require_peft_backend,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ..stable_diffusion.ip_adapter_tester import IPAdapterTesterMixin
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


class StableDiffusionXLAdapterPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionXLAdapterPipeline
    required_input_params_in_call_signature = TEXT_GUIDED_IMAGE_VARIATION_PARAMS
    batch_input_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    output_shape = (3, 64, 64)

    def get_dummy_components(self, adapter_type="full_adapter_xl", time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
            time_cond_proj_dim=time_cond_proj_dim,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
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
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        if adapter_type == "full_adapter_xl":
            adapter = T2IAdapter(
                in_channels=3,
                channels=[32, 64],
                num_res_blocks=2,
                downscale_factor=4,
                adapter_type=adapter_type,
            )
        elif adapter_type == "multi_adapter":
            adapter = MultiAdapter(
                [
                    T2IAdapter(
                        in_channels=3,
                        channels=[32, 64],
                        num_res_blocks=2,
                        downscale_factor=4,
                        adapter_type="full_adapter_xl",
                    ),
                    T2IAdapter(
                        in_channels=3,
                        channels=[32, 64],
                        num_res_blocks=2,
                        downscale_factor=4,
                        adapter_type="full_adapter_xl",
                    ),
                ]
            )
        else:
            raise ValueError(
                f"Unknown adapter type: {adapter_type}, must be one of 'full_adapter_xl', or 'multi_adapter''"
            )

        components = {
            "adapter": adapter,
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "feature_extractor": None,
            "image_encoder": None,
        }
        return components

    def get_dummy_components_with_full_downscaling(self, adapter_type="full_adapter_xl"):
        """Get dummy components with x8 VAE downscaling and 3 UNet down blocks.
        These dummy components are intended to fully-exercise the T2I-Adapter
        downscaling behavior.
        """
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=2,
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=1,
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 32, 32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
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
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        if adapter_type == "full_adapter_xl":
            adapter = T2IAdapter(
                in_channels=3,
                channels=[32, 32, 64],
                num_res_blocks=2,
                downscale_factor=16,
                adapter_type=adapter_type,
            )
        elif adapter_type == "multi_adapter":
            adapter = MultiAdapter(
                [
                    T2IAdapter(
                        in_channels=3,
                        channels=[32, 32, 64],
                        num_res_blocks=2,
                        downscale_factor=16,
                        adapter_type="full_adapter_xl",
                    ),
                    T2IAdapter(
                        in_channels=3,
                        channels=[32, 32, 64],
                        num_res_blocks=2,
                        downscale_factor=16,
                        adapter_type="full_adapter_xl",
                    ),
                ]
            )
        else:
            raise ValueError(
                f"Unknown adapter type: {adapter_type}, must be one of 'full_adapter_xl', or 'multi_adapter''"
            )

        components = {
            "adapter": adapter,
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "feature_extractor": None,
            "image_encoder": None,
        }
        return components

    def get_dummy_inputs(self, seed=0, height=64, width=64, num_images=1):
        if num_images == 1:
            image = floats_tensor((1, 3, height, width), rng=random.Random(seed)).to(torch_device)
        else:
            image = [
                floats_tensor((1, 3, height, width), rng=random.Random(seed)).to(torch_device)
                for _ in range(num_images)
            ]

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": self.get_generator(seed),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }
        return inputs


class TestStableDiffusionXLAdapterPipeline(StableDiffusionXLAdapterPipelineTesterConfig, PipelineTesterMixin):
    @pytest.mark.skip("Every `_optional_component` is needed to encode the prompt this pipeline requires.")
    def test_save_load_optional_components(self):
        pass

    def test_stable_diffusion_adapter_default_case(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 64, 64)

        # fmt: off
        expected_slice = torch.tensor([0.6002, 0.6262, 0.4981, 0.5304, 0.5774, 0.4685, 0.5228, 0.5208, 0.4938])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=5e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    @pytest.mark.parametrize(
        "dim",
        [
            # (dim=144) The internal feature map will be 9x9 after initial pixel unshuffling (downscaled x16).
            ((4 * 2 + 1) * 16),
            # (dim=160) The internal feature map will be 5x5 after the first T2I down block (downscaled x32).
            ((4 * 1 + 1) * 32),
        ],
    )
    def test_multiple_image_dimensions(self, dim):
        """Test that the T2I-Adapter pipeline supports any input dimension that
        is divisible by the adapter's `downscale_factor`. This test was added in
        response to an issue where the T2I Adapter's downscaling padding
        behavior did not match the UNet's behavior.

        Note that we have selected `dim` values to produce odd resolutions at
        each downscaling level.
        """
        sd_pipe = self.get_pipeline(**self.get_dummy_components_with_full_downscaling()).to(torch_device)

        image = sd_pipe(**self.get_dummy_inputs(height=dim, width=dim)).images

        assert image.shape == (1, 3, dim, dim)

    @pytest.mark.parametrize("adapter_type", ["full_adapter", "full_adapter_xl", "light_adapter"])
    def test_total_downscale_factor(self, adapter_type):
        """Test that the T2IAdapter correctly reports its total_downscale_factor."""
        batch_size = 1
        in_channels = 3
        out_channels = [320, 640, 1280, 1280]
        in_image_size = 512

        adapter = T2IAdapter(
            in_channels=in_channels,
            channels=out_channels,
            num_res_blocks=2,
            downscale_factor=8,
            adapter_type=adapter_type,
        )
        adapter.to(torch_device)

        in_image = floats_tensor((batch_size, in_channels, in_image_size, in_image_size)).to(torch_device)

        adapter_state = adapter(in_image)

        # Assume that the last element in `adapter_state` has been downsampled the most, and check
        # that it matches the `total_downscale_factor`.
        expected_out_image_size = in_image_size // adapter.total_downscale_factor
        assert adapter_state[-1].shape == (
            batch_size,
            out_channels[-1],
            expected_out_image_size,
            expected_out_image_size,
        )

    def test_adapter_sdxl_lcm(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline(**self.get_dummy_components(time_cond_proj_dim=256))
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 64, 64)

        # fmt: off
        expected_slice = torch.tensor([0.5425, 0.5385, 0.4964, 0.5045, 0.6149, 0.4974, 0.5469, 0.5332, 0.5426])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_adapter_sdxl_lcm_custom_timesteps(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline(**self.get_dummy_components(time_cond_proj_dim=256))
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)

        inputs = self.get_dummy_inputs()
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        image = sd_pipe(**inputs).images
        assert image.shape == (1, 3, 64, 64)

        # Custom timesteps matching the default 2-step schedule reproduce `test_adapter_sdxl_lcm`'s output.
        # fmt: off
        expected_slice = torch.tensor([0.5425, 0.5385, 0.4964, 0.5045, 0.6149, 0.4974, 0.5469, 0.5332, 0.5426])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)


class TestStableDiffusionXLAdapterPipelineMemory(StableDiffusionXLAdapterPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SDXL adapter pipeline."""


class TestStableDiffusionXLAdapterPipelineIPAdapter(
    StableDiffusionXLAdapterPipelineTesterConfig, IPAdapterTesterMixin
):
    """IP-Adapter tests for the SDXL adapter pipeline."""


class StableDiffusionXLMultiAdapterPipelineTesterConfig(StableDiffusionXLAdapterPipelineTesterConfig):
    """Same pipeline, driven by a `MultiAdapter` — so `image` is a list of one image per adapter."""

    def get_dummy_components(self, time_cond_proj_dim=None):
        return super().get_dummy_components("multi_adapter", time_cond_proj_dim=time_cond_proj_dim)

    def get_dummy_components_with_full_downscaling(self):
        return super().get_dummy_components_with_full_downscaling("multi_adapter")

    def get_dummy_inputs(self, seed=0, height=64, width=64):
        inputs = super().get_dummy_inputs(seed, height, width, num_images=2)
        inputs["adapter_conditioning_scale"] = [0.5, 0.5]
        return inputs

    def batchify_inputs(self, inputs, batch_size):
        """Batch the standard inputs, keeping `image` as one list of `batch_size` images per adapter."""
        batched_inputs = dict(inputs)
        for name in self.batch_input_params:
            if name not in inputs:
                continue

            value = inputs[name]
            if name == "prompt":
                len_prompt = len(value)
                # make unequal batch sizes
                batched_inputs[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]
                # make last batch super long
                batched_inputs[name][-1] = 100 * "very long"
            elif name == "image":
                batched_inputs[name] = [batch_size * [image] for image in value]
            else:
                batched_inputs[name] = batch_size * [value]

        return batched_inputs


class TestStableDiffusionXLMultiAdapterPipeline(
    StableDiffusionXLMultiAdapterPipelineTesterConfig, PipelineTesterMixin
):
    def test_stable_diffusion_adapter_default_case(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 64, 64)

        # fmt: off
        expected_slice = torch.tensor([0.6114, 0.6256, 0.4972, 0.5219, 0.5668, 0.4658, 0.5210, 0.5188, 0.4908])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=5e-3)

    def test_inference_batch_consistent(self, batch_sizes=[2], batch_generator=True):
        # `image` holds one conditioning image per adapter, so it needs the multi-adapter batching.
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        for batch_size in batch_sizes:
            batched_inputs = self.batchify_inputs(inputs, batch_size)
            if batch_generator:
                batched_inputs["generator"] = [self.get_generator(i) for i in range(batch_size)]

            output = pipe(**batched_inputs)
            assert len(output[0]) == batch_size

    def test_inference_batch_single_identical(
        self, batch_size=3, expected_max_diff=2e-3, additional_params_copy_to_batched_inputs=["num_inference_steps"]
    ):
        # `image` holds one conditioning image per adapter, so it needs the multi-adapter batching.
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        batched_inputs = self.batchify_inputs(inputs, batch_size)
        batched_inputs["generator"] = [self.get_generator(i) for i in range(batch_size)]
        for arg in additional_params_copy_to_batched_inputs:
            batched_inputs[arg] = inputs[arg]

        output_batch = pipe(**batched_inputs)
        assert output_batch[0].shape[0] == batch_size

        inputs["generator"] = self.get_generator(0)
        output = pipe(**inputs)

        assert_tensors_close(
            output_batch[0][0], output[0][0], atol=expected_max_diff, msg="Batched output differs from single."
        )

    def test_num_images_per_prompt(self):
        # `image` holds one conditioning image per adapter, so it needs the multi-adapter batching.
        pipe = self.get_pipeline().to(torch_device)

        for batch_size in [1, 2]:
            for num_images_per_prompt in [1, 2]:
                inputs = self.batchify_inputs(self.get_dummy_inputs(), batch_size)
                # `test_inference_batch_*` makes the prompts unequal on purpose; here they just get duplicated.
                inputs["prompt"] = batch_size * [self.get_dummy_inputs()["prompt"]]

                images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt)[0]

                assert images.shape[0] == batch_size * num_images_per_prompt

    def test_adapter_sdxl_lcm(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline(**self.get_dummy_components(time_cond_proj_dim=256))
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 64, 64)

        # fmt: off
        expected_slice = torch.tensor([0.5313, 0.5375, 0.4942, 0.5021, 0.6142, 0.4968, 0.5434, 0.5311, 0.5448])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_adapter_sdxl_lcm_custom_timesteps(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline(**self.get_dummy_components(time_cond_proj_dim=256))
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)

        inputs = self.get_dummy_inputs()
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        image = sd_pipe(**inputs).images
        assert image.shape == (1, 3, 64, 64)

        # Custom timesteps matching the default 2-step schedule reproduce `test_adapter_sdxl_lcm`'s output.
        # fmt: off
        expected_slice = torch.tensor([0.5313, 0.5375, 0.4942, 0.5021, 0.6142, 0.4968, 0.5434, 0.5311, 0.5448])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    @pytest.mark.skip("Every `_optional_component` is needed to encode the prompt this pipeline requires.")
    def test_save_load_optional_components(self):
        pass


class TestStableDiffusionXLMultiAdapterPipelineMemory(
    StableDiffusionXLMultiAdapterPipelineTesterConfig, MemoryTesterMixin
):
    """Memory optimization tests for the SDXL pipeline driven by a `MultiAdapter`."""


class TestStableDiffusionXLMultiAdapterPipelineIPAdapter(
    StableDiffusionXLMultiAdapterPipelineTesterConfig, IPAdapterTesterMixin
):
    """IP-Adapter tests for the SDXL pipeline driven by a `MultiAdapter`."""


@slow
@nightly
@require_torch_accelerator
@require_peft_backend
class TestStableDiffusionXLAdapterLoRAIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_sdxl_t2i_adapter_canny_lora(self):
        adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-lineart-sdxl-1.0", torch_dtype=torch.float16).to(
            "cpu"
        )
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            adapter=adapter,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors")
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "toy"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/toy_canny.png"
        )

        images = pipe(prompt, image=image, generator=generator, output_type="np", num_inference_steps=3).images

        assert images[0].shape == (768, 512, 3)

        image_slice = images[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.4284, 0.4337, 0.4319, 0.4255, 0.4329, 0.4280, 0.4338, 0.4420, 0.4226])
        assert numpy_cosine_similarity_distance(image_slice, expected_slice) < 1e-4
