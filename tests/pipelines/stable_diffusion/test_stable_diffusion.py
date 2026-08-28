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
import time

import numpy as np
import pytest
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
)

from diffusers import (
    AutoencoderKL,
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    logging,
)
from diffusers.utils.import_utils import is_accelerate_available

from ...models.testing_utils.lora import check_if_lora_correctly_set
from ...testing_utils import (
    CaptureLogger,
    Expectations,
    assert_tensors_close,
    backend_empty_cache,
    backend_max_memory_allocated,
    backend_reset_max_memory_allocated,
    backend_reset_peak_memory_stats,
    load_image,
    load_numpy,
    nightly,
    numpy_cosine_similarity_distance,
    require_accelerate_version_greater,
    require_peft_backend,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    skip_mps,
    slow,
    torch_device,
)
from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
    UNetLoraTesterMixin,
)
from .ip_adapter_tester import IPAdapterTesterMixin


if is_accelerate_available():
    from accelerate.utils import release_memory


class StableDiffusionPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS
    output_shape = (3, 64, 64)
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS

    def get_dummy_components(self, time_cond_proj_dim=None):
        cross_attention_dim = 8

        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=1,
            sample_size=32,
            time_cond_proj_dim=time_cond_proj_dim,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=cross_attention_dim,
            norm_num_groups=2,
        )
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
            hidden_size=cross_attention_dim,
            intermediate_size=16,
            layer_norm_eps=1e-05,
            num_attention_heads=2,
            num_hidden_layers=2,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
        }
        return components

    def get_dummy_inputs(self):
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }
        return inputs


class TestStableDiffusionPipeline(StableDiffusionPipelineTesterConfig, PipelineTesterMixin):
    def test_stable_diffusion_ddim(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 64, 64)

        # fmt: off
        expected_slice = torch.tensor([0.1641, 0.4640, 0.4864, 0.2683, 0.3652, 0.4507, 0.5290, 0.3307, 0.3977])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_lcm(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline(**self.get_dummy_components(time_cond_proj_dim=256))
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 64, 64)

        # fmt: off
        expected_slice = torch.tensor([0.2368, 0.4900, 0.5019, 0.2723, 0.4473, 0.4578, 0.4551, 0.3532, 0.4133])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_lcm_custom_timesteps(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline(**self.get_dummy_components(time_cond_proj_dim=256))
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)

        inputs = self.get_dummy_inputs()
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        image = sd_pipe(**inputs).images
        assert image.shape == (1, 3, 64, 64)

        # Custom timesteps matching the default 2-step schedule reproduce `test_stable_diffusion_lcm`'s output.
        # fmt: off
        expected_slice = torch.tensor([0.2368, 0.4900, 0.5019, 0.2723, 0.4473, 0.4578, 0.4551, 0.3532, 0.4133])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_ays(self):
        from diffusers.schedulers import AysSchedules

        timestep_schedule = AysSchedules["StableDiffusionTimesteps"]
        sigma_schedule = AysSchedules["StableDiffusionSigmas"]

        sd_pipe = self.get_pipeline(**self.get_dummy_components(time_cond_proj_dim=256)).to(torch_device)
        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)

        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 10
        output = sd_pipe(**inputs).images

        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = None
        inputs["timesteps"] = timestep_schedule
        output_ts = sd_pipe(**inputs).images

        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = None
        inputs["sigmas"] = sigma_schedule
        output_sigmas = sd_pipe(**inputs).images

        assert (output_sigmas - output_ts).abs().max() < 1e-3, (
            "ays timesteps and ays sigmas should have the same outputs"
        )
        assert (output - output_ts).abs().max() > 1e-3, "use ays timesteps should have different outputs"
        assert (output - output_sigmas).abs().max() > 1e-3, "use ays sigmas should have different outputs"

    def test_stable_diffusion_prompt_embeds(self):
        sd_pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = 3 * [inputs["prompt"]]
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -1, -3:, -3:]

        inputs = self.get_dummy_inputs()
        prompt = 3 * [inputs.pop("prompt")]

        text_inputs = sd_pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=sd_pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = text_inputs["input_ids"].to(torch_device)
        inputs["prompt_embeds"] = sd_pipe.text_encoder(text_inputs)[0]

        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -1, -3:, -3:]

        assert_tensors_close(
            image_slice_2, image_slice_1, atol=1e-4, msg="Passing `prompt_embeds` changed the output."
        )

    def test_stable_diffusion_negative_prompt_embeds(self):
        sd_pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        negative_prompt = 3 * ["this is a negative prompt"]
        inputs["negative_prompt"] = negative_prompt
        inputs["prompt"] = 3 * [inputs["prompt"]]
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -1, -3:, -3:]

        inputs = self.get_dummy_inputs()
        prompt = 3 * [inputs.pop("prompt")]

        embeds = []
        for p in [prompt, negative_prompt]:
            text_inputs = sd_pipe.tokenizer(
                p,
                padding="max_length",
                max_length=sd_pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = text_inputs["input_ids"].to(torch_device)
            embeds.append(sd_pipe.text_encoder(text_inputs)[0])

        inputs["prompt_embeds"], inputs["negative_prompt_embeds"] = embeds

        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -1, -3:, -3:]

        assert_tensors_close(
            image_slice_2, image_slice_1, atol=1e-4, msg="Passing `negative_prompt_embeds` changed the output."
        )

    def test_stable_diffusion_ddim_factor_8(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()

        image = sd_pipe(**self.get_dummy_inputs(), height=136, width=136).images
        assert image.shape == (1, 3, 136, 136)

        # fmt: off
        expected_slice = torch.tensor([0.4587, 0.5238, 0.5006, 0.3869, 0.4538, 0.4228, 0.5666, 0.5814, 0.5468])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_pndm(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()
        sd_pipe.scheduler = PNDMScheduler(skip_prk_steps=True)

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 64, 64)

        # fmt: off
        expected_slice = torch.tensor([0.1791, 0.4611, 0.4741, 0.2336, 0.4175, 0.4484, 0.5582, 0.3556, 0.3951])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_k_lms(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 64, 64)

        # fmt: off
        expected_slice = torch.tensor([0.2185, 0.4538, 0.4660, 0.2454, 0.4408, 0.4377, 0.5629, 0.3754, 0.3927])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_k_euler_ancestral(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()
        sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_pipe.scheduler.config)

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 64, 64)

        # fmt: off
        expected_slice = torch.tensor([0.2185, 0.4534, 0.4657, 0.2452, 0.4406, 0.4375, 0.5631, 0.3755, 0.3927])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_k_euler(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()
        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 64, 64)

        # fmt: off
        expected_slice = torch.tensor([0.2185, 0.4538, 0.4660, 0.2454, 0.4408, 0.4377, 0.5629, 0.3754, 0.3927])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_negative_prompt(self):
        # Run on CPU: the expected slice below is CPU-specific.
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler(skip_prk_steps=True)
        sd_pipe = self.get_pipeline(**components)

        image = sd_pipe(**self.get_dummy_inputs(), negative_prompt="french fries").images
        assert image.shape == (1, 3, 64, 64)

        # fmt: off
        expected_slice = torch.tensor([0.1772, 0.4578, 0.4700, 0.2363, 0.4173, 0.4462, 0.5595, 0.3588, 0.3959])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_no_safety_checker(self, tmp_path):
        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
        )
        assert isinstance(pipe, StableDiffusionPipeline)
        assert isinstance(pipe.scheduler, PNDMScheduler)
        assert pipe.safety_checker is None

        image = pipe("example prompt", num_inference_steps=2).images[0]
        assert image is not None

        # check that there's no error when saving a pipeline with one of the models being None
        pipe.save_pretrained(tmp_path)
        pipe = StableDiffusionPipeline.from_pretrained(tmp_path)

        # sanity check that the pipeline still works
        assert pipe.safety_checker is None
        image = pipe("example prompt", num_inference_steps=2).images[0]
        assert image is not None

    def test_stable_diffusion_vae_slicing(self):
        # Run on CPU: sliced VAE decoding is compared against a full-batch decode of the same run.
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = self.get_pipeline(**components)

        image_count = 4

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * image_count
        output_1 = sd_pipe(**inputs)

        # make sure sliced vae decode yields the same result
        sd_pipe.vae.enable_slicing()
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * image_count
        output_2 = sd_pipe(**inputs)

        # there is a small discrepancy at image borders vs. full batch decode
        assert (output_2.images - output_1.images).abs().max() < 3e-3

    def test_stable_diffusion_vae_tiling(self, base_pipe_output):
        sd_pipe = self.get_pipeline().to(torch_device)

        # Tiled decode should yield the same result as the non-tiled decode cached by `base_pipe_output`.
        sd_pipe.vae.enable_tiling()
        torch.manual_seed(0)
        output_tiled = sd_pipe(**self.get_dummy_inputs()).images

        assert (output_tiled - base_pipe_output).abs().max() < 5e-1

        # test that tiled decode works with various shapes
        for shape in [(1, 4, 73, 97), (1, 4, 97, 73), (1, 4, 49, 65), (1, 4, 65, 49)]:
            sd_pipe.vae.decode(torch.zeros(shape, device=torch_device))

    def test_stable_diffusion_long_prompt(self):
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = self.get_pipeline(**components).to(torch_device)

        do_classifier_free_guidance = True
        negative_prompt = None
        num_images_per_prompt = 1
        logger = logging.get_logger("diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion")
        logger.setLevel(logging.WARNING)

        prompt = 100 * "@"
        with CaptureLogger(logger) as cap_logger:
            negative_text_embeddings, text_embeddings = sd_pipe.encode_prompt(
                prompt, torch_device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
            if negative_text_embeddings is not None:
                text_embeddings = torch.cat([negative_text_embeddings, text_embeddings])

        # 100 - 77 + 1 (BOS token) + 1 (EOS token) = 25
        assert cap_logger.out.count("@") == 25

        negative_prompt = "Hello"
        with CaptureLogger(logger) as cap_logger_2:
            negative_text_embeddings_2, text_embeddings_2 = sd_pipe.encode_prompt(
                prompt, torch_device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
            if negative_text_embeddings_2 is not None:
                text_embeddings_2 = torch.cat([negative_text_embeddings_2, text_embeddings_2])

        assert cap_logger.out == cap_logger_2.out

        prompt = 25 * "@"
        with CaptureLogger(logger) as cap_logger_3:
            negative_text_embeddings_3, text_embeddings_3 = sd_pipe.encode_prompt(
                prompt, torch_device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
            if negative_text_embeddings_3 is not None:
                text_embeddings_3 = torch.cat([negative_text_embeddings_3, text_embeddings_3])

        assert text_embeddings_3.shape == text_embeddings_2.shape == text_embeddings.shape
        assert text_embeddings.shape[1] == 77
        assert cap_logger_3.out == ""

    def test_stable_diffusion_height_width_opt(self):
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = self.get_pipeline(**components).to(torch_device)

        prompt = "hey"

        output = sd_pipe(prompt, num_inference_steps=1, output_type="pt")
        assert output.images[0].shape[-2:] == (64, 64)

        output = sd_pipe(prompt, num_inference_steps=1, height=96, width=96, output_type="pt")
        assert output.images[0].shape[-2:] == (96, 96)

        config = dict(sd_pipe.unet.config)
        config["sample_size"] = 96
        sd_pipe.unet = UNet2DConditionModel.from_config(config).to(torch_device)
        output = sd_pipe(prompt, num_inference_steps=1, output_type="pt")
        assert output.images[0].shape[-2:] == (192, 192)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    # MPS currently doesn't support ComplexFloats, which are required for freeU - see https://github.com/huggingface/diffusers/issues/7569.
    @skip_mps
    def test_freeu_enabled(self, base_pipe_output):
        sd_pipe = self.get_pipeline().to(torch_device)

        sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        torch.manual_seed(0)
        output_freeu = sd_pipe(**self.get_dummy_inputs()).images

        assert not torch.allclose(base_pipe_output[0, -1, -3:, -3:], output_freeu[0, -1, -3:, -3:]), (
            "Enabling of FreeU should lead to different results."
        )

    def test_freeu_disabled(self, base_pipe_output):
        sd_pipe = self.get_pipeline().to(torch_device)

        sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        sd_pipe.disable_freeu()

        for upsample_block in sd_pipe.unet.up_blocks:
            for key in {"s1", "s2", "b1", "b2"}:
                assert getattr(upsample_block, key) is None, f"Disabling of FreeU should have set {key} to None."

        torch.manual_seed(0)
        output_no_freeu = sd_pipe(**self.get_dummy_inputs()).images

        assert torch.allclose(base_pipe_output[0, -1, -3:, -3:], output_no_freeu[0, -1, -3:, -3:]), (
            "Disabling of FreeU should lead to results similar to the default pipeline results."
        )

    def test_fused_qkv_projections(self, base_pipe_output):
        # The unfused reference is the class-cached `base_pipe_output`, so the runs below reseed the global RNG to
        # reproduce it and the only remaining difference comes from the projection fusion itself.
        sd_pipe = self.get_pipeline().to(torch_device)

        original_image_slice = base_pipe_output[0, -1, -3:, -3:]

        sd_pipe.fuse_qkv_projections()
        torch.manual_seed(0)
        image_slice_fused = sd_pipe(**self.get_dummy_inputs()).images[0, -1, -3:, -3:]

        sd_pipe.unfuse_qkv_projections()
        torch.manual_seed(0)
        image_slice_disabled = sd_pipe(**self.get_dummy_inputs()).images[0, -1, -3:, -3:]

        assert torch.allclose(original_image_slice, image_slice_fused, atol=1e-2, rtol=1e-2), (
            "Fusion of QKV projections shouldn't affect the outputs."
        )
        assert torch.allclose(image_slice_fused, image_slice_disabled, atol=1e-2, rtol=1e-2), (
            "Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled."
        )
        assert torch.allclose(original_image_slice, image_slice_disabled, atol=1e-2, rtol=1e-2), (
            "Original outputs should match when fused QKV projections are disabled."
        )

    def test_pipeline_interrupt(self):
        sd_pipe = self.get_pipeline().to(torch_device)

        prompt = "hey"
        num_inference_steps = 3

        # store intermediate latents from the generation process
        class PipelineState:
            def __init__(self):
                self.state = []

            def apply(self, pipe, i, t, callback_kwargs):
                self.state.append(callback_kwargs["latents"])
                return callback_kwargs

        pipe_state = PipelineState()
        sd_pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            output_type="pt",
            generator=torch.Generator("cpu").manual_seed(0),
            callback_on_step_end=pipe_state.apply,
        ).images

        # interrupt generation at step index
        interrupt_step_idx = 1

        def callback_on_step_end(pipe, i, t, callback_kwargs):
            if i == interrupt_step_idx:
                pipe._interrupt = True

            return callback_kwargs

        output_interrupted = sd_pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            output_type="latent",
            generator=torch.Generator("cpu").manual_seed(0),
            callback_on_step_end=callback_on_step_end,
        ).images

        # fetch intermediate latents at the interrupted step
        # from the completed generation process
        intermediate_latent = pipe_state.state[interrupt_step_idx]

        # compare the intermediate latent to the output of the interrupted process
        # they should be the same
        assert torch.allclose(intermediate_latent, output_interrupted, atol=1e-4)

    def test_pipeline_accept_tuple_type_unet_sample_size(self):
        # the purpose of this test is to see whether the pipeline would accept a unet with the tuple-typed sample size
        sd_repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        sample_size = [60, 80]
        customised_unet = UNet2DConditionModel(sample_size=sample_size)
        pipe = StableDiffusionPipeline.from_pretrained(sd_repo_id, unet=customised_unet)
        assert pipe.unet.config.sample_size == sample_size

    def test_encode_prompt_works_in_isolation(self):
        extra_required_param_value_dict = {
            "device": torch.device(torch_device).type,
            "do_classifier_free_guidance": self.get_dummy_inputs().get("guidance_scale", 1.0) > 1.0,
        }
        return super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict)


class TestStableDiffusionPipelineMemory(StableDiffusionPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Stable Diffusion pipeline."""


class TestStableDiffusionPipelineLoRA(StableDiffusionPipelineTesterConfig, LoraTesterMixin, UNetLoraTesterMixin):
    """LoRA tests for the Stable Diffusion pipeline."""


class TestStableDiffusionPipelineLoRAMemory(StableDiffusionPipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests for the Stable Diffusion pipeline."""


class TestStableDiffusionPipelineIPAdapter(StableDiffusionPipelineTesterConfig, IPAdapterTesterMixin):
    """IP-Adapter tests for the Stable Diffusion pipeline."""


@nightly
@require_torch_accelerator
class TestStableDiffusionPipelineIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_1_1_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-1")
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.4363, 0.4355, 0.3667, 0.4066, 0.3970, 0.3866, 0.4394, 0.4356, 0.4059])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_stable_diffusion_v1_4_with_freeu(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 25

        sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        image = sd_pipe(**inputs).images
        image = image[0, -3:, -3:, -1].flatten()
        expected_image = [0.0721, 0.0588, 0.0268, 0.0384, 0.0636, 0.0, 0.0429, 0.0344, 0.0309]
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_1_4_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.5740, 0.4784, 0.3162, 0.6358, 0.5831, 0.5505, 0.5082, 0.5631, 0.5575])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_stable_diffusion_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.38019, 0.28647, 0.27321, 0.40377, 0.38290, 0.35446, 0.39218, 0.38165, 0.42239])
        assert np.abs(image_slice - expected_slice).max() < 1e-4

    def test_stable_diffusion_lms(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.10542, 0.09620, 0.07332, 0.09015, 0.09382, 0.07597, 0.08496, 0.07806, 0.06455])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_stable_diffusion_dpm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            sd_pipe.scheduler.config,
            final_sigmas_type="sigma_min",
        )
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.03503, 0.03494, 0.01087, 0.03128, 0.02552, 0.00803, 0.00742, 0.00372, 0.00000])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_stable_diffusion_attention_slicing(self):
        backend_reset_peak_memory_stats(torch_device)
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe.unet.set_default_attn_processor()
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # enable attention slicing
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image_sliced = pipe(**inputs).images

        mem_bytes = backend_max_memory_allocated(torch_device)
        backend_reset_peak_memory_stats(torch_device)
        # make sure that less than 3.75 GB is allocated
        assert mem_bytes < 3.75 * 10**9

        # disable slicing
        pipe.disable_attention_slicing()
        pipe.unet.set_default_attn_processor()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image = pipe(**inputs).images

        # make sure that more than 3.75 GB is allocated
        mem_bytes = backend_max_memory_allocated(torch_device)
        assert mem_bytes > 3.75 * 10**9
        max_diff = numpy_cosine_similarity_distance(image_sliced.flatten(), image.flatten())
        assert max_diff < 1e-3

    def test_stable_diffusion_vae_slicing(self):
        backend_reset_peak_memory_stats(torch_device)
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        # enable vae slicing
        pipe.vae.enable_slicing()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        inputs["prompt"] = [inputs["prompt"]] * 4
        inputs["latents"] = torch.cat([inputs["latents"]] * 4)
        image_sliced = pipe(**inputs).images

        mem_bytes = backend_max_memory_allocated(torch_device)
        backend_reset_peak_memory_stats(torch_device)
        # make sure that less than 4 GB is allocated
        assert mem_bytes < 4e9

        # disable vae slicing
        pipe.vae.disable_slicing()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        inputs["prompt"] = [inputs["prompt"]] * 4
        inputs["latents"] = torch.cat([inputs["latents"]] * 4)
        image = pipe(**inputs).images

        # make sure that more than 4 GB is allocated
        mem_bytes = backend_max_memory_allocated(torch_device)
        assert mem_bytes > 4e9
        # There is a small discrepancy at the image borders vs. a fully batched version.
        max_diff = numpy_cosine_similarity_distance(image_sliced.flatten(), image.flatten())
        assert max_diff < 1e-2

    def test_stable_diffusion_vae_tiling(self):
        backend_reset_peak_memory_stats(torch_device)
        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, variant="fp16", torch_dtype=torch.float16, safety_checker=None
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae = pipe.vae.to(memory_format=torch.channels_last)

        prompt = "a photograph of an astronaut riding a horse"

        # enable vae tiling
        pipe.vae.enable_tiling()
        pipe.enable_model_cpu_offload(device=torch_device)
        generator = torch.Generator(device="cpu").manual_seed(0)
        output_chunked = pipe(
            [prompt],
            width=1024,
            height=1024,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="np",
        )
        image_chunked = output_chunked.images

        mem_bytes = backend_max_memory_allocated(torch_device)

        # disable vae tiling
        pipe.vae.disable_tiling()
        generator = torch.Generator(device="cpu").manual_seed(0)
        output = pipe(
            [prompt],
            width=1024,
            height=1024,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="np",
        )
        image = output.images

        assert mem_bytes < 1e10
        max_diff = numpy_cosine_similarity_distance(image_chunked.flatten(), image.flatten())
        assert max_diff < 1e-2

    def test_stable_diffusion_fp16_vs_autocast(self):
        # this test makes sure that the original model with autocast
        # and the new model with fp16 yield the same result
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image_fp16 = pipe(**inputs).images

        with torch.autocast(torch_device):
            inputs = self.get_inputs(torch_device)
            image_autocast = pipe(**inputs).images

        # Make sure results are close enough
        diff = np.abs(image_fp16.flatten() - image_autocast.flatten())
        # They ARE different since ops are not run always at the same precision
        # however, they should be extremely close.
        assert diff.mean() < 2e-2

    def test_stable_diffusion_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: torch.Tensor) -> None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.5693, -0.3018, -0.9746, 0.0518, -0.8770, 0.7559, -1.7402, 0.1022, 1.1582]
                )

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.1958, -0.2993, -1.0166, -0.5005, -0.4810, 0.6162, -0.9492, 0.6621, 1.4492]
                )

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2

        callback_fn.has_been_called = False

        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == inputs["num_inference_steps"]

    def test_stable_diffusion_low_cpu_mem_usage(self):
        pipeline_id = "CompVis/stable-diffusion-v1-4"

        start_time = time.time()
        pipeline_low_cpu_mem_usage = StableDiffusionPipeline.from_pretrained(pipeline_id, torch_dtype=torch.float16)
        pipeline_low_cpu_mem_usage.to(torch_device)
        low_cpu_mem_usage_time = time.time() - start_time

        start_time = time.time()
        _ = StableDiffusionPipeline.from_pretrained(pipeline_id, torch_dtype=torch.float16, low_cpu_mem_usage=False)
        normal_load_time = time.time() - start_time

        assert 2 * low_cpu_mem_usage_time < normal_load_time

    def test_stable_diffusion_pipeline_with_sequential_cpu_offloading(self):
        backend_empty_cache(torch_device)
        backend_reset_max_memory_allocated(torch_device)
        backend_reset_peak_memory_stats(torch_device)

        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload(device=torch_device)

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        _ = pipe(**inputs)

        mem_bytes = backend_max_memory_allocated(torch_device)
        # make sure that less than 2.8 GB is allocated
        assert mem_bytes < 2.8 * 10**9

    def test_stable_diffusion_pipeline_with_model_offloading(self):
        backend_empty_cache(torch_device)
        backend_reset_peak_memory_stats(torch_device)

        inputs = self.get_inputs(torch_device, dtype=torch.float16)

        # Normal inference

        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
        )
        pipe.unet.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        outputs = pipe(**inputs)
        mem_bytes = backend_max_memory_allocated(torch_device)

        # With model offloading

        # Reload but don't move to cuda
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
        )
        pipe.unet.set_default_attn_processor()

        backend_empty_cache(torch_device)
        backend_reset_max_memory_allocated(torch_device)
        backend_reset_peak_memory_stats(torch_device)

        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs(torch_device, dtype=torch.float16)

        outputs_offloaded = pipe(**inputs)
        mem_bytes_offloaded = backend_max_memory_allocated(torch_device)

        images = outputs.images
        offloaded_images = outputs_offloaded.images

        max_diff = numpy_cosine_similarity_distance(images.flatten(), offloaded_images.flatten())
        assert max_diff < 1e-3
        assert mem_bytes_offloaded < mem_bytes
        assert mem_bytes_offloaded < 3.5 * 10**9
        for module in pipe.text_encoder, pipe.unet, pipe.vae:
            assert module.device == torch.device("cpu")

        # With attention slicing
        backend_empty_cache(torch_device)
        backend_reset_max_memory_allocated(torch_device)
        backend_reset_peak_memory_stats(torch_device)

        pipe.enable_attention_slicing()
        _ = pipe(**inputs)
        mem_bytes_slicing = backend_max_memory_allocated(torch_device)

        assert mem_bytes_slicing < mem_bytes_offloaded
        assert mem_bytes_slicing < 3 * 10**9

    def test_stable_diffusion_textual_inversion(self):
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe.load_textual_inversion("sd-concepts-library/low-poly-hd-logos-icons")

        a111_file = hf_hub_download("hf-internal-testing/text_inv_embedding_a1111_format", "winter_style.pt")
        a111_file_neg = hf_hub_download(
            "hf-internal-testing/text_inv_embedding_a1111_format", "winter_style_negative.pt"
        )
        pipe.load_textual_inversion(a111_file)
        pipe.load_textual_inversion(a111_file_neg)
        pipe.to(torch_device)

        generator = torch.Generator(device="cpu").manual_seed(1)

        prompt = "An logo of a turtle in strong Style-Winter with <low-poly-hd-logos-icons>"
        neg_prompt = "Style-Winter-neg"

        image = pipe(prompt=prompt, negative_prompt=neg_prompt, generator=generator, output_type="np").images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text_inv/winter_logo_style.npy"
        )

        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 8e-1

    def test_stable_diffusion_textual_inversion_with_model_cpu_offload(self):
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.load_textual_inversion("sd-concepts-library/low-poly-hd-logos-icons")

        a111_file = hf_hub_download("hf-internal-testing/text_inv_embedding_a1111_format", "winter_style.pt")
        a111_file_neg = hf_hub_download(
            "hf-internal-testing/text_inv_embedding_a1111_format", "winter_style_negative.pt"
        )
        pipe.load_textual_inversion(a111_file)
        pipe.load_textual_inversion(a111_file_neg)

        generator = torch.Generator(device="cpu").manual_seed(1)

        prompt = "An logo of a turtle in strong Style-Winter with <low-poly-hd-logos-icons>"
        neg_prompt = "Style-Winter-neg"

        image = pipe(prompt=prompt, negative_prompt=neg_prompt, generator=generator, output_type="np").images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text_inv/winter_logo_style.npy"
        )

        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 8e-1

    def test_stable_diffusion_textual_inversion_with_sequential_cpu_offload(self):
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe.enable_sequential_cpu_offload(device=torch_device)
        pipe.load_textual_inversion("sd-concepts-library/low-poly-hd-logos-icons").to(torch_device)

        a111_file = hf_hub_download("hf-internal-testing/text_inv_embedding_a1111_format", "winter_style.pt")
        a111_file_neg = hf_hub_download(
            "hf-internal-testing/text_inv_embedding_a1111_format", "winter_style_negative.pt"
        )
        pipe.load_textual_inversion(a111_file)
        pipe.load_textual_inversion(a111_file_neg)

        generator = torch.Generator(device="cpu").manual_seed(1)

        prompt = "An logo of a turtle in strong Style-Winter with <low-poly-hd-logos-icons>"
        neg_prompt = "Style-Winter-neg"

        image = pipe(prompt=prompt, negative_prompt=neg_prompt, generator=generator, output_type="np").images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text_inv/winter_logo_style.npy"
        )

        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 8e-1

    def test_stable_diffusion_1_4_pndm_reference_image(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 50
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_text2img/stable_diffusion_1_4_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_1_5_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5").to(
            torch_device
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 50
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_text2img/stable_diffusion_1_5_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_ddim_reference_image(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 50
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_text2img/stable_diffusion_1_4_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 3e-3

    def test_stable_diffusion_lms_reference_image(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 50
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_text2img/stable_diffusion_1_4_lms.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_euler(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)
        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 50
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_text2img/stable_diffusion_1_4_euler.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3


@nightly
@require_torch_accelerator
class TestStableDiffusionPipelineCkpt:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_download_from_hub(self):
        ckpt_paths = [
            "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors",
            "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors",
        ]

        for ckpt_path in ckpt_paths:
            pipe = StableDiffusionPipeline.from_single_file(ckpt_path, torch_dtype=torch.float16)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe.to(torch_device)

        image_out = pipe("test", num_inference_steps=1, output_type="np").images[0]

        assert image_out.shape == (512, 512, 3)

    def test_download_local(self):
        ckpt_filename = hf_hub_download(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", filename="v1-5-pruned-emaonly.safetensors"
        )
        config_filename = hf_hub_download("stable-diffusion-v1-5/stable-diffusion-v1-5", filename="v1-inference.yaml")

        pipe = StableDiffusionPipeline.from_single_file(
            ckpt_filename, config_files={"v1": config_filename}, torch_dtype=torch.float16
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)

        image_out = pipe("test", num_inference_steps=1, output_type="np").images[0]

        assert image_out.shape == (512, 512, 3)


# (sayakpaul): This test suite was run in the DGX with two GPUs (1, 2).
@nightly
@require_torch_multi_accelerator
@require_accelerate_version_greater("0.27.0")
class TestStableDiffusionPipelineDeviceMap:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, generator_device="cpu", seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def get_pipeline_output_without_device_map(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to(torch_device)
        sd_pipe.set_progress_bar_config(disable=True)
        inputs = self.get_inputs()
        no_device_map_image = sd_pipe(**inputs).images

        del sd_pipe

        return no_device_map_image

    def test_forward_pass_balanced_device_map(self):
        no_device_map_image = self.get_pipeline_output_without_device_map()

        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )
        sd_pipe_with_device_map.set_progress_bar_config(disable=True)
        inputs = self.get_inputs()
        device_map_image = sd_pipe_with_device_map(**inputs).images

        max_diff = np.abs(device_map_image - no_device_map_image).max()
        assert max_diff < 1e-3

    def test_components_put_in_right_devices(self):
        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )

        assert len(set(sd_pipe_with_device_map.hf_device_map.values())) >= 2

    def test_max_memory(self):
        no_device_map_image = self.get_pipeline_output_without_device_map()

        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            device_map="balanced",
            max_memory={0: "1GB", 1: "1GB"},
            torch_dtype=torch.float16,
        )
        sd_pipe_with_device_map.set_progress_bar_config(disable=True)
        inputs = self.get_inputs()
        device_map_image = sd_pipe_with_device_map(**inputs).images

        max_diff = np.abs(device_map_image - no_device_map_image).max()
        assert max_diff < 1e-3

    def test_reset_device_map(self):
        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )
        sd_pipe_with_device_map.reset_device_map()

        assert sd_pipe_with_device_map.hf_device_map is None

        for name, component in sd_pipe_with_device_map.components.items():
            if isinstance(component, torch.nn.Module):
                assert component.device.type == "cpu"

    def test_reset_device_map_to(self):
        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )
        sd_pipe_with_device_map.reset_device_map()

        assert sd_pipe_with_device_map.hf_device_map is None

        # Make sure `to()` can be used and the pipeline can be called.
        pipe = sd_pipe_with_device_map.to(torch_device)
        _ = pipe("hello", num_inference_steps=2)

    def test_reset_device_map_enable_model_cpu_offload(self):
        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )
        sd_pipe_with_device_map.reset_device_map()

        assert sd_pipe_with_device_map.hf_device_map is None

        # Make sure `enable_model_cpu_offload()` can be used and the pipeline can be called.
        sd_pipe_with_device_map.enable_model_cpu_offload(device=torch_device)
        _ = sd_pipe_with_device_map("hello", num_inference_steps=2)

    def test_reset_device_map_enable_sequential_cpu_offload(self):
        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )
        sd_pipe_with_device_map.reset_device_map()

        assert sd_pipe_with_device_map.hf_device_map is None

        # Make sure `enable_sequential_cpu_offload()` can be used and the pipeline can be called.
        sd_pipe_with_device_map.enable_sequential_cpu_offload(device=torch_device)
        _ = sd_pipe_with_device_map("hello", num_inference_steps=2)


@slow
@require_torch_accelerator
@require_peft_backend
class TestStableDiffusionLoRADeviceIntegration:
    """`set_lora_device` behavior on the real SD 1.5 checkpoint (no logit assertions)."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_integration_move_lora_cpu(self):
        path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        lora_id = "takuma104/lora-test-text-encoder-lora-target"

        pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16)
        pipe.load_lora_weights(lora_id, adapter_name="adapter-1")
        pipe.load_lora_weights(lora_id, adapter_name="adapter-2")
        pipe = pipe.to(torch_device)

        assert check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"

        assert check_if_lora_correctly_set(pipe.unet), "Lora not correctly set in unet"

        # We will offload the first adapter in CPU and check if the offloading
        # has been performed correctly
        pipe.set_lora_device(["adapter-1"], "cpu")

        for name, module in pipe.unet.named_modules():
            if "adapter-1" in name and not isinstance(module, (nn.Dropout, nn.Identity)):
                assert module.weight.device == torch.device("cpu")
            elif "adapter-2" in name and not isinstance(module, (nn.Dropout, nn.Identity)):
                assert module.weight.device != torch.device("cpu")

        for name, module in pipe.text_encoder.named_modules():
            if "adapter-1" in name and not isinstance(module, (nn.Dropout, nn.Identity)):
                assert module.weight.device == torch.device("cpu")
            elif "adapter-2" in name and not isinstance(module, (nn.Dropout, nn.Identity)):
                assert module.weight.device != torch.device("cpu")

        pipe.set_lora_device(["adapter-1"], 0)

        for n, m in pipe.unet.named_modules():
            if "adapter-1" in n and not isinstance(m, (nn.Dropout, nn.Identity)):
                assert m.weight.device != torch.device("cpu")

        for n, m in pipe.text_encoder.named_modules():
            if "adapter-1" in n and not isinstance(m, (nn.Dropout, nn.Identity)):
                assert m.weight.device != torch.device("cpu")

        pipe.set_lora_device(["adapter-1", "adapter-2"], torch_device)

        for n, m in pipe.unet.named_modules():
            if ("adapter-1" in n or "adapter-2" in n) and not isinstance(m, (nn.Dropout, nn.Identity)):
                assert m.weight.device != torch.device("cpu")

        for n, m in pipe.text_encoder.named_modules():
            if ("adapter-1" in n or "adapter-2" in n) and not isinstance(m, (nn.Dropout, nn.Identity)):
                assert m.weight.device != torch.device("cpu")

    def test_integration_move_lora_dora_cpu(self):
        from peft import LoraConfig

        path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        unet_lora_config = LoraConfig(
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            use_dora=True,
        )
        text_lora_config = LoraConfig(
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            use_dora=True,
        )

        pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16)
        pipe.unet.add_adapter(unet_lora_config, "adapter-1")
        pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")

        assert check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"

        assert check_if_lora_correctly_set(pipe.unet), "Lora not correctly set in unet"

        for name, param in pipe.unet.named_parameters():
            if "lora_" in name:
                assert param.device == torch.device("cpu")

        for name, param in pipe.text_encoder.named_parameters():
            if "lora_" in name:
                assert param.device == torch.device("cpu")

        pipe.set_lora_device(["adapter-1"], torch_device)

        for name, param in pipe.unet.named_parameters():
            if "lora_" in name:
                assert param.device != torch.device("cpu")

        for name, param in pipe.text_encoder.named_parameters():
            if "lora_" in name:
                assert param.device != torch.device("cpu")

    def test_integration_set_lora_device_different_target_layers(self):
        # fixes a bug that occurred when calling set_lora_device with multiple adapters loaded that target different
        # layers, see #11833
        from peft import LoraConfig

        path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16)
        # configs partly target the same, partly different layers
        config0 = LoraConfig(target_modules=["to_k", "to_v"])
        config1 = LoraConfig(target_modules=["to_k", "to_q"])
        pipe.unet.add_adapter(config0, adapter_name="adapter-0")
        pipe.unet.add_adapter(config1, adapter_name="adapter-1")
        pipe = pipe.to(torch_device)

        assert check_if_lora_correctly_set(pipe.unet), "Lora not correctly set in unet"

        # sanity check that the adapters don't target the same layers, otherwise the test passes even without the fix
        modules_adapter_0 = {n for n, _ in pipe.unet.named_modules() if n.endswith(".adapter-0")}
        modules_adapter_1 = {n for n, _ in pipe.unet.named_modules() if n.endswith(".adapter-1")}
        assert modules_adapter_0 != modules_adapter_1
        assert modules_adapter_0 - modules_adapter_1
        assert modules_adapter_1 - modules_adapter_0

        # setting both separately works
        pipe.set_lora_device(["adapter-0"], "cpu")
        pipe.set_lora_device(["adapter-1"], "cpu")

        for name, module in pipe.unet.named_modules():
            if "adapter-0" in name and not isinstance(module, (nn.Dropout, nn.Identity)):
                assert module.weight.device == torch.device("cpu")
            elif "adapter-1" in name and not isinstance(module, (nn.Dropout, nn.Identity)):
                assert module.weight.device == torch.device("cpu")

        # setting both at once also works
        pipe.set_lora_device(["adapter-0", "adapter-1"], torch_device)

        for name, module in pipe.unet.named_modules():
            if "adapter-0" in name and not isinstance(module, (nn.Dropout, nn.Identity)):
                assert module.weight.device != torch.device("cpu")
            elif "adapter-1" in name and not isinstance(module, (nn.Dropout, nn.Identity)):
                assert module.weight.device != torch.device("cpu")


@slow
@nightly
@require_torch_accelerator
@require_peft_backend
class TestStableDiffusionLoRAIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_integration_logits_with_scale(self):
        path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        lora_id = "takuma104/lora-test-text-encoder-lora-target"

        pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float32)
        pipe.load_lora_weights(lora_id)
        pipe = pipe.to(torch_device)

        assert check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"

        prompt = "a red sks dog"

        images = pipe(
            prompt=prompt,
            num_inference_steps=15,
            cross_attention_kwargs={"scale": 0.5},
            generator=torch.manual_seed(0),
            output_type="np",
        ).images

        expected_slice_scale = np.array([0.307, 0.283, 0.310, 0.310, 0.300, 0.314, 0.336, 0.314, 0.321])
        predicted_slice = images[0, -3:, -3:, -1].flatten()

        max_diff = numpy_cosine_similarity_distance(expected_slice_scale, predicted_slice)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_integration_logits_no_scale(self):
        path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        lora_id = "takuma104/lora-test-text-encoder-lora-target"

        pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float32)
        pipe.load_lora_weights(lora_id)
        pipe = pipe.to(torch_device)

        assert check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"

        prompt = "a red sks dog"

        images = pipe(prompt=prompt, num_inference_steps=30, generator=torch.manual_seed(0), output_type="np").images

        expected_slice_scale = np.array([0.074, 0.064, 0.073, 0.0842, 0.069, 0.0641, 0.0794, 0.076, 0.084])
        predicted_slice = images[0, -3:, -3:, -1].flatten()

        max_diff = numpy_cosine_similarity_distance(expected_slice_scale, predicted_slice)

        assert max_diff < 1e-3

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_dreambooth_old_format(self):
        generator = torch.Generator("cpu").manual_seed(0)

        lora_model_id = "hf-internal-testing/lora_dreambooth_dog_example"

        base_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, safety_checker=None)
        pipe = pipe.to(torch_device)
        pipe.load_lora_weights(lora_model_id)

        images = pipe(
            "A photo of a sks dog floating in the river", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.7207, 0.6787, 0.6010, 0.7478, 0.6838, 0.6064, 0.6984, 0.6443, 0.5785])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-4

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_dreambooth_text_encoder_new_format(self):
        generator = torch.Generator().manual_seed(0)

        lora_model_id = "hf-internal-testing/lora-trained"

        base_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, safety_checker=None)
        pipe = pipe.to(torch_device)
        pipe.load_lora_weights(lora_model_id)

        images = pipe("A photo of a sks dog", output_type="np", generator=generator, num_inference_steps=2).images

        images = images[0, -3:, -3:, -1].flatten()

        expected = np.array([0.6628, 0.6138, 0.5390, 0.6625, 0.6130, 0.5463, 0.6166, 0.5788, 0.5359])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-4

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_a1111(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained("hf-internal-testing/Counterfeit-V2.5", safety_checker=None).to(
            torch_device
        )
        lora_model_id = "hf-internal-testing/civitai-light-shadow-lora"
        lora_filename = "light_and_shadow.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.3636, 0.3708, 0.3694, 0.3679, 0.3829, 0.3677, 0.3692, 0.3688, 0.3292])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_lycoris(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/Amixx", safety_checker=None, use_safetensors=True, variant="fp16"
        ).to(torch_device)
        lora_model_id = "hf-internal-testing/edgLycorisMugler-light"
        lora_filename = "edgLycorisMugler-light.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.6463, 0.658, 0.599, 0.6542, 0.6512, 0.6213, 0.658, 0.6485, 0.6017])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_a1111_with_model_cpu_offload(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained("hf-internal-testing/Counterfeit-V2.5", safety_checker=None)
        pipe.enable_model_cpu_offload(device=torch_device)
        lora_model_id = "hf-internal-testing/civitai-light-shadow-lora"
        lora_filename = "light_and_shadow.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.3636, 0.3708, 0.3694, 0.3679, 0.3829, 0.3677, 0.3692, 0.3688, 0.3292])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_a1111_with_sequential_cpu_offload(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained("hf-internal-testing/Counterfeit-V2.5", safety_checker=None)
        pipe.enable_sequential_cpu_offload(device=torch_device)
        lora_model_id = "hf-internal-testing/civitai-light-shadow-lora"
        lora_filename = "light_and_shadow.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.3636, 0.3708, 0.3694, 0.3679, 0.3829, 0.3677, 0.3692, 0.3688, 0.3292])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_kohya_sd_v15_with_higher_dimensions(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None
        ).to(torch_device)
        lora_model_id = "hf-internal-testing/urushisato-lora"
        lora_filename = "urushisato_v15.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.7165, 0.6616, 0.5833, 0.7504, 0.6718, 0.587, 0.6871, 0.6361, 0.5694])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_vanilla_funetuning(self):
        generator = torch.Generator().manual_seed(0)

        lora_model_id = "hf-internal-testing/sd-model-finetuned-lora-t4"

        base_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, safety_checker=None)
        pipe = pipe.to(torch_device)
        pipe.load_lora_weights(lora_model_id)

        images = pipe("A pokemon with blue eyes.", output_type="np", generator=generator, num_inference_steps=2).images

        image_slice = images[0, -3:, -3:, -1].flatten()

        expected_slices = Expectations(
            {
                ("xpu", 3): np.array(
                    [
                        0.6544,
                        0.6127,
                        0.5397,
                        0.6845,
                        0.6047,
                        0.5469,
                        0.6349,
                        0.5906,
                        0.5382,
                    ]
                ),
                ("cuda", 7): np.array(
                    [
                        0.7406,
                        0.699,
                        0.5963,
                        0.7493,
                        0.7045,
                        0.6096,
                        0.6886,
                        0.6388,
                        0.583,
                    ]
                ),
                ("cuda", 8): np.array(
                    [
                        0.6542,
                        0.61253,
                        0.5396,
                        0.6843,
                        0.6044,
                        0.5468,
                        0.6349,
                        0.5905,
                        0.5381,
                    ]
                ),
            }
        )
        expected_slice = expected_slices.get_expectation()

        max_diff = numpy_cosine_similarity_distance(expected_slice, image_slice)
        assert max_diff < 1e-4

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_unload_kohya_lora(self):
        generator = torch.manual_seed(0)
        prompt = "masterpiece, best quality, mountain"
        num_inference_steps = 2

        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None
        ).to(torch_device)
        initial_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        initial_images = initial_images[0, -3:, -3:, -1].flatten()

        lora_model_id = "hf-internal-testing/civitai-colored-icons-lora"
        lora_filename = "Colored_Icons_by_vizsumit.safetensors"

        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        generator = torch.manual_seed(0)
        lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        lora_images = lora_images[0, -3:, -3:, -1].flatten()

        pipe.unload_lora_weights()
        generator = torch.manual_seed(0)
        unloaded_lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        unloaded_lora_images = unloaded_lora_images[0, -3:, -3:, -1].flatten()

        assert not np.allclose(initial_images, lora_images)
        assert np.allclose(initial_images, unloaded_lora_images, atol=1e-3)

        release_memory(pipe)

    def test_load_unload_load_kohya_lora(self):
        # This test ensures that a Kohya-style LoRA can be safely unloaded and then loaded
        # without introducing any side-effects. Even though the test uses a Kohya-style
        # LoRA, the underlying adapter handling mechanism is format-agnostic.
        generator = torch.manual_seed(0)
        prompt = "masterpiece, best quality, mountain"
        num_inference_steps = 2

        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None
        ).to(torch_device)
        initial_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        initial_images = initial_images[0, -3:, -3:, -1].flatten()

        lora_model_id = "hf-internal-testing/civitai-colored-icons-lora"
        lora_filename = "Colored_Icons_by_vizsumit.safetensors"

        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        generator = torch.manual_seed(0)
        lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        lora_images = lora_images[0, -3:, -3:, -1].flatten()

        pipe.unload_lora_weights()
        generator = torch.manual_seed(0)
        unloaded_lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        unloaded_lora_images = unloaded_lora_images[0, -3:, -3:, -1].flatten()

        assert not np.allclose(initial_images, lora_images)
        assert np.allclose(initial_images, unloaded_lora_images, atol=1e-3)

        # make sure we can load a LoRA again after unloading and they don't have
        # any undesired effects.
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        generator = torch.manual_seed(0)
        lora_images_again = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        lora_images_again = lora_images_again[0, -3:, -3:, -1].flatten()

        assert np.allclose(lora_images, lora_images_again, atol=1e-3)
        release_memory(pipe)

    def test_not_empty_state_dict(self):
        # Makes sure https://github.com/huggingface/diffusers/issues/7054 does not happen again
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to(torch_device)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        cached_file = hf_hub_download("hf-internal-testing/lcm-lora-test-sd-v1-5", "test_lora.safetensors")
        lcm_lora = load_file(cached_file)

        pipe.load_lora_weights(lcm_lora, adapter_name="lcm")
        assert lcm_lora != {}
        release_memory(pipe)

    def test_load_unload_load_state_dict(self):
        # Makes sure https://github.com/huggingface/diffusers/issues/7054 does not happen again
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to(torch_device)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        cached_file = hf_hub_download("hf-internal-testing/lcm-lora-test-sd-v1-5", "test_lora.safetensors")
        lcm_lora = load_file(cached_file)
        previous_state_dict = lcm_lora.copy()

        pipe.load_lora_weights(lcm_lora, adapter_name="lcm")
        assert lcm_lora == previous_state_dict

        pipe.unload_lora_weights()
        pipe.load_lora_weights(lcm_lora, adapter_name="lcm")
        assert lcm_lora == previous_state_dict

        release_memory(pipe)

    def test_sdv1_5_lcm_lora(self):
        pipe = DiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        pipe.to(torch_device)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        generator = torch.Generator("cpu").manual_seed(0)

        lora_model_id = "latent-consistency/lcm-lora-sdv1-5"
        pipe.load_lora_weights(lora_model_id)

        image = pipe(
            "masterpiece, best quality, mountain", generator=generator, num_inference_steps=4, guidance_scale=0.5
        ).images[0]

        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/lcm_lora/sdv15_lcm_lora.png"
        )

        image_np = pipe.image_processor.pil_to_numpy(image)
        expected_image_np = pipe.image_processor.pil_to_numpy(expected_image)

        max_diff = numpy_cosine_similarity_distance(image_np.flatten(), expected_image_np.flatten())
        assert max_diff < 1e-4

        pipe.unload_lora_weights()

        release_memory(pipe)

    def test_sdv1_5_lcm_lora_img2img(self):
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        pipe.to(torch_device)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/fantasy_landscape.png"
        )

        generator = torch.Generator("cpu").manual_seed(0)

        lora_model_id = "latent-consistency/lcm-lora-sdv1-5"
        pipe.load_lora_weights(lora_model_id)

        image = pipe(
            "snowy mountain",
            generator=generator,
            image=init_image,
            strength=0.5,
            num_inference_steps=4,
            guidance_scale=0.5,
        ).images[0]

        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/lcm_lora/sdv15_lcm_lora_img2img.png"
        )

        image_np = pipe.image_processor.pil_to_numpy(image)
        expected_image_np = pipe.image_processor.pil_to_numpy(expected_image)

        max_diff = numpy_cosine_similarity_distance(image_np.flatten(), expected_image_np.flatten())
        assert max_diff < 1e-4

        pipe.unload_lora_weights()

        release_memory(pipe)

    def test_sd_load_civitai_empty_network_alpha(self):
        """
        This test simply checks that loading a LoRA with an empty network alpha works fine
        See: https://github.com/huggingface/diffusers/issues/5606
        """
        pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
        pipeline.enable_sequential_cpu_offload(device=torch_device)
        civitai_path = hf_hub_download("ybelkada/test-ahi-civitai", "ahi_lora_weights.safetensors")
        pipeline.load_lora_weights(civitai_path, adapter_name="ahri")

        images = pipeline(
            "ahri, masterpiece, league of legends",
            output_type="np",
            generator=torch.manual_seed(156),
            num_inference_steps=5,
        ).images
        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.0, 0.0, 0.0, 0.002557, 0.020954, 0.001792, 0.006581, 0.00591, 0.002995])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipeline.unload_lora_weights()
        release_memory(pipeline)
