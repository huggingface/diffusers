import contextlib
import io
import re
import unittest

import torch
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AnimateDiffPipeline,
    AnimateDiffVideoToVideoPipeline,
    AutoencoderKL,
    DDIMScheduler,
    MotionAdapter,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.pipeline_utils import is_safetensors_compatible
from diffusers.utils.testing_utils import torch_device


class IsSafetensorsCompatibleTests(unittest.TestCase):
    def test_all_is_compatible(self):
        filenames = [
            "safety_checker/pytorch_model.bin",
            "safety_checker/model.safetensors",
            "vae/diffusion_pytorch_model.bin",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/pytorch_model.bin",
            "text_encoder/model.safetensors",
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames))

    def test_diffusers_model_is_compatible(self):
        filenames = [
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames))

    def test_diffusers_model_is_not_compatible(self):
        filenames = [
            "safety_checker/pytorch_model.bin",
            "safety_checker/model.safetensors",
            "vae/diffusion_pytorch_model.bin",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/pytorch_model.bin",
            "text_encoder/model.safetensors",
            "unet/diffusion_pytorch_model.bin",
            # Removed: 'unet/diffusion_pytorch_model.safetensors',
        ]
        self.assertFalse(is_safetensors_compatible(filenames))

    def test_transformer_model_is_compatible(self):
        filenames = [
            "text_encoder/pytorch_model.bin",
            "text_encoder/model.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames))

    def test_transformer_model_is_not_compatible(self):
        filenames = [
            "safety_checker/pytorch_model.bin",
            "safety_checker/model.safetensors",
            "vae/diffusion_pytorch_model.bin",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/pytorch_model.bin",
            # Removed: 'text_encoder/model.safetensors',
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.safetensors",
        ]
        self.assertFalse(is_safetensors_compatible(filenames))

    def test_all_is_compatible_variant(self):
        filenames = [
            "safety_checker/pytorch_model.fp16.bin",
            "safety_checker/model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.bin",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/pytorch_model.fp16.bin",
            "text_encoder/model.fp16.safetensors",
            "unet/diffusion_pytorch_model.fp16.bin",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames))

    def test_diffusers_model_is_compatible_variant(self):
        filenames = [
            "unet/diffusion_pytorch_model.fp16.bin",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames))

    def test_diffusers_model_is_compatible_variant_mixed(self):
        filenames = [
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames))

    def test_diffusers_model_is_not_compatible_variant(self):
        filenames = [
            "safety_checker/pytorch_model.fp16.bin",
            "safety_checker/model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.bin",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/pytorch_model.fp16.bin",
            "text_encoder/model.fp16.safetensors",
            "unet/diffusion_pytorch_model.fp16.bin",
            # Removed: 'unet/diffusion_pytorch_model.fp16.safetensors',
        ]
        self.assertFalse(is_safetensors_compatible(filenames))

    def test_transformer_model_is_compatible_variant(self):
        filenames = [
            "text_encoder/pytorch_model.fp16.bin",
            "text_encoder/model.fp16.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames))

    def test_transformer_model_is_not_compatible_variant(self):
        filenames = [
            "safety_checker/pytorch_model.fp16.bin",
            "safety_checker/model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.bin",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/pytorch_model.fp16.bin",
            "unet/diffusion_pytorch_model.fp16.bin",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ]
        self.assertFalse(is_safetensors_compatible(filenames))

    def test_transformer_model_is_compatible_variant_extra_folder(self):
        filenames = [
            "safety_checker/pytorch_model.fp16.bin",
            "safety_checker/model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.bin",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/pytorch_model.fp16.bin",
            "unet/diffusion_pytorch_model.fp16.bin",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames, folder_names={"vae", "unet"}))

    def test_transformer_model_is_not_compatible_variant_extra_folder(self):
        filenames = [
            "safety_checker/pytorch_model.fp16.bin",
            "safety_checker/model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.bin",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/pytorch_model.fp16.bin",
            "unet/diffusion_pytorch_model.fp16.bin",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ]
        self.assertFalse(is_safetensors_compatible(filenames, folder_names={"text_encoder"}))

    def test_transformers_is_compatible_sharded(self):
        filenames = [
            "text_encoder/pytorch_model.bin",
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/model-00002-of-00002.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames))

    def test_transformers_is_compatible_variant_sharded(self):
        filenames = [
            "text_encoder/pytorch_model.bin",
            "text_encoder/model.fp16-00001-of-00002.safetensors",
            "text_encoder/model.fp16-00001-of-00002.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames))

    def test_diffusers_is_compatible_sharded(self):
        filenames = [
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model-00001-of-00002.safetensors",
            "unet/diffusion_pytorch_model-00002-of-00002.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames))

    def test_diffusers_is_compatible_variant_sharded(self):
        filenames = [
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
            "unet/diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames))

    def test_diffusers_is_compatible_only_variants(self):
        filenames = [
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames))


class ProgressBarTests(unittest.TestCase):
    def get_dummy_components_image_generation(self):
        cross_attention_dim = 8

        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=1,
            sample_size=32,
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

    def get_dummy_components_video_generation(self):
        cross_attention_dim = 8
        block_out_channels = (8, 8)

        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=block_out_channels,
            layers_per_block=2,
            sample_size=8,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=cross_attention_dim,
            norm_num_groups=2,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            clip_sample=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=block_out_channels,
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
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        torch.manual_seed(0)
        motion_adapter = MotionAdapter(
            block_out_channels=block_out_channels,
            motion_layers_per_block=2,
            motion_norm_num_groups=2,
            motion_num_attention_heads=4,
        )

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "motion_adapter": motion_adapter,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "feature_extractor": None,
            "image_encoder": None,
        }
        return components

    def test_text_to_image(self):
        components = self.get_dummy_components_image_generation()
        pipe = StableDiffusionPipeline(**components)
        pipe.to(torch_device)

        inputs = {"prompt": "a cute cat", "num_inference_steps": 2}
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            stderr = stderr.getvalue()
            # we can't calculate the number of progress steps beforehand e.g. for strength-dependent img2img,
            # so we just match "5" in "#####| 1/5 [00:01<00:00]"
            max_steps = re.search("/(.*?) ", stderr).group(1)
            self.assertTrue(max_steps is not None and len(max_steps) > 0)
            self.assertTrue(
                f"{max_steps}/{max_steps}" in stderr, "Progress bar should be enabled and stopped at the max step"
            )

        pipe.set_progress_bar_config(disable=True)
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            self.assertTrue(stderr.getvalue() == "", "Progress bar should be disabled")

    def test_image_to_image(self):
        components = self.get_dummy_components_image_generation()
        pipe = StableDiffusionImg2ImgPipeline(**components)
        pipe.to(torch_device)

        image = Image.new("RGB", (32, 32))
        inputs = {"prompt": "a cute cat", "num_inference_steps": 2, "strength": 0.5, "image": image}
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            stderr = stderr.getvalue()
            # we can't calculate the number of progress steps beforehand e.g. for strength-dependent img2img,
            # so we just match "5" in "#####| 1/5 [00:01<00:00]"
            max_steps = re.search("/(.*?) ", stderr).group(1)
            self.assertTrue(max_steps is not None and len(max_steps) > 0)
            self.assertTrue(
                f"{max_steps}/{max_steps}" in stderr, "Progress bar should be enabled and stopped at the max step"
            )

        pipe.set_progress_bar_config(disable=True)
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            self.assertTrue(stderr.getvalue() == "", "Progress bar should be disabled")

    def test_inpainting(self):
        components = self.get_dummy_components_image_generation()
        pipe = StableDiffusionInpaintPipeline(**components)
        pipe.to(torch_device)

        image = Image.new("RGB", (32, 32))
        mask = Image.new("RGB", (32, 32))
        inputs = {
            "prompt": "a cute cat",
            "num_inference_steps": 2,
            "strength": 0.5,
            "image": image,
            "mask_image": mask,
        }
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            stderr = stderr.getvalue()
            # we can't calculate the number of progress steps beforehand e.g. for strength-dependent img2img,
            # so we just match "5" in "#####| 1/5 [00:01<00:00]"
            max_steps = re.search("/(.*?) ", stderr).group(1)
            self.assertTrue(max_steps is not None and len(max_steps) > 0)
            self.assertTrue(
                f"{max_steps}/{max_steps}" in stderr, "Progress bar should be enabled and stopped at the max step"
            )

        pipe.set_progress_bar_config(disable=True)
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            self.assertTrue(stderr.getvalue() == "", "Progress bar should be disabled")

    def test_text_to_video(self):
        components = self.get_dummy_components_video_generation()
        pipe = AnimateDiffPipeline(**components)
        pipe.to(torch_device)

        inputs = {"prompt": "a cute cat", "num_inference_steps": 2, "num_frames": 2}
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            stderr = stderr.getvalue()
            # we can't calculate the number of progress steps beforehand e.g. for strength-dependent img2img,
            # so we just match "5" in "#####| 1/5 [00:01<00:00]"
            max_steps = re.search("/(.*?) ", stderr).group(1)
            self.assertTrue(max_steps is not None and len(max_steps) > 0)
            self.assertTrue(
                f"{max_steps}/{max_steps}" in stderr, "Progress bar should be enabled and stopped at the max step"
            )

        pipe.set_progress_bar_config(disable=True)
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            self.assertTrue(stderr.getvalue() == "", "Progress bar should be disabled")

    def test_video_to_video(self):
        components = self.get_dummy_components_video_generation()
        pipe = AnimateDiffVideoToVideoPipeline(**components)
        pipe.to(torch_device)

        num_frames = 2
        video = [Image.new("RGB", (32, 32))] * num_frames
        inputs = {"prompt": "a cute cat", "num_inference_steps": 2, "video": video}
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            stderr = stderr.getvalue()
            # we can't calculate the number of progress steps beforehand e.g. for strength-dependent img2img,
            # so we just match "5" in "#####| 1/5 [00:01<00:00]"
            max_steps = re.search("/(.*?) ", stderr).group(1)
            self.assertTrue(max_steps is not None and len(max_steps) > 0)
            self.assertTrue(
                f"{max_steps}/{max_steps}" in stderr, "Progress bar should be enabled and stopped at the max step"
            )

        pipe.set_progress_bar_config(disable=True)
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            self.assertTrue(stderr.getvalue() == "", "Progress bar should be disabled")
