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
from diffusers.pipelines.pipeline_loading_utils import is_safetensors_compatible, variant_compatible_siblings

from ..testing_utils import require_torch_accelerator, torch_device


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
        self.assertFalse(is_safetensors_compatible(filenames))
        self.assertTrue(is_safetensors_compatible(filenames, variant="fp16"))

    def test_diffusers_model_is_compatible_variant(self):
        filenames = [
            "unet/diffusion_pytorch_model.fp16.bin",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ]
        self.assertFalse(is_safetensors_compatible(filenames))
        self.assertTrue(is_safetensors_compatible(filenames, variant="fp16"))

    def test_diffusers_model_is_compatible_variant_mixed(self):
        filenames = [
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ]
        self.assertFalse(is_safetensors_compatible(filenames))
        self.assertTrue(is_safetensors_compatible(filenames, variant="fp16"))

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
        self.assertFalse(is_safetensors_compatible(filenames))
        self.assertTrue(is_safetensors_compatible(filenames, variant="fp16"))

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
        self.assertFalse(is_safetensors_compatible(filenames, folder_names={"vae", "unet"}))
        self.assertTrue(is_safetensors_compatible(filenames, folder_names={"vae", "unet"}, variant="fp16"))

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
        self.assertFalse(is_safetensors_compatible(filenames))
        self.assertTrue(is_safetensors_compatible(filenames, variant="fp16"))

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
        self.assertFalse(is_safetensors_compatible(filenames))
        self.assertTrue(is_safetensors_compatible(filenames, variant="fp16"))

    def test_diffusers_is_compatible_only_variants(self):
        filenames = [
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ]
        self.assertFalse(is_safetensors_compatible(filenames))
        self.assertTrue(is_safetensors_compatible(filenames, variant="fp16"))

    def test_diffusers_is_compatible_no_components(self):
        filenames = [
            "diffusion_pytorch_model.bin",
        ]
        self.assertFalse(is_safetensors_compatible(filenames))

    def test_diffusers_is_compatible_no_components_only_variants(self):
        filenames = [
            "diffusion_pytorch_model.fp16.bin",
        ]
        self.assertFalse(is_safetensors_compatible(filenames))

    def test_is_compatible_mixed_variants(self):
        filenames = [
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames, variant="fp16"))

    def test_is_compatible_variant_and_non_safetensors(self):
        filenames = [
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.bin",
        ]
        self.assertFalse(is_safetensors_compatible(filenames, variant="fp16"))


class VariantCompatibleSiblingsTest(unittest.TestCase):
    def test_only_non_variants_downloaded(self):
        ignore_patterns = ["*.bin"]
        variant = "fp16"
        filenames = [
            f"vae/diffusion_pytorch_model.{variant}.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            f"text_encoder/model.{variant}.safetensors",
            "text_encoder/model.safetensors",
            f"unet/diffusion_pytorch_model.{variant}.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
        ]

        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=None, ignore_patterns=ignore_patterns
        )
        assert all(variant not in f for f in model_filenames)

    def test_only_variants_downloaded(self):
        ignore_patterns = ["*.bin"]
        variant = "fp16"
        filenames = [
            f"vae/diffusion_pytorch_model.{variant}.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            f"text_encoder/model.{variant}.safetensors",
            "text_encoder/model.safetensors",
            f"unet/diffusion_pytorch_model.{variant}.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
        ]

        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=variant, ignore_patterns=ignore_patterns
        )
        assert all(variant in f for f in model_filenames)

    def test_mixed_variants_downloaded(self):
        ignore_patterns = ["*.bin"]
        variant = "fp16"
        non_variant_file = "text_encoder/model.safetensors"
        filenames = [
            f"vae/diffusion_pytorch_model.{variant}.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/model.safetensors",
            f"unet/diffusion_pytorch_model.{variant}.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=variant, ignore_patterns=ignore_patterns
        )
        assert all(variant in f if f != non_variant_file else variant not in f for f in model_filenames)

    def test_non_variants_in_main_dir_downloaded(self):
        ignore_patterns = ["*.bin"]
        variant = "fp16"
        filenames = [
            f"diffusion_pytorch_model.{variant}.safetensors",
            "diffusion_pytorch_model.safetensors",
            "model.safetensors",
            f"model.{variant}.safetensors",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=None, ignore_patterns=ignore_patterns
        )
        assert all(variant not in f for f in model_filenames)

    def test_variants_in_main_dir_downloaded(self):
        ignore_patterns = ["*.bin"]
        variant = "fp16"
        filenames = [
            f"diffusion_pytorch_model.{variant}.safetensors",
            "diffusion_pytorch_model.safetensors",
            "model.safetensors",
            f"model.{variant}.safetensors",
            f"diffusion_pytorch_model.{variant}.safetensors",
            "diffusion_pytorch_model.safetensors",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=variant, ignore_patterns=ignore_patterns
        )
        assert all(variant in f for f in model_filenames)

    def test_mixed_variants_in_main_dir_downloaded(self):
        ignore_patterns = ["*.bin"]
        variant = "fp16"
        non_variant_file = "model.safetensors"
        filenames = [
            f"diffusion_pytorch_model.{variant}.safetensors",
            "diffusion_pytorch_model.safetensors",
            "model.safetensors",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=variant, ignore_patterns=ignore_patterns
        )
        assert all(variant in f if f != non_variant_file else variant not in f for f in model_filenames)

    def test_sharded_variants_in_main_dir_downloaded(self):
        ignore_patterns = ["*.bin"]
        variant = "fp16"
        filenames = [
            "diffusion_pytorch_model.safetensors.index.json",
            "diffusion_pytorch_model-00001-of-00003.safetensors",
            "diffusion_pytorch_model-00002-of-00003.safetensors",
            "diffusion_pytorch_model-00003-of-00003.safetensors",
            f"diffusion_pytorch_model.{variant}-00001-of-00002.safetensors",
            f"diffusion_pytorch_model.{variant}-00002-of-00002.safetensors",
            f"diffusion_pytorch_model.safetensors.index.{variant}.json",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=variant, ignore_patterns=ignore_patterns
        )
        assert all(variant in f for f in model_filenames)

    def test_mixed_sharded_and_variant_in_main_dir_downloaded(self):
        ignore_patterns = ["*.bin"]
        variant = "fp16"
        filenames = [
            "diffusion_pytorch_model.safetensors.index.json",
            "diffusion_pytorch_model-00001-of-00003.safetensors",
            "diffusion_pytorch_model-00002-of-00003.safetensors",
            "diffusion_pytorch_model-00003-of-00003.safetensors",
            f"diffusion_pytorch_model.{variant}.safetensors",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=variant, ignore_patterns=ignore_patterns
        )
        assert all(variant in f for f in model_filenames)

    def test_mixed_sharded_non_variants_in_main_dir_downloaded(self):
        ignore_patterns = ["*.bin"]
        variant = "fp16"
        filenames = [
            f"diffusion_pytorch_model.safetensors.index.{variant}.json",
            "diffusion_pytorch_model.safetensors.index.json",
            "diffusion_pytorch_model-00001-of-00003.safetensors",
            "diffusion_pytorch_model-00002-of-00003.safetensors",
            "diffusion_pytorch_model-00003-of-00003.safetensors",
            f"diffusion_pytorch_model.{variant}-00001-of-00002.safetensors",
            f"diffusion_pytorch_model.{variant}-00002-of-00002.safetensors",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=None, ignore_patterns=ignore_patterns
        )
        assert all(variant not in f for f in model_filenames)

    def test_sharded_non_variants_downloaded(self):
        ignore_patterns = ["*.bin"]
        variant = "fp16"
        filenames = [
            f"unet/diffusion_pytorch_model.safetensors.index.{variant}.json",
            "unet/diffusion_pytorch_model.safetensors.index.json",
            "unet/diffusion_pytorch_model-00001-of-00003.safetensors",
            "unet/diffusion_pytorch_model-00002-of-00003.safetensors",
            "unet/diffusion_pytorch_model-00003-of-00003.safetensors",
            f"unet/diffusion_pytorch_model.{variant}-00001-of-00002.safetensors",
            f"unet/diffusion_pytorch_model.{variant}-00002-of-00002.safetensors",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=None, ignore_patterns=ignore_patterns
        )
        assert all(variant not in f for f in model_filenames)

    def test_sharded_variants_downloaded(self):
        ignore_patterns = ["*.bin"]
        variant = "fp16"
        filenames = [
            f"unet/diffusion_pytorch_model.safetensors.index.{variant}.json",
            "unet/diffusion_pytorch_model.safetensors.index.json",
            "unet/diffusion_pytorch_model-00001-of-00003.safetensors",
            "unet/diffusion_pytorch_model-00002-of-00003.safetensors",
            "unet/diffusion_pytorch_model-00003-of-00003.safetensors",
            f"unet/diffusion_pytorch_model.{variant}-00001-of-00002.safetensors",
            f"unet/diffusion_pytorch_model.{variant}-00002-of-00002.safetensors",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=variant, ignore_patterns=ignore_patterns
        )
        assert all(variant in f for f in model_filenames)
        assert model_filenames == variant_filenames

    def test_single_variant_with_sharded_non_variant_downloaded(self):
        ignore_patterns = ["*.bin"]
        variant = "fp16"
        filenames = [
            "unet/diffusion_pytorch_model.safetensors.index.json",
            "unet/diffusion_pytorch_model-00001-of-00003.safetensors",
            "unet/diffusion_pytorch_model-00002-of-00003.safetensors",
            "unet/diffusion_pytorch_model-00003-of-00003.safetensors",
            f"unet/diffusion_pytorch_model.{variant}.safetensors",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=variant, ignore_patterns=ignore_patterns
        )
        assert all(variant in f for f in model_filenames)

    def test_mixed_single_variant_with_sharded_non_variant_downloaded(self):
        ignore_patterns = ["*.bin"]
        variant = "fp16"
        allowed_non_variant = "unet"
        filenames = [
            "vae/diffusion_pytorch_model.safetensors.index.json",
            "vae/diffusion_pytorch_model-00001-of-00003.safetensors",
            "vae/diffusion_pytorch_model-00002-of-00003.safetensors",
            "vae/diffusion_pytorch_model-00003-of-00003.safetensors",
            f"vae/diffusion_pytorch_model.{variant}.safetensors",
            "unet/diffusion_pytorch_model.safetensors.index.json",
            "unet/diffusion_pytorch_model-00001-of-00003.safetensors",
            "unet/diffusion_pytorch_model-00002-of-00003.safetensors",
            "unet/diffusion_pytorch_model-00003-of-00003.safetensors",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=variant, ignore_patterns=ignore_patterns
        )
        assert all(variant in f if allowed_non_variant not in f else variant not in f for f in model_filenames)

    def test_sharded_mixed_variants_downloaded(self):
        ignore_patterns = ["*.bin"]
        variant = "fp16"
        allowed_non_variant = "unet"
        filenames = [
            f"vae/diffusion_pytorch_model.safetensors.index.{variant}.json",
            "vae/diffusion_pytorch_model.safetensors.index.json",
            "unet/diffusion_pytorch_model.safetensors.index.json",
            "unet/diffusion_pytorch_model-00001-of-00003.safetensors",
            "unet/diffusion_pytorch_model-00002-of-00003.safetensors",
            "unet/diffusion_pytorch_model-00003-of-00003.safetensors",
            f"vae/diffusion_pytorch_model.{variant}-00001-of-00002.safetensors",
            f"vae/diffusion_pytorch_model.{variant}-00002-of-00002.safetensors",
            "vae/diffusion_pytorch_model-00001-of-00003.safetensors",
            "vae/diffusion_pytorch_model-00002-of-00003.safetensors",
            "vae/diffusion_pytorch_model-00003-of-00003.safetensors",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=variant, ignore_patterns=ignore_patterns
        )
        assert all(variant in f if allowed_non_variant not in f else variant not in f for f in model_filenames)

    def test_downloading_when_no_variant_exists(self):
        ignore_patterns = ["*.bin"]
        variant = "fp16"
        filenames = ["model.safetensors", "diffusion_pytorch_model.safetensors"]
        with self.assertRaisesRegex(ValueError, "but no such modeling files are available. "):
            model_filenames, variant_filenames = variant_compatible_siblings(
                filenames, variant=variant, ignore_patterns=ignore_patterns
            )

    def test_downloading_use_safetensors_false(self):
        ignore_patterns = ["*.safetensors"]
        filenames = [
            "text_encoder/model.bin",
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.safetensors",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=None, ignore_patterns=ignore_patterns
        )

        assert all(".safetensors" not in f for f in model_filenames)

    def test_non_variant_in_main_dir_with_variant_in_subfolder(self):
        ignore_patterns = ["*.bin"]
        variant = "fp16"
        allowed_non_variant = "diffusion_pytorch_model.safetensors"
        filenames = [
            f"unet/diffusion_pytorch_model.{variant}.safetensors",
            "diffusion_pytorch_model.safetensors",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=variant, ignore_patterns=ignore_patterns
        )
        assert all(variant in f if allowed_non_variant not in f else variant not in f for f in model_filenames)

    def test_download_variants_when_component_has_no_safetensors_variant(self):
        ignore_patterns = None
        variant = "fp16"
        filenames = [
            f"unet/diffusion_pytorch_model.{variant}.bin",
            "vae/diffusion_pytorch_model.safetensors",
            f"vae/diffusion_pytorch_model.{variant}.safetensors",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=variant, ignore_patterns=ignore_patterns
        )
        assert {
            f"unet/diffusion_pytorch_model.{variant}.bin",
            f"vae/diffusion_pytorch_model.{variant}.safetensors",
        } == model_filenames

    def test_error_when_download_sharded_variants_when_component_has_no_safetensors_variant(self):
        ignore_patterns = ["*.bin"]
        variant = "fp16"
        filenames = [
            f"vae/diffusion_pytorch_model.bin.index.{variant}.json",
            "vae/diffusion_pytorch_model.safetensors.index.json",
            f"vae/diffusion_pytorch_model.{variant}-00002-of-00002.bin",
            "vae/diffusion_pytorch_model-00001-of-00003.safetensors",
            "vae/diffusion_pytorch_model-00002-of-00003.safetensors",
            "vae/diffusion_pytorch_model-00003-of-00003.safetensors",
            "unet/diffusion_pytorch_model.safetensors.index.json",
            "unet/diffusion_pytorch_model-00001-of-00003.safetensors",
            "unet/diffusion_pytorch_model-00002-of-00003.safetensors",
            "unet/diffusion_pytorch_model-00003-of-00003.safetensors",
            f"vae/diffusion_pytorch_model.{variant}-00001-of-00002.bin",
        ]
        with self.assertRaisesRegex(ValueError, "but no such modeling files are available. "):
            model_filenames, variant_filenames = variant_compatible_siblings(
                filenames, variant=variant, ignore_patterns=ignore_patterns
            )

    def test_download_sharded_variants_when_component_has_no_safetensors_variant_and_safetensors_false(self):
        ignore_patterns = ["*.safetensors"]
        allowed_non_variant = "unet"
        variant = "fp16"
        filenames = [
            f"vae/diffusion_pytorch_model.bin.index.{variant}.json",
            "vae/diffusion_pytorch_model.safetensors.index.json",
            f"vae/diffusion_pytorch_model.{variant}-00002-of-00002.bin",
            "vae/diffusion_pytorch_model-00001-of-00003.safetensors",
            "vae/diffusion_pytorch_model-00002-of-00003.safetensors",
            "vae/diffusion_pytorch_model-00003-of-00003.safetensors",
            "unet/diffusion_pytorch_model.safetensors.index.json",
            "unet/diffusion_pytorch_model-00001-of-00003.safetensors",
            "unet/diffusion_pytorch_model-00002-of-00003.safetensors",
            "unet/diffusion_pytorch_model-00003-of-00003.safetensors",
            f"vae/diffusion_pytorch_model.{variant}-00001-of-00002.bin",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=variant, ignore_patterns=ignore_patterns
        )
        assert all(variant in f if allowed_non_variant not in f else variant not in f for f in model_filenames)

    def test_download_sharded_legacy_variants(self):
        ignore_patterns = None
        variant = "fp16"
        filenames = [
            f"vae/transformer/diffusion_pytorch_model.safetensors.{variant}.index.json",
            "vae/diffusion_pytorch_model.safetensors.index.json",
            f"vae/diffusion_pytorch_model-00002-of-00002.{variant}.safetensors",
            "vae/diffusion_pytorch_model-00001-of-00003.safetensors",
            "vae/diffusion_pytorch_model-00002-of-00003.safetensors",
            "vae/diffusion_pytorch_model-00003-of-00003.safetensors",
            f"vae/diffusion_pytorch_model-00001-of-00002.{variant}.safetensors",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=variant, ignore_patterns=ignore_patterns
        )
        assert all(variant in f for f in model_filenames)

    def test_download_onnx_models(self):
        ignore_patterns = ["*.safetensors"]
        filenames = [
            "vae/model.onnx",
            "unet/model.onnx",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=None, ignore_patterns=ignore_patterns
        )
        assert model_filenames == set(filenames)

    def test_download_flax_models(self):
        ignore_patterns = ["*.safetensors", "*.bin"]
        filenames = [
            "vae/diffusion_flax_model.msgpack",
            "unet/diffusion_flax_model.msgpack",
        ]
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=None, ignore_patterns=ignore_patterns
        )
        assert model_filenames == set(filenames)


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


@require_torch_accelerator
class PipelineDeviceAndDtypeStabilityTests(unittest.TestCase):
    expected_pipe_device = torch.device(f"{torch_device}:0")
    expected_pipe_dtype = torch.float64

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

    def test_deterministic_device(self):
        components = self.get_dummy_components_image_generation()

        pipe = StableDiffusionPipeline(**components)
        pipe.to(device=torch_device, dtype=torch.float32)

        pipe.unet.to(device="cpu")
        pipe.vae.to(device=torch_device)
        pipe.text_encoder.to(device=f"{torch_device}:0")

        pipe_device = pipe.device

        self.assertEqual(
            self.expected_pipe_device,
            pipe_device,
            f"Wrong expected device. Expected {self.expected_pipe_device}. Got {pipe_device}.",
        )

    def test_deterministic_dtype(self):
        components = self.get_dummy_components_image_generation()

        pipe = StableDiffusionPipeline(**components)
        pipe.to(device=torch_device, dtype=torch.float32)

        pipe.unet.to(dtype=torch.float16)
        pipe.vae.to(dtype=torch.float32)
        pipe.text_encoder.to(dtype=torch.float64)

        pipe_dtype = pipe.dtype

        self.assertEqual(
            self.expected_pipe_dtype,
            pipe_dtype,
            f"Wrong expected dtype. Expected {self.expected_pipe_dtype}. Got {pipe_dtype}.",
        )
