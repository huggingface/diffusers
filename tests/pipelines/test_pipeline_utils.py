import unittest

from diffusers.pipelines.pipeline_utils import is_safetensors_compatible


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
        variant = "fp16"
        self.assertTrue(is_safetensors_compatible(filenames, variant=variant))

    def test_diffusers_model_is_compatible_variant(self):
        filenames = [
            "unet/diffusion_pytorch_model.fp16.bin",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ]
        variant = "fp16"
        self.assertTrue(is_safetensors_compatible(filenames, variant=variant))

    def test_diffusers_model_is_compatible_variant_partial(self):
        # pass variant but use the non-variant filenames
        filenames = [
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.safetensors",
        ]
        variant = "fp16"
        self.assertFalse(is_safetensors_compatible(filenames, variant=variant))

    def test_diffusers_model_is_compatible_variant_mixed(self):
        # pass variant but use the non-variant filenames
        filenames = [
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ]
        variant = "fp16"
        self.assertTrue(is_safetensors_compatible(filenames, variant=variant))

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
        variant = "fp16"
        self.assertFalse(is_safetensors_compatible(filenames, variant=variant))

    def test_transformer_model_is_compatible_variant(self):
        filenames = [
            "text_encoder/pytorch_model.fp16.bin",
            "text_encoder/model.fp16.safetensors",
        ]
        variant = "fp16"
        self.assertTrue(is_safetensors_compatible(filenames, variant=variant))

    def test_transformer_model_is_compatible_variant_partial(self):
        # pass variant but use the non-variant filenames
        filenames = [
            "text_encoder/pytorch_model.bin",
            "text_encoder/model.safetensors",
        ]
        variant = "fp16"
        self.assertFalse(is_safetensors_compatible(filenames, variant=variant))

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
        variant = "fp16"
        self.assertFalse(is_safetensors_compatible(filenames, variant=variant))

    def test_transformers_is_compatible_sharded(self):
        filenames = [
            "text_encoder/pytorch_model.bin",
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/model-00002-of-00002.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames, variant=None))

    def test_transformers_is_compatible_variant_sharded(self):
        filenames = [
            "text_encoder/pytorch_model.bin",
            "text_encoder/model.fp16-00001-of-00002.safetensors",
            "text_encoder/model.fp16-00001-of-00002.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames, variant=None))

    def test_diffusers_is_compatible_sharded(self):
        filenames = [
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model-00001-of-00002.safetensors",
            "unet/diffusion_pytorch_model-00002-of-00002.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames, variant=None))

    def test_diffusers_is_compatible_variant_sharded(self):
        filenames = [
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
            "unet/diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames, variant=None))

    def test_diffusers_is_compatible_only_variants(self):
        filenames = [
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames, variant=None))
