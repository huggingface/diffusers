import unittest

import numpy as np
import torch
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model

from diffusers.models.transformers.transformer_prx import PRXTransformer2DModel
from diffusers.pipelines.prx.pipeline_prx_pixel import PRXPixelPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from ..pipeline_params import TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


class PRXPixelPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    # PRXPixelPipeline is standalone: it inherits from DiffusionPipeline (not PRXPipeline) and always has its own
    # image_processor, so it denoises raw RGB in pixel space and supports output_type="pil"/"np" without a VAE.
    pipeline_class = PRXPixelPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = frozenset(["prompt", "negative_prompt", "num_images_per_prompt"])
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    def get_dummy_components(self):
        torch.manual_seed(0)
        # Pixel-space PRX: in_channels=3 (RGB), bottleneck img_in, resolution_embeds=True.
        # context_in_dim must match the text encoder hidden_size (16).
        transformer = PRXTransformer2DModel(
            patch_size=1,
            in_channels=3,
            context_in_dim=16,
            hidden_size=8,
            mlp_ratio=2.0,
            num_heads=2,
            depth=1,
            axes_dim=[2, 2],
            bottleneck_size=8,
            resolution_embeds=True,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler()

        # Tiny Qwen3 text encoder returning `last_hidden_state` (Qwen3-VL-style backbone).
        torch.manual_seed(0)
        config = Qwen3Config(
            hidden_size=16,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=151936,
            max_position_embeddings=512,
        )
        text_encoder = Qwen3Model(config)
        tokenizer = Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

        return {
            "transformer": transformer,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "prompt_max_tokens": 16,
        }

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        return {
            "prompt": "",
            "negative_prompt": "",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 32,
            "width": 32,
            # Pixel-space PRX has no VAE and returns raw (C, H, W) tensors for output_type="pt". The generic
            # PipelineTesterMixin tests compare these tensors directly, so default to "pt" here; the PIL/np default
            # path is exercised explicitly in test_inference and test_inference_pil_and_np_output.
            "output_type": "pt",
            # 32px is not in the 1024 aspect-ratio bins, so binning must be disabled for these tiny fast tests.
            "use_resolution_binning": False,
        }

    def _build_pipe(self, device="cpu"):
        components = self.get_dummy_components()
        pipe = PRXPixelPipeline(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)
        return pipe

    def test_inference(self):
        device = "cpu"
        pipe = self._build_pipe(device)

        # Pixel space: vae_scale_factor is always 1, and the pipeline always carries an image processor
        # so postprocessing (and the default output_type="pil") works without any VAE.
        self.assertEqual(pipe.vae_scale_factor, 1)
        self.assertIsNotNone(pipe.image_processor)

        # Default output is PIL (no VAE needed: the image processor denormalizes the denoised pixels directly).
        inputs = self.get_dummy_inputs(device)
        inputs.pop("output_type")  # default is "pil"
        images = pipe(**inputs).images
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].size, (32, 32))

        # Raw "pt" output is the denoised RGB tensor at the requested resolution.
        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs)[0]
        generated_image = image[0]
        self.assertEqual(generated_image.shape, (3, 32, 32))
        expected_image = torch.zeros(3, 32, 32)
        max_diff = np.abs(generated_image.cpu().numpy() - expected_image.numpy()).max()
        self.assertLessEqual(max_diff, 1e10)

    def test_inference_batch(self):
        device = "cpu"
        pipe = self._build_pipe(device)

        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = ["", ""]
        inputs["negative_prompt"] = ["", ""]
        image = pipe(**inputs)[0]

        self.assertEqual(image.shape[0], 2)
        self.assertEqual(tuple(image.shape[1:]), (3, 32, 32))

    def test_inference_with_cfg(self):
        device = "cpu"
        pipe = self._build_pipe(device)

        # CFG off.
        inputs = self.get_dummy_inputs(device)
        inputs["guidance_scale"] = 1.0
        out_no_cfg = pipe(**inputs)[0]
        self.assertFalse(pipe.do_classifier_free_guidance)
        self.assertEqual(out_no_cfg[0].shape, (3, 32, 32))

        # CFG on.
        inputs = self.get_dummy_inputs(device)
        inputs["guidance_scale"] = 5.0
        out_cfg = pipe(**inputs)[0]
        self.assertTrue(pipe.do_classifier_free_guidance)
        self.assertEqual(out_cfg[0].shape, (3, 32, 32))

        # Guidance should actually change the output.
        max_diff = np.abs(out_no_cfg.cpu().numpy() - out_cfg.cpu().numpy()).max()
        self.assertGreater(max_diff, 0.0)

    def test_inference_with_prompt_embeds(self):
        device = "cpu"
        pipe = self._build_pipe(device)

        # Precompute embeddings via the public encode_prompt API (CFG on so we get negatives too).
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
            pipe.encode_prompt(
                prompt="a prompt",
                device=device,
                do_classifier_free_guidance=True,
                negative_prompt="",
            )
        )

        inputs = self.get_dummy_inputs(device)
        inputs.pop("prompt")
        inputs.pop("negative_prompt")
        inputs["guidance_scale"] = 5.0
        inputs["prompt_embeds"] = prompt_embeds
        inputs["negative_prompt_embeds"] = negative_prompt_embeds
        inputs["prompt_attention_mask"] = prompt_attention_mask
        inputs["negative_prompt_attention_mask"] = negative_prompt_attention_mask

        image = pipe(**inputs)[0]
        self.assertEqual(image[0].shape, (3, 32, 32))

    def test_inference_pil_and_np_output(self):
        # The default output_type="pil" must work without a VAE: the denoised pixels are denormalized
        # directly by the image processor instead of being decoded.
        device = "cpu"
        pipe = self._build_pipe(device)

        inputs = self.get_dummy_inputs(device)
        inputs.pop("output_type")  # default is "pil"
        images = pipe(**inputs).images
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].size, (32, 32))

        inputs = self.get_dummy_inputs(device)
        inputs["output_type"] = "np"
        images = pipe(**inputs).images
        self.assertEqual(images.shape, (1, 32, 32, 3))
        self.assertGreaterEqual(images.min(), 0.0)
        self.assertLessEqual(images.max(), 1.0)

    def test_non_multiple_size_raises(self):
        # height/width must be divisible by vae_scale_factor * transformer patch_size; check_inputs must raise
        # a clear ValueError instead of letting the transformer fail on an invalid reshape mid-denoising.
        device = "cpu"
        components = self.get_dummy_components()
        torch.manual_seed(0)
        components["transformer"] = PRXTransformer2DModel(
            patch_size=2,
            in_channels=3,
            context_in_dim=16,
            hidden_size=8,
            mlp_ratio=2.0,
            num_heads=2,
            depth=1,
            axes_dim=[2, 2],
            bottleneck_size=8,
            resolution_embeds=True,
        )
        pipe = PRXPixelPipeline(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["height"] = 31  # vae_scale_factor (1) * patch_size (2) = 2; 31 is not a multiple
        with self.assertRaisesRegex(ValueError, "divisible"):
            pipe(**inputs)

    def test_callback_inputs(self):
        device = "cpu"
        pipe = self._build_pipe(device)
        self.assertTrue(
            hasattr(pipe, "_callback_tensor_inputs"),
            f" {PRXPixelPipeline} should have `_callback_tensor_inputs` that defines a list of tensor variables its"
            " callback function can use as inputs",
        )

        def callback_inputs_subset(pipe, i, t, callback_kwargs):
            for tensor_name in callback_kwargs.keys():
                assert tensor_name in pipe._callback_tensor_inputs
            return callback_kwargs

        def callback_inputs_all(pipe, i, t, callback_kwargs):
            for tensor_name in pipe._callback_tensor_inputs:
                assert tensor_name in callback_kwargs
            for tensor_name in callback_kwargs.keys():
                assert tensor_name in pipe._callback_tensor_inputs
            return callback_kwargs

        inputs = self.get_dummy_inputs(device)
        inputs["callback_on_step_end"] = callback_inputs_subset
        inputs["callback_on_step_end_tensor_inputs"] = ["latents"]
        _ = pipe(**inputs)[0]

        inputs = self.get_dummy_inputs(device)
        inputs["callback_on_step_end"] = callback_inputs_all
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        _ = pipe(**inputs)[0]

    def test_attention_slicing_forward_pass(self, expected_max_diff=1e-3):
        # Overridden: the mixin version calls assert_mean_pixel_difference, which assumes HWC image
        # arrays. Pixel-space PRX has no VAE; compare raw (C, H, W) tensors directly ("pt") instead of
        # going through PIL.
        if not self.test_attention_slicing:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to("cpu")
        pipe.set_progress_bar_config(disable=None)

        def to_np_local(tensor):
            if isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu().numpy()
            return tensor

        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        output_without_slicing = pipe(**inputs)[0]

        pipe.enable_attention_slicing(slice_size=1)
        inputs = self.get_dummy_inputs(generator_device)
        output_with_slicing1 = pipe(**inputs)[0]

        pipe.enable_attention_slicing(slice_size=2)
        inputs = self.get_dummy_inputs(generator_device)
        output_with_slicing2 = pipe(**inputs)[0]

        max_diff1 = np.abs(to_np_local(output_with_slicing1) - to_np_local(output_without_slicing)).max()
        max_diff2 = np.abs(to_np_local(output_with_slicing2) - to_np_local(output_without_slicing)).max()
        self.assertLess(max(max_diff1, max_diff2), expected_max_diff)

    @unittest.skip("Slow original-vs-diffusers parity test is optional and intentionally skipped for fast CI.")
    def test_prx_pixel_original_parity(self):
        pass
