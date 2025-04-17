import unittest

import numpy as np
import PIL.Image
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers import (
    AuraFlowImg2ImgPipeline,
    AuraFlowTransformer2DModel,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.testing_utils import torch_device

from ..test_pipelines_common import (
    PipelineTesterMixin,
    check_qkv_fusion_matches_attn_procs_length,
    check_qkv_fusion_processors_exist,
)


class AuraFlowImg2ImgPipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = AuraFlowImg2ImgPipeline
    params = frozenset(
        [
            "prompt",
            "image",
            "strength",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
    )
    batch_params = frozenset(["prompt", "image", "negative_prompt"])
    test_layerwise_casting = False  # T5 uses multiple devices
    test_group_offloading = False  # T5 uses multiple devices

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = AuraFlowTransformer2DModel(
            sample_size=32,
            patch_size=2,
            in_channels=4,
            num_mmdit_layers=1,
            num_single_dit_layers=1,
            attention_head_dim=8,
            num_attention_heads=4,
            caption_projection_dim=32,
            joint_attention_dim=32,
            out_channels=4,
            pos_embed_max_size=256,
        )

        text_encoder = UMT5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-umt5")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=32,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer": transformer,
            "vae": vae,
        }

    def get_dummy_inputs(self, device, seed=0):
        # Ensure image dimensions are divisible by VAE scale factor * transformer patch size
        # vae_scale_factor = 8, patch_size = 2 => divisible by 16
        image = PIL.Image.new("RGB", (64, 64))
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "strength": 0.75,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            # height/width are inferred from image in img2img
        }
        return inputs

    def test_attention_slicing_forward_pass(self):
        # Attention slicing needs to implemented differently for this because how single DiT and MMDiT
        # blocks interfere with each other.
        return

    def test_fused_qkv_projections(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        original_image_slice = image[0, -3:, -3:, -1]

        pipe.transformer.fuse_qkv_projections()
        assert check_qkv_fusion_processors_exist(pipe.transformer)
        assert check_qkv_fusion_matches_attn_procs_length(pipe.transformer, pipe.transformer.original_attn_processors)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice_fused = image[0, -3:, -3:, -1]

        pipe.transformer.unfuse_qkv_projections()
        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice_disabled = image[0, -3:, -3:, -1]

        assert np.allclose(original_image_slice, image_slice_fused, atol=1e-3, rtol=1e-3)
        assert np.allclose(image_slice_fused, image_slice_disabled, atol=1e-3, rtol=1e-3)
        assert np.allclose(original_image_slice, image_slice_disabled, atol=1e-2, rtol=1e-2)

    @unittest.skip("xformers attention processor does not exist for AuraFlow")
    def test_xformers_attention_forwardGenerator_pass(self):
        pass

    def test_aura_flow_img2img_output_shape(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(torch_device)

        # The positional embedding has a max size of 256
        # Each position is a (height/vae_scale_factor/patch_size) × (width/vae_scale_factor/patch_size) grid
        # To stay within limits: (height/8/2) * (width/8/2) < 256
        height_width_pairs = [(32, 32), (64, 32)]  # creates 4 and 16 positions respectively

        for height, width in height_width_pairs:
            inputs = self.get_dummy_inputs(torch_device)
            # Override dummy image size
            inputs["image"] = PIL.Image.new("RGB", (width, height))
            # Pass height/width explicitly to test pipeline handles them (though inferred by default)
            inputs["height"] = height
            inputs["width"] = width

            output = pipe(**inputs)
            image = output.images[0]

            # Expected shape is (height, width, 3) for np output
            self.assertEqual(image.shape, (height, width, 3))

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=0.001):
        self._test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_num_images_per_prompt(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        batch_sizes = [1]
        num_images_per_prompts = [1, 2]

        for batch_size in batch_sizes:
            for num_images_per_prompt in num_images_per_prompts:
                inputs = self.get_dummy_inputs(torch_device)
                inputs["num_inference_steps"] = 2

                inputs["image"] = PIL.Image.new("RGB", (32, 32))

                for key in inputs.keys():
                    if key in self.batch_params:
                        inputs[key] = batch_size * [inputs[key]]

                images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images

                assert len(images) == batch_size * num_images_per_prompt
