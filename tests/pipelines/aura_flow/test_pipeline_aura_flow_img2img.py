import unittest

import numpy as np
import PIL.Image
import torch
from diffusers.utils.testing_utils import require_torch_gpu, torch_device
from transformers import AutoTokenizer, UMT5EncoderModel, AuraFlowPipelineFastTests

from diffusers import (
    AuraFlowImg2ImgPipeline,  # Added for Img2Img
    AuraFlowPipeline,
    AuraFlowTransformer2DModel,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)

from ..test_pipelines_common import (
    PipelineTesterMixin,
    check_qkv_fusion_matches_attn_procs_length,
    check_qkv_fusion_processors_exist,
)

class AuraFlowImg2ImgPipelineFastTests(AuraFlowPipelineFastTests):
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
    batch_params = frozenset(["prompt", "negative_prompt", "image"])
    test_layerwise_casting = False # T5 uses multiple devices
    test_group_offloading = False # T5 uses multiple devices

    # Redefine get_dummy_inputs for Img2Img
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

    # Override T2I test that requires height/width
    def test_fused_qkv_projections(self):
        # Inherited test expects height/width, skip for img2img dummy inputs
        # Call the parent T2I test method directly if needed for coverage,
        # but adapt inputs or skip if incompatible.
        # For now, simply reimplement with img2img inputs
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
        assert check_qkv_fusion_matches_attn_procs_length(
            pipe.transformer, pipe.transformer.original_attn_processors
        )

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


    def test_aura_flow_img2img_output_shape(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(torch_device)

        # Use dimensions divisible by vae_scale_factor * patch_size (8*2=16)
        height_width_pairs = [(64, 64), (128, 48)] # 48 is divisible by 16

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