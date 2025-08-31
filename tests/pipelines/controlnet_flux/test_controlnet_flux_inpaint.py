import random
import unittest

import numpy as np
import torch

# torch_device,  # {{ edit_1 }} Removed unused import
from transformers import (
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlNetInpaintPipeline,
    FluxControlNetModel,
    FluxTransformer2DModel,
)
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import (
    enable_full_determinism,
    floats_tensor,
    torch_device,
)
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class FluxControlNetInpaintPipelineTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = FluxControlNetInpaintPipeline
    params = frozenset(
        [
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "prompt_embeds",
            "pooled_prompt_embeds",
            "image",
            "mask_image",
            "control_image",
            "strength",
            "num_inference_steps",
            "controlnet_conditioning_scale",
        ]
    )
    batch_params = frozenset(["prompt", "image", "mask_image", "control_image"])
    test_xformers_attention = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=8,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
        )
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )

        torch.manual_seed(0)
        text_encoder = CLIPTextModel(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_2 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=2,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        torch.manual_seed(0)
        controlnet = FluxControlNetModel(
            patch_size=1,
            in_channels=8,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "transformer": transformer,
            "vae": vae,
            "controlnet": controlnet,
        }

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        mask_image = torch.ones((1, 1, 32, 32)).to(device)
        control_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "mask_image": mask_image,
            "control_image": control_image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 48,
            "strength": 0.8,
            "output_type": "np",
        }
        return inputs

    def test_flux_controlnet_inpaint_with_num_images_per_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["num_images_per_prompt"] = 2
        output = pipe(**inputs)
        images = output.images

        assert images.shape == (2, 32, 32, 3)

    def test_flux_controlnet_inpaint_with_controlnet_conditioning_scale(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output_default = pipe(**inputs)
        image_default = output_default.images

        inputs["controlnet_conditioning_scale"] = 0.5
        output_scaled = pipe(**inputs)
        image_scaled = output_scaled.images

        # Ensure that changing the controlnet_conditioning_scale produces a different output
        assert not np.allclose(image_default, image_scaled, atol=0.01)

    def test_attention_slicing_forward_pass(self):
        super().test_attention_slicing_forward_pass(expected_max_diff=3e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    def test_flux_image_output_shape(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        height_width_pairs = [(32, 32), (72, 56)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            inputs.update(
                {
                    "control_image": randn_tensor(
                        (1, 3, height, width),
                        device=torch_device,
                        dtype=torch.float16,
                    ),
                    "image": randn_tensor(
                        (1, 3, height, width),
                        device=torch_device,
                        dtype=torch.float16,
                    ),
                    "mask_image": torch.ones((1, 1, height, width)).to(torch_device),
                    "height": height,
                    "width": width,
                }
            )
            image = pipe(**inputs).images[0]
            output_height, output_width, _ = image.shape
            assert (output_height, output_width) == (expected_height, expected_width)
