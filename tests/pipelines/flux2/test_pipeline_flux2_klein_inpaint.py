import random
import unittest

import numpy as np
import torch
from transformers import Qwen2TokenizerFast, Qwen3Config, Qwen3ForCausalLM

from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinInpaintPipeline,
    Flux2Transformer2DModel,
)

from ...testing_utils import (
    enable_full_determinism,
    floats_tensor,
    torch_device,
)
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class Flux2KleinInpaintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = Flux2KleinInpaintPipeline
    params = frozenset(["prompt", "height", "width", "guidance_scale", "prompt_embeds"])
    batch_params = frozenset(["prompt"])

    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    supports_dduf = False

    def get_dummy_components(self, num_layers: int = 1, num_single_layers: int = 1):
        torch.manual_seed(0)
        transformer = Flux2Transformer2DModel(
            patch_size=1,
            in_channels=4,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=16,
            timestep_guidance_channels=256,
            axes_dims_rope=[4, 4, 4, 4],
            guidance_embeds=False,
        )

        # Create minimal Qwen3 config
        config = Qwen3Config(
            intermediate_size=16,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=151936,
            max_position_embeddings=512,
        )
        torch.manual_seed(0)
        text_encoder = Qwen3ForCausalLM(config)

        # Use a simple tokenizer for testing
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"
        )

        torch.manual_seed(0)
        vae = AutoencoderKLFlux2(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=1,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
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
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        mask_image = torch.ones((1, 1, 32, 32)).to(device)

        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 8.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 64,
            "strength": 0.8,
            "output_type": "np",
            "text_encoder_out_layers": (1,),
        }
        return inputs

    def test_flux2_klein_inpaint_different_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt"] = "a different prompt"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = np.abs(output_same_prompt - output_different_prompts).max()

        # Outputs should be different here
        assert max_diff > 1e-6

    def test_flux2_klein_inpaint_image_output_shape(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        height_width_pairs = [(32, 32), (72, 56)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            # Update image and mask to match height/width
            image = floats_tensor((1, 3, height, width), rng=random.Random(0)).to(torch_device)
            mask_image = torch.ones((1, 1, height, width)).to(torch_device)

            inputs.update({"height": height, "width": width, "image": image, "mask_image": mask_image})
            image = pipe(**inputs).images[0]
            output_height, output_width, _ = image.shape
            self.assertEqual(
                (output_height, output_width),
                (expected_height, expected_width),
                f"Output shape {image.shape} does not match expected shape {(expected_height, expected_width)}",
            )

    def test_flux2_klein_inpaint_strength(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)

        # Test with strength=1.0 (full denoising)
        inputs = self.get_dummy_inputs(torch_device)
        inputs["strength"] = 1.0
        output_full_strength = pipe(**inputs).images[0]

        # Test with strength=0.5 (partial denoising)
        inputs = self.get_dummy_inputs(torch_device)
        inputs["strength"] = 0.5
        output_half_strength = pipe(**inputs).images[0]

        max_diff = np.abs(output_full_strength - output_half_strength).max()

        # Outputs should be different with different strength values
        assert max_diff > 1e-6

    @unittest.skip("Needs to be revisited")
    def test_encode_prompt_works_in_isolation(self):
        pass
