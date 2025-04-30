import random
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel, VisualClozeGenerationPipeline
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    torch_device,
)

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class VisualClozePipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = VisualClozeGenerationPipeline
    params = frozenset(
        [
            "task_prompt",
            "content_prompt",
            "guidance_scale",
            "prompt_embeds",
            "pooled_prompt_embeds",
        ]
    )
    batch_params = frozenset(["task_prompt", "content_prompt", "image"])
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=12,
            out_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=6,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[2, 2, 2],
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
            latent_channels=1,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
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
            "resolution": 32,
        }

    def get_dummy_inputs(self, device, seed=0):
        # Create example images to simulate the input format required by VisualCloze
        context_image = [
            Image.fromarray(floats_tensor((32, 32, 3), rng=random.Random(seed), scale=255).numpy().astype(np.uint8))
            for _ in range(2)
        ]
        query_image = [
            Image.fromarray(
                floats_tensor((32, 32, 3), rng=random.Random(seed + 1), scale=255).numpy().astype(np.uint8)
            ),
            None,
        ]

        # Create an image list that conforms to the VisualCloze input format
        image = [
            context_image,  # In-Context example
            query_image,  # Query image
        ]

        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        inputs = {
            "task_prompt": "Each row outlines a logical process, starting from [IMAGE1] gray-based depth map with detailed object contours, to achieve [IMAGE2] an image with flawless clarity.",
            "content_prompt": "A beautiful landscape with mountains and a lake",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "max_sequence_length": 48,
            "output_type": "np",
        }
        return inputs

    def test_visualcloze_different_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["content_prompt"] = "A different landscape with forests and rivers"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = np.abs(output_same_prompt - output_different_prompts).max()

        # Outputs should be different
        assert max_diff > 1e-6

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=1e-3)

    def test_different_task_prompts(self, expected_min_diff=1e-1):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        output_original = pipe(**inputs).images[0]

        inputs["task_prompt"] = "A different task description for image generation"
        output_different_task = pipe(**inputs).images[0]

        # Different task prompts should produce different outputs
        max_diff = np.abs(output_original - output_different_task).max()
        assert max_diff > expected_min_diff
