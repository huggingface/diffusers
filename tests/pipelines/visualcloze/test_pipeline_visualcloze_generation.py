import random

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
    VisualClozeGenerationPipeline,
)

from ...testing_utils import floats_tensor, torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


class VisualClozeGenerationPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = VisualClozeGenerationPipeline
    required_input_params_in_call_signature = frozenset(
        [
            "task_prompt",
            "content_prompt",
            "guidance_scale",
            "prompt_embeds",
            "pooled_prompt_embeds",
        ]
    )
    batch_input_params = frozenset(["task_prompt", "content_prompt", "image"])
    output_shape = (3, 32, 32)

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
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder_2 = T5EncoderModel(config)

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

    def get_dummy_inputs(self, seed=0):
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

        inputs = {
            "task_prompt": "Each row outlines a logical process, starting from [IMAGE1] gray-based depth map with detailed object contours, to achieve [IMAGE2] an image with flawless clarity.",
            "content_prompt": "A beautiful landscape with mountains and a lake",
            "image": image,
            "generator": self.get_generator(seed),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "max_sequence_length": 77,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }
        return inputs


class TestVisualClozeGenerationPipeline(VisualClozeGenerationPipelineTesterConfig, PipelineTesterMixin):
    def test_visualcloze_different_task_prompts(self, expected_min_diff=1e-1):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        output_original = pipe(**inputs).images[0]

        inputs["task_prompt"] = "A different task description for image generation"
        output_different_task = pipe(**inputs).images[0]

        # Different task prompts should produce different outputs
        max_diff = (output_original - output_different_task).abs().max()
        assert max_diff > expected_min_diff, "Outputs should be different for different task prompts."

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    @pytest.mark.skip(
        "`encode_prompt` requires a `layout_prompt` that the pipeline derives from the image grid and that is not a "
        "`__call__` argument, and `__call__` cannot run without `task_prompt`, so the prompt encoder cannot be "
        "swapped out the way this test does."
    )
    def test_encode_prompt_works_in_isolation(self):
        pass


class TestVisualClozeGenerationPipelineMemory(VisualClozeGenerationPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the VisualCloze pipeline."""
