import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers import AuraFlowPipeline, AuraFlowTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.utils.testing_utils import (
    torch_device,
)

from ..test_pipelines_common import PipelineTesterMixin


class AuraFlowPipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = AuraFlowPipeline
    params = frozenset(
        [
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
    )
    batch_params = frozenset(["prompt", "negative_prompt"])

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
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "height": None,
            "width": None,
        }
        return inputs

    def test_aura_flow_prompt_embeds(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        output_with_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = inputs.pop("prompt")

        do_classifier_free_guidance = inputs["guidance_scale"] > 1
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipe.encode_prompt(
            prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            device=torch_device,
        )
        output_with_embeds = pipe(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **inputs,
        ).images[0]

        max_diff = np.abs(output_with_prompt - output_with_embeds).max()
        assert max_diff < 1e-4

    def test_attention_slicing_forward_pass(self):
        # Attention slicing needs to implemented differently for this because how single DiT and MMDiT
        # blocks interfere with each other.
        return
