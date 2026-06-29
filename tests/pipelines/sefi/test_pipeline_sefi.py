import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SeFiPipeline, SeFiTransformer2DModel

from ...testing_utils import torch_device
from ..test_pipelines_common import PipelineTesterMixin


_TEXT_DIM = 16


def _build_tiny_text_encoder():
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": _TEXT_DIM,
            "intermediate_size": _TEXT_DIM,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "rope_scaling": {
                "mrope_section": [1, 1, 2],
                "rope_type": "default",
                "type": "default",
            },
            "rope_theta": 1000000.0,
            "vocab_size": 151936,
            "head_dim": 8,
        },
        vision_config={
            "depth": 2,
            "hidden_size": _TEXT_DIM,
            "intermediate_size": _TEXT_DIM,
            "num_heads": 2,
            "out_channels": _TEXT_DIM,
            "out_hidden_size": _TEXT_DIM,
            "patch_size": 14,
        },
    )
    return Qwen3VLForConditionalGeneration(config).eval()


class SeFiPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = SeFiPipeline
    params = frozenset(["prompt", "height", "width", "guidance_scale", "prompt_embeds"])
    batch_params = frozenset(["prompt", "num_images_per_prompt"])
    required_optional_params = frozenset(["num_inference_steps", "generator", "output_type", "return_dict"])
    test_xformers_attention = False
    test_attention_slicing = False
    test_layerwise_casting = False
    test_group_offloading = False
    supports_dduf = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = SeFiTransformer2DModel(
            patch_size=1,
            in_channels=8,
            out_channels=8,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=_TEXT_DIM,
            timestep_guidance_channels=16,
            axes_dims_rope=[2, 2, 2, 2],
        )

        torch.manual_seed(0)
        vae = AutoencoderKL(
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
            shift_factor=0.0,
            scaling_factor=1.0,
        )

        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

        torch.manual_seed(0)
        text_encoder = _build_tiny_text_encoder()
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

        return {
            "transformer": transformer,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "semantic_channels": 4,
            "texture_vae_name": "sd1.5",
            "default_guidance_scale": 1.0,
            "default_num_inference_steps": 2,
            "text_encoder_hidden_layers": "1",
            "max_sequence_length": 8,
        }

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        return {
            "prompt": "a small dog",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 8,
            "text_encoder_hidden_layers": (1,),
            "output_type": "np",
        }

    def test_inference(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        image = pipe(**self.get_dummy_inputs(torch_device)).images
        assert image.shape == (1, 32, 32, 3)
        image_slice = image[0, -3:, -3:, -1]
        assert np.isfinite(image_slice).all()

    def test_turbo_validation(self):
        components = self.get_dummy_components()
        components["is_turbo"] = True
        components["default_num_inference_steps"] = 4
        pipe = self.pipeline_class(**components).to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = 2
        with self.assertRaises(ValueError):
            pipe(**inputs)

    def test_prompt_embeds_cfg_batch(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        prompt_embeds = torch.randn(2, 8, _TEXT_DIM, device=torch_device)
        image = pipe(
            prompt_embeds=prompt_embeds,
            num_inference_steps=1,
            guidance_scale=2.0,
            height=32,
            width=32,
            output_type="np",
        ).images

        assert image.shape == (2, 32, 32, 3)

    def test_text_encoder_rotary_dtype_alignment(self):
        components = self.get_dummy_components()
        components["text_encoder"].to(dtype=torch.bfloat16)
        pipe = self.pipeline_class(**components)

        rotary_emb = pipe.text_encoder.model.language_model.rotary_emb
        rotary_emb.to(dtype=torch.float32)
        assert rotary_emb.inv_freq.dtype == torch.float32

        pipe._align_text_encoder_rotary_dtype(torch.device("cpu"))

        assert rotary_emb.inv_freq.dtype == torch.bfloat16
        assert rotary_emb.original_inv_freq.dtype == torch.bfloat16
