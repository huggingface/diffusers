import unittest

import torch
from transformers import AutoTokenizer, Gemma2Config, Gemma2Model

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    Lumina2Pipeline,
    Lumina2Transformer2DModel,
)

from ..test_pipelines_common import PipelineTesterMixin


class Lumina2PipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = Lumina2Pipeline
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
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "return_dict",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        ]
    )

    supports_dduf = False
    test_xformers_attention = False
    test_layerwise_casting = True

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = Lumina2Transformer2DModel(
            sample_size=4,
            patch_size=2,
            in_channels=4,
            hidden_size=8,
            num_layers=2,
            num_attention_heads=1,
            num_kv_heads=1,
            multiple_of=16,
            ffn_dim_multiplier=None,
            norm_eps=1e-5,
            scaling_factor=1.0,
            axes_dim_rope=[4, 2, 2],
            cap_feat_dim=8,
        )

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=4,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/dummy-gemma")

        torch.manual_seed(0)
        config = Gemma2Config(
            head_dim=4,
            hidden_size=8,
            intermediate_size=8,
            num_attention_heads=2,
            num_hidden_layers=2,
            num_key_value_heads=2,
            sliding_window=2,
        )
        text_encoder = Gemma2Model(config)

        components = {
            "transformer": transformer,
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        return components

    def test_image_seq_len_uses_spatial_dimensions(self):
        """Test that image_seq_len is computed from spatial dims, not channel dim.

        Lumina2 latents have shape (batch, channels, height, width) and are NOT
        packed before image_seq_len is computed. The transformer patchifies
        internally with patch_size=2, so the correct sequence length is
        (H // patch_size) * (W // patch_size).

        Previously, the code used latents.shape[1] which gives the channel
        count (e.g. 4) instead of the spatial sequence length (e.g. 64 for
        16x16 latents with patch_size=2). This caused calculate_shift() to
        compute a completely wrong mu value for the scheduler.
        """
        components = self.get_dummy_components()
        pipe = Lumina2Pipeline(**components)
        pipe.to(torch.device("cpu"))

        patch_size = pipe.transformer.config.patch_size  # 2

        # Use height=32, width=32 -> latent size 4x4 (vae downscale 8x)
        # With patch_size=2: seq_len = (4//2)*(4//2) = 4
        # Channel dim = 4, which would be wrong if used as seq_len
        # Use a larger size to make the distinction clearer
        height, width = 64, 64
        latent_h, latent_w = height // 8, width // 8  # 8, 8
        expected_seq_len = (latent_h // patch_size) * (latent_w // patch_size)  # 16

        # The channel dimension is 4 (from vae latent_channels)
        # If the bug were present, image_seq_len would be 4 instead of 16
        channels = components["vae"].config.latent_channels  # 4
        self.assertNotEqual(channels, expected_seq_len, "Test needs channels != expected_seq_len to be meaningful")

        # Capture the mu value passed to the scheduler
        captured = {}
        original_set_timesteps = pipe.scheduler.set_timesteps

        def capture_mu_set_timesteps(*args, **kwargs):
            captured["mu"] = kwargs.get("mu")
            return original_set_timesteps(*args, **kwargs)

        pipe.scheduler.set_timesteps = capture_mu_set_timesteps

        # Run pipeline with specific dimensions
        generator = torch.Generator(device="cpu").manual_seed(0)
        pipe(
            prompt="test",
            height=height,
            width=width,
            num_inference_steps=1,
            generator=generator,
            output_type="latent",
        )

        # Verify mu was computed using spatial seq_len, not channel dim
        from diffusers.pipelines.lumina2.pipeline_lumina2 import calculate_shift

        correct_mu = calculate_shift(expected_seq_len)
        wrong_mu = calculate_shift(channels)

        self.assertAlmostEqual(captured["mu"], correct_mu, places=5, msg="mu should use spatial sequence length")
        self.assertNotAlmostEqual(captured["mu"], wrong_mu, places=5, msg="mu should NOT use channel dimension")

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
            "height": 32,
            "width": 32,
            "output_type": "np",
        }
        return inputs
