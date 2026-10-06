import torch
from transformers import AutoTokenizer, Gemma2Config, Gemma2Model

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    Lumina2Pipeline,
    Lumina2Transformer2DModel,
)

from ...testing_utils import assert_tensors_close
from ..testing_utils import (
    BasePipelineTesterConfig,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


class Lumina2PipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = Lumina2Pipeline
    required_input_params_in_call_signature = frozenset(
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
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    # The dummy one-block VAE decodes the 4x4 latents at scale 1, so requested 32x32 comes out 4x4
    output_shape = (3, 4, 4)

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

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 32,
            "width": 32,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestLumina2Pipeline(Lumina2PipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        generated_image = image[0]
        assert generated_image.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.4409, 0.6402, 0.1740, 0.4674, 0.4631, 0.3840, 0.5556, 0.4289, 0.4979, 0.4755, 0.5825, 0.6095, 0.7116, 0.5101, 0.6170, 0.6536])
        # fmt: on

        generated_slice = generated_image.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert_tensors_close(generated_slice, expected_slice, atol=1e-3)


class TestLumina2PipelineMemory(Lumina2PipelineTesterConfig, MemoryTesterMixin):
    pass


class TestLumina2PipelineLoRA(Lumina2PipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the Lumina2 pipeline."""


class TestLumina2PipelineLoRAMemory(Lumina2PipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA offloading tests for the Lumina2 pipeline."""
