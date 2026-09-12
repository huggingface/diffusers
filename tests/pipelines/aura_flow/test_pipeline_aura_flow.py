import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers import AuraFlowPipeline, AuraFlowTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler

from ..testing_utils import (
    BasePipelineTesterConfig,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


class AuraFlowPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = AuraFlowPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    # `height` / `width` default to `transformer.config.sample_size * vae_scale_factor` (32 * 2).
    output_shape = (3, 64, 64)

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

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": None,
            "width": None,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestAuraFlowPipeline(AuraFlowPipelineTesterConfig, PipelineTesterMixin):
    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-3):
        # AuraFlow pads the prompt embeddings to a common length, so batched and single runs diverge slightly more.
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)


class TestAuraFlowPipelineMemory(AuraFlowPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the AuraFlow pipeline."""


class TestAuraFlowPipelineLoRA(AuraFlowPipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the AuraFlow pipeline."""

    # Adapting the attention projections alone barely moves the output of this tiny AuraFlow (max diff ~3e-5, below
    # the tolerances the tests assert against), so the feed-forward `linear_1` layers are adapted as well.
    denoiser_target_modules = {"transformer": ["to_q", "to_k", "to_v", "to_out.0", "linear_1"]}


class TestAuraFlowPipelineLoRAMemory(AuraFlowPipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the AuraFlow pipeline."""

    # See `TestAuraFlowPipelineLoRA`.
    denoiser_target_modules = {"transformer": ["to_q", "to_k", "to_v", "to_out.0", "linear_1"]}
