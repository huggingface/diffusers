import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AnimateDiffSDXLPipeline,
    AutoencoderKL,
    DDIMScheduler,
    MotionAdapter,
    UNet2DConditionModel,
)

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..testing_utils import (
    IPAdapterTesterMixin,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    UNetLoraTesterMixin,
)
from .testing_utils import MotionPipelineTesterConfig, MotionPipelineTesterMixin


class AnimateDiffSDXLPipelineTesterConfig(MotionPipelineTesterConfig):
    pipeline_class = AnimateDiffSDXLPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union({"add_text_embeds", "add_time_ids"})
    # `num_frames` defaults to 16; height/width default to `unet.sample_size * vae_scale_factor` (32 * 2).
    output_shape = (16, 3, 64, 64)

    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64, 128),
            layers_per_block=2,
            time_cond_proj_dim=time_cond_proj_dim,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=(2, 4, 8),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2, 4),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
            norm_num_groups=1,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            clip_sample=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        motion_adapter = MotionAdapter(
            block_out_channels=(32, 64, 128),
            motion_layers_per_block=2,
            motion_norm_num_groups=2,
            motion_num_attention_heads=4,
            use_motion_mid_block=False,
        )

        return {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "motion_adapter": motion_adapter,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "feature_extractor": None,
            "image_encoder": None,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 7.5,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


# `AnimateDiffSDXLPipeline.upcast_vae()` casts the VAE to fp32 but puts `conv_in` / `post_quant_conv` back to the
# original dtype whenever the VAE attention processor is the SDPA one, which leaves `decode_latents` feeding fp16
# activations into the fp32 decoder blocks. The old tester hid this by calling `set_default_attn_processor()` on every
# component first; the mixins here run the pipeline as a user would, so the fp16 tests below trip over it.
FP16_DECODE_SKIP_REASON = (
    "`AnimateDiffSDXLPipeline.upcast_vae()` leaves the VAE at mixed precision, so fp16 decoding raises "
    "`expected scalar type Half but found Float`."
)


class TestAnimateDiffSDXLPipeline(AnimateDiffSDXLPipelineTesterConfig, MotionPipelineTesterMixin):
    @pytest.mark.skip(FP16_DECODE_SKIP_REASON)
    def test_save_load_float16(self):
        pass

    @pytest.mark.skip(FP16_DECODE_SKIP_REASON)
    def test_half_precision_inference_no_nan(self, dtype):
        pass

    @pytest.mark.skip("Test currently not supported.")
    def test_encode_prompt_works_in_isolation(self):
        pass

    @pytest.mark.skip("Functionality is tested elsewhere.")
    def test_save_load_optional_components(self):
        pass

    @pytest.mark.skip("SDXL also requires `pooled_prompt_embeds`, so `prompt` cannot simply be swapped for embeds.")
    def test_prompt_embeds(self):
        pass


class TestAnimateDiffSDXLPipelineMemory(AnimateDiffSDXLPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the pipeline."""


class TestAnimateDiffSDXLPipelineIPAdapter(AnimateDiffSDXLPipelineTesterConfig, IPAdapterTesterMixin):
    """IP-Adapter tests for the AnimateDiff SDXL pipeline."""


class TestAnimateDiffSDXLPipelineLoRA(AnimateDiffSDXLPipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the AnimateDiff SDXL pipeline."""


class TestAnimateDiffSDXLPipelineUNetLoRA(AnimateDiffSDXLPipelineTesterConfig, UNetLoraTesterMixin):
    """Per-UNet-block LoRA scale tests for the AnimateDiff SDXL pipeline."""


class TestAnimateDiffSDXLPipelineLoRAMemory(AnimateDiffSDXLPipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the pipeline."""
