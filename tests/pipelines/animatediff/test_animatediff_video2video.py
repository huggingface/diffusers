import torch
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AnimateDiffVideoToVideoPipeline,
    AutoencoderKL,
    DDIMScheduler,
    MotionAdapter,
    UNet2DConditionModel,
)

from ...testing_utils import torch_device
from ..pipeline_params import TEXT_TO_IMAGE_PARAMS, VIDEO_TO_VIDEO_BATCH_PARAMS
from ..testing_utils import (
    FromPipeTesterMixin,
    IPAdapterTesterMixin,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    UNetLoraTesterMixin,
)
from .testing_utils import (
    FreeInitTesterMixin,
    FreeNoiseSplitInferenceTesterMixin,
    MotionPipelineTesterConfig,
    MotionPipelineTesterMixin,
)


class AnimateDiffVideoToVideoPipelineTesterConfig(MotionPipelineTesterConfig):
    pipeline_class = AnimateDiffVideoToVideoPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS
    batch_input_params = VIDEO_TO_VIDEO_BATCH_PARAMS
    # The frame count comes from the conditioning video (2 frames below); height/width default to
    # `unet.sample_size * vae_scale_factor` (8 * 2).
    output_shape = (2, 3, 16, 16)

    def get_dummy_components(self):
        cross_attention_dim = 8
        block_out_channels = (8, 8)

        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=block_out_channels,
            layers_per_block=2,
            sample_size=8,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=cross_attention_dim,
            norm_num_groups=2,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            clip_sample=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=block_out_channels,
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=2,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=cross_attention_dim,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        torch.manual_seed(0)
        motion_adapter = MotionAdapter(
            block_out_channels=block_out_channels,
            motion_layers_per_block=2,
            motion_norm_num_groups=2,
            motion_num_attention_heads=4,
        )

        return {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "motion_adapter": motion_adapter,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "feature_extractor": None,
            "image_encoder": None,
        }

    def get_dummy_inputs(self, num_frames: int = 2):
        video_height = 32
        video_width = 32
        video = [Image.new("RGB", (video_width, video_height))] * num_frames

        return {
            "video": video,
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 7.5,
            "output_type": "pt",
        }


class TestAnimateDiffVideoToVideoPipeline(
    AnimateDiffVideoToVideoPipelineTesterConfig,
    MotionPipelineTesterMixin,
    FreeInitTesterMixin,
    FreeNoiseSplitInferenceTesterMixin,
):
    def get_free_noise_inputs(self):
        # The frame count is derived from the conditioning video, so the longer run is requested by building a
        # longer video rather than by passing `num_frames`.
        inputs = self.get_dummy_inputs(num_frames=16)
        inputs["strength"] = 0.5
        return inputs

    def test_latent_inputs(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        sample_size = pipe.unet.config.sample_size
        inputs["latents"] = torch.randn((1, 4, 1, sample_size, sample_size), device=torch_device)
        inputs.pop("video")
        pipe(**inputs)


class TestAnimateDiffVideoToVideoPipelineMemory(AnimateDiffVideoToVideoPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the pipeline."""


class TestAnimateDiffVideoToVideoPipelineIPAdapter(AnimateDiffVideoToVideoPipelineTesterConfig, IPAdapterTesterMixin):
    """IP-Adapter tests for the AnimateDiff video-to-video pipeline."""


class TestAnimateDiffVideoToVideoPipelineLoRA(AnimateDiffVideoToVideoPipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the AnimateDiff video-to-video pipeline."""


class TestAnimateDiffVideoToVideoPipelineUNetLoRA(AnimateDiffVideoToVideoPipelineTesterConfig, UNetLoraTesterMixin):
    """Per-UNet-block LoRA scale tests for the AnimateDiff video-to-video pipeline."""


class TestAnimateDiffVideoToVideoPipelineLoRAMemory(
    AnimateDiffVideoToVideoPipelineTesterConfig, LoraMemoryTesterMixin
):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the pipeline."""


class TestAnimateDiffVideoToVideoPipelineFromPipe(AnimateDiffVideoToVideoPipelineTesterConfig, FromPipeTesterMixin):
    """`from_pipe` round-trip tests against `StableDiffusionPipeline` for the AnimateDiff video-to-video pipeline."""

    original_pipeline_repo = "hf-internal-testing/tinier-stable-diffusion-pipe"
