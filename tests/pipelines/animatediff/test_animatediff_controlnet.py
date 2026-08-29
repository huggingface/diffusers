import pytest
import torch
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AnimateDiffControlNetPipeline,
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    MotionAdapter,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from ...testing_utils import torch_device
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineFromPipeTesterMixin
from ..testing_utils import (
    IPAdapterTesterMixin,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    UNetLoraTesterMixin,
)
from .testing_utils import (
    FROM_PIPE_SKIP_REASON,
    FreeInitTesterMixin,
    FreeNoiseTesterMixin,
    MotionPipelineTesterConfig,
    MotionPipelineTesterMixin,
)


class AnimateDiffControlNetPipelineTesterConfig(MotionPipelineTesterConfig):
    pipeline_class = AnimateDiffControlNetPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS.union({"conditioning_frames"})
    # `get_dummy_inputs` asks for 2 frames; height/width default to `unet.sample_size * vae_scale_factor` (8 * 2).
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
        controlnet = ControlNetModel(
            block_out_channels=block_out_channels,
            layers_per_block=2,
            in_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            cross_attention_dim=cross_attention_dim,
            conditioning_embedding_out_channels=(8, 8),
            norm_num_groups=1,
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
        motion_adapter = MotionAdapter(
            block_out_channels=block_out_channels,
            motion_layers_per_block=2,
            motion_norm_num_groups=2,
            motion_num_attention_heads=4,
        )

        return {
            "unet": unet,
            "controlnet": controlnet,
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
        conditioning_frames = [Image.new("RGB", (video_width, video_height))] * num_frames

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "conditioning_frames": conditioning_frames,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "num_frames": num_frames,
            "guidance_scale": 7.5,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestAnimateDiffControlNetPipeline(
    AnimateDiffControlNetPipelineTesterConfig,
    MotionPipelineTesterMixin,
    FreeInitTesterMixin,
    FreeNoiseTesterMixin,
):
    def get_free_noise_inputs(self):
        # One conditioning frame is needed per generated frame, so the longer run is requested through
        # `get_dummy_inputs` rather than by overriding `num_frames` on the returned dict.
        return self.get_dummy_inputs(num_frames=16)

    def test_from_pipe_consistent_config(self):
        original_repo = "hf-internal-testing/tinier-stable-diffusion-pipe"

        # create StableDiffusionPipeline
        pipe_original = StableDiffusionPipeline.from_pretrained(original_repo, requires_safety_checker=False)

        # StableDiffusionPipeline -> AnimateDiffControlNetPipeline
        pipe_components = self.get_dummy_components()
        pipe_additional_components = {
            name: component for name, component in pipe_components.items() if name not in pipe_original.components
        }
        pipe = self.pipeline_class.from_pipe(pipe_original, **pipe_additional_components)

        # AnimateDiffControlNetPipeline -> StableDiffusionPipeline
        original_pipe_additional_components = {}
        for name, component in pipe_original.components.items():
            if name not in pipe.components or not isinstance(component, pipe.components[name].__class__):
                original_pipe_additional_components[name] = component

        pipe_original_2 = StableDiffusionPipeline.from_pipe(pipe, **original_pipe_additional_components)

        # compare the config
        original_config = {k: v for k, v in pipe_original.config.items() if not k.startswith("_")}
        original_config_2 = {k: v for k, v in pipe_original_2.config.items() if not k.startswith("_")}
        assert original_config_2 == original_config

    def test_dict_tuple_outputs_equivalent(self, expected_slice=None, expected_max_difference=1e-4):
        if torch_device == "cpu" and expected_slice is None:
            # fmt: off
            expected_slice = torch.tensor([0.5885, 0.5630, 0.5083, 0.5943, 0.4598, 0.5104])
            # fmt: on
        super().test_dict_tuple_outputs_equivalent(
            expected_slice=expected_slice, expected_max_difference=expected_max_difference
        )


class TestAnimateDiffControlNetPipelineMemory(AnimateDiffControlNetPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the pipeline."""


class TestAnimateDiffControlNetPipelineIPAdapter(AnimateDiffControlNetPipelineTesterConfig, IPAdapterTesterMixin):
    """IP-Adapter tests for the AnimateDiff ControlNet pipeline."""


class TestAnimateDiffControlNetPipelineLoRA(AnimateDiffControlNetPipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the AnimateDiff ControlNet pipeline."""


class TestAnimateDiffControlNetPipelineUNetLoRA(AnimateDiffControlNetPipelineTesterConfig, UNetLoraTesterMixin):
    """Per-UNet-block LoRA scale tests for the AnimateDiff ControlNet pipeline."""


class TestAnimateDiffControlNetPipelineLoRAMemory(AnimateDiffControlNetPipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the pipeline."""


@pytest.mark.skip(FROM_PIPE_SKIP_REASON)
class TestAnimateDiffControlNetPipelineFromPipe(
    AnimateDiffControlNetPipelineTesterConfig, PipelineFromPipeTesterMixin
):
    """`from_pipe` forward-pass parity and offload round trip for the AnimateDiff ControlNet pipeline.

    Parked, not deleted: `test_from_pipe_consistent_config` runs for real as a method on the main test class above,
    but the forward-pass checks in `PipelineFromPipeTesterMixin` have no pytest-style equivalent yet.
    """
