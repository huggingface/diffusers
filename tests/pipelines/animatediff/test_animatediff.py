import gc

import numpy as np
import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AnimateDiffPipeline,
    AutoencoderKL,
    DDIMScheduler,
    MotionAdapter,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from ...testing_utils import (
    backend_empty_cache,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)
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
    FreeNoiseSplitInferenceTesterMixin,
    MotionPipelineTesterConfig,
    MotionPipelineTesterMixin,
)


class AnimateDiffPipelineTesterConfig(MotionPipelineTesterConfig):
    pipeline_class = AnimateDiffPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    # `num_frames` defaults to 16; height/width default to `unet.sample_size * vae_scale_factor` (8 * 2).
    output_shape = (16, 3, 16, 16)

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

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 7.5,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestAnimateDiffPipeline(
    AnimateDiffPipelineTesterConfig,
    MotionPipelineTesterMixin,
    FreeInitTesterMixin,
    FreeNoiseSplitInferenceTesterMixin,
):
    def test_from_pipe_consistent_config(self):
        original_repo = "hf-internal-testing/tinier-stable-diffusion-pipe"

        # create StableDiffusionPipeline
        pipe_original = StableDiffusionPipeline.from_pretrained(original_repo, requires_safety_checker=False)

        # StableDiffusionPipeline -> AnimateDiffPipeline
        pipe_components = self.get_dummy_components()
        pipe_additional_components = {
            name: component for name, component in pipe_components.items() if name not in pipe_original.components
        }
        pipe = self.pipeline_class.from_pipe(pipe_original, **pipe_additional_components)

        # AnimateDiffPipeline -> StableDiffusionPipeline
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
            expected_slice = torch.tensor([0.5136, 0.4370, 0.5325, 0.4617, 0.4962, 0.5454, 0.4988, 0.5016, 0.5651])
            # fmt: on
        super().test_dict_tuple_outputs_equivalent(
            expected_slice=expected_slice, expected_max_difference=expected_max_difference
        )


class TestAnimateDiffPipelineMemory(AnimateDiffPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the AnimateDiff pipeline."""


class TestAnimateDiffPipelineIPAdapter(AnimateDiffPipelineTesterConfig, IPAdapterTesterMixin):
    """IP-Adapter tests for the AnimateDiff pipeline."""


class TestAnimateDiffPipelineLoRA(AnimateDiffPipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the AnimateDiff pipeline."""


class TestAnimateDiffPipelineUNetLoRA(AnimateDiffPipelineTesterConfig, UNetLoraTesterMixin):
    """Per-UNet-block LoRA scale tests for the AnimateDiff pipeline."""


class TestAnimateDiffPipelineLoRAMemory(AnimateDiffPipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the AnimateDiff pipeline."""


@slow
@require_torch_accelerator
class TestAnimateDiffPipelineIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_animatediff(self):
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
        pipe = AnimateDiffPipeline.from_pretrained("frankjoshua/toonyou_beta6", motion_adapter=adapter)
        pipe = pipe.to(torch_device)
        pipe.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            steps_offset=1,
            clip_sample=False,
        )
        pipe.vae.enable_slicing()
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        prompt = "night, b&w photo of old house, post apocalypse, forest, storm weather, wind, rocks, 8k uhd, dslr, soft lighting, high quality, film grain"
        negative_prompt = "bad quality, worse quality"

        generator = torch.Generator("cpu").manual_seed(0)
        output = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_frames=16,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=3,
            output_type="np",
        )

        image = output.frames[0]
        assert image.shape == (16, 512, 512, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array(
            [
                0.11357737,
                0.11285847,
                0.11180121,
                0.11084166,
                0.11414117,
                0.09785956,
                0.10742754,
                0.10510018,
                0.08045256,
            ]
        )
        assert numpy_cosine_similarity_distance(image_slice.flatten(), expected_slice.flatten()) < 1e-3


@pytest.mark.skip(FROM_PIPE_SKIP_REASON)
class TestAnimateDiffPipelineFromPipe(AnimateDiffPipelineTesterConfig, PipelineFromPipeTesterMixin):
    """`from_pipe` forward-pass parity and offload round trip for the AnimateDiff pipeline.

    Parked, not deleted: `test_from_pipe_consistent_config` runs for real as a method on the main test class above,
    but the forward-pass checks in `PipelineFromPipeTesterMixin` have no pytest-style equivalent yet.
    """
