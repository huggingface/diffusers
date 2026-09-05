import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AnimateDiffPAGPipeline,
    AnimateDiffPipeline,
    AutoencoderKL,
    DDIMScheduler,
    MotionAdapter,
    UNet2DConditionModel,
)

from ...testing_utils import torch_device
from ..animatediff.testing_utils import (
    FreeInitTesterMixin,
    FreeNoiseTesterMixin,
    MotionPipelineTesterConfig,
    MotionPipelineTesterMixin,
)
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..testing_utils import FromPipeTesterMixin, IPAdapterTesterMixin, MemoryTesterMixin
from .testing_utils import PAGPipelineTesterMixin


class AnimateDiffPAGPipelineTesterConfig(MotionPipelineTesterConfig):
    pipeline_class = AnimateDiffPAGPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS.union({"pag_scale", "pag_adaptive_scale"})
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
            "pag_scale": 3.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestAnimateDiffPAGPipeline(
    AnimateDiffPAGPipelineTesterConfig,
    MotionPipelineTesterMixin,
    PAGPipelineTesterMixin,
    FreeInitTesterMixin,
    FreeNoiseTesterMixin,
):
    base_pipeline_class = AnimateDiffPipeline
    # AnimateDiff's PAG layers resolve through the motion modules, so the "PAG enabled" leg keeps the pipeline
    # default layers and just leaves `pag_scale` at the dummy inputs' 3.0.
    pag_enabled_applied_layers = None

    @pytest.mark.skip(
        "`AnimateDiffPAGPipeline.check_inputs` rejects a dict `prompt`, so FreeNoise's per-frame prompts cannot be "
        "passed to it (the non-PAG `AnimateDiffPipeline` accepts them)."
    )
    def test_free_noise_multi_prompt(self):
        pass

    def test_dict_tuple_outputs_equivalent(self, expected_slice=None, expected_max_difference=1e-4):
        # Re-recorded: `get_dummy_components` now seeds the `MotionAdapter` like every other component, which it
        # did not before the migration, so the adapter's weights (and this slice) changed.
        if torch_device == "cpu" and expected_slice is None:
            # fmt: off
            expected_slice = torch.tensor([0.5132, 0.4380, 0.5327, 0.4619, 0.4955, 0.5457, 0.4980, 0.5015, 0.5652])
            # fmt: on
        super().test_dict_tuple_outputs_equivalent(
            expected_slice=expected_slice, expected_max_difference=expected_max_difference
        )

    def test_pag_applied_layers(self):
        pipe = self.get_pipeline()

        # pag_applied_layers = ["mid","up","down"] should apply to all self-attention layers
        # Note that for motion modules in AnimateDiff, both attn1 and attn2 are self-attention
        all_self_attn_layers = [
            k for k in pipe.unet.attn_processors.keys() if "attn1" in k or ("motion_modules" in k and "attn2" in k)
        ]
        original_attn_procs = pipe.unet.attn_processors
        pag_layers = [
            "down",
            "mid",
            "up",
        ]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_layers)

        # pag_applied_layers = ["mid"], or ["mid_block.0"] should apply to all self-attention layers in mid_block, i.e.
        # mid_block.motion_modules.0.transformer_blocks.0.attn1.processor
        # mid_block.attentions.0.transformer_blocks.0.attn1.processor
        all_self_attn_mid_layers = [
            "mid_block.attentions.0.transformer_blocks.0.attn1.processor",
            "mid_block.motion_modules.0.transformer_blocks.0.attn1.processor",
            "mid_block.motion_modules.0.transformer_blocks.0.attn2.processor",
        ]
        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["mid"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_mid_layers)

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["mid_block"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_mid_layers)

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["mid_block.(attentions|motion_modules)"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_mid_layers)

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["mid_block.attentions.1"]
        with pytest.raises(ValueError):
            pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)

        # pag_applied_layers = "down" should apply to all self-attention layers in down_blocks
        # down_blocks.1.(attentions|motion_modules).0.transformer_blocks.0.attn1.processor
        # down_blocks.1.(attentions|motion_modules).0.transformer_blocks.1.attn1.processor
        # down_blocks.1.(attentions|motion_modules).0.transformer_blocks.0.attn1.processor

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["down"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 10

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["down_blocks.0"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert (len(pipe.pag_attn_processors)) == 6

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 10

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["motion_modules.42"]
        with pytest.raises(ValueError):
            pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)


class TestAnimateDiffPAGPipelineMemory(AnimateDiffPAGPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the AnimateDiff PAG pipeline."""


class TestAnimateDiffPAGPipelineIPAdapter(AnimateDiffPAGPipelineTesterConfig, IPAdapterTesterMixin):
    """IP-Adapter tests for the AnimateDiff PAG pipeline."""


class TestAnimateDiffPAGPipelineFromPipe(AnimateDiffPAGPipelineTesterConfig, FromPipeTesterMixin):
    """`from_pipe` round-trip tests against `StableDiffusionPipeline`."""

    original_pipeline_repo = "hf-internal-testing/tinier-stable-diffusion-pipe"
