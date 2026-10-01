import pytest
import torch
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model

from diffusers.models.transformers.transformer_prx import PRXTransformer2DModel
from diffusers.pipelines.prx.pipeline_prx_pixel import PRXPixelPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from ...testing_utils import assert_tensors_close, torch_device
from ..pipeline_params import TEXT_TO_IMAGE_PARAMS
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


# Both PRX pipelines read `callback_on_step_end`'s inputs out of `locals()` but throw away what the callback
# returns, so a callback that rewrites `latents` (or `prompt_embeds`) has no effect on the denoising loop. Every
# other diffusers pipeline pops those keys back off the returned dict. Fixing it is a `src/` change and out of
# scope for this test migration, so the one shared test that exercises the write-back is marked `xfail`: whoever
# adds the pop-back will see it XPASS and can drop this marker.
CALLBACK_OUTPUTS_IGNORED = pytest.mark.xfail(
    reason="`PRX` pipelines discard the dict `callback_on_step_end` returns, so callback edits to `latents` are lost.",
    strict=True,
)


class PRXPixelPipelineTesterConfig(BasePipelineTesterConfig):
    # PRXPixelPipeline is standalone: it inherits from DiffusionPipeline (not PRXPipeline) and always has its own
    # image_processor, so it denoises raw RGB in pixel space and supports output_type="pil"/"np" without a VAE.
    pipeline_class = PRXPixelPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_input_params = frozenset(["prompt", "negative_prompt", "num_images_per_prompt"])
    output_shape = (3, 32, 32)

    def get_dummy_components(self):
        torch.manual_seed(0)
        # Pixel-space PRX: in_channels=3 (RGB), bottleneck img_in, resolution_embeds=True.
        # context_in_dim must match the text encoder hidden_size (16).
        transformer = PRXTransformer2DModel(
            patch_size=1,
            in_channels=3,
            context_in_dim=16,
            hidden_size=8,
            mlp_ratio=2.0,
            num_heads=2,
            depth=1,
            axes_dim=[2, 2],
            bottleneck_size=8,
            resolution_embeds=True,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler()

        # Tiny Qwen3 text encoder returning `last_hidden_state` (Qwen3-VL-style backbone).
        torch.manual_seed(0)
        config = Qwen3Config(
            hidden_size=16,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=151936,
            max_position_embeddings=512,
        )
        text_encoder = Qwen3Model(config)
        tokenizer = Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

        return {
            "transformer": transformer,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "prompt_max_tokens": 16,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "",
            "negative_prompt": "",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 32,
            "width": 32,
            # Pixel-space PRX has no VAE and returns raw (C, H, W) tensors for output_type="pt". The generic
            # PipelineTesterMixin tests compare these tensors directly, so default to "pt" here; the PIL/np default
            # path is exercised explicitly in test_inference_pil_and_np_output.
            "output_type": "pt",
            # 32px is not in the 1024 aspect-ratio bins, so binning must be disabled for these tiny fast tests.
            "use_resolution_binning": False,
        }


class TestPRXPixelPipeline(PRXPixelPipelineTesterConfig, PipelineTesterMixin):
    @CALLBACK_OUTPUTS_IGNORED
    def test_callback_inputs(self):
        super().test_callback_inputs()

    def test_pixel_space_has_no_vae(self):
        # Pixel space: vae_scale_factor is always 1, and the pipeline always carries an image processor
        # so postprocessing (and the default output_type="pil") works without any VAE.
        pipe = self.get_pipeline()

        assert pipe.vae_scale_factor == 1
        assert pipe.image_processor is not None

    def test_inference_batch(self):
        pipe = self.get_pipeline()

        image = self.run_pipe(pipe, prompt=["", ""], negative_prompt=["", ""])

        assert image.shape[0] == 2
        assert tuple(image.shape[1:]) == self.output_shape

    def test_inference_with_cfg(self):
        pipe = self.get_pipeline()

        # CFG off.
        out_no_cfg = self.run_pipe(pipe, guidance_scale=1.0)
        assert not pipe.do_classifier_free_guidance
        assert out_no_cfg[0].shape == self.output_shape

        # CFG on.
        out_cfg = self.run_pipe(pipe, guidance_scale=5.0)
        assert pipe.do_classifier_free_guidance
        assert out_cfg[0].shape == self.output_shape

        # Guidance should actually change the output.
        assert not torch.allclose(out_no_cfg, out_cfg)

    def test_inference_with_prompt_embeds(self):
        pipe = self.get_pipeline()

        # Precompute embeddings via the public encode_prompt API (CFG on so we get negatives too).
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
            pipe.encode_prompt(
                prompt="a prompt",
                device=torch.device("cpu"),
                do_classifier_free_guidance=True,
                negative_prompt="",
            )
        )

        inputs = self.get_dummy_inputs()
        inputs.pop("prompt")
        inputs.pop("negative_prompt")
        inputs["guidance_scale"] = 5.0
        inputs["prompt_embeds"] = prompt_embeds
        inputs["negative_prompt_embeds"] = negative_prompt_embeds
        inputs["prompt_attention_mask"] = prompt_attention_mask
        inputs["negative_prompt_attention_mask"] = negative_prompt_attention_mask

        image = pipe(**inputs)[0]
        assert image[0].shape == self.output_shape

    def test_inference_pil_and_np_output(self):
        # The default output_type="pil" must work without a VAE: the denoised pixels are denormalized
        # directly by the image processor instead of being decoded.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs.pop("output_type")  # default is "pil"
        images = pipe(**inputs).images
        assert len(images) == 1
        assert images[0].size == (32, 32)

        inputs = self.get_dummy_inputs()
        inputs["output_type"] = "np"
        images = pipe(**inputs).images
        assert images.shape == (1, 32, 32, 3)
        assert images.min() >= 0.0
        assert images.max() <= 1.0

    def test_non_multiple_size_raises(self):
        # height/width must be divisible by vae_scale_factor * transformer patch_size; check_inputs must raise
        # a clear ValueError instead of letting the transformer fail on an invalid reshape mid-denoising.
        components = self.get_dummy_components()
        torch.manual_seed(0)
        components["transformer"] = PRXTransformer2DModel(
            patch_size=2,
            in_channels=3,
            context_in_dim=16,
            hidden_size=8,
            mlp_ratio=2.0,
            num_heads=2,
            depth=1,
            axes_dim=[2, 2],
            bottleneck_size=8,
            resolution_embeds=True,
        )
        pipe = self.get_pipeline(**components)

        inputs = self.get_dummy_inputs()
        inputs["height"] = 31  # vae_scale_factor (1) * patch_size (2) = 2; 31 is not a multiple
        with pytest.raises(ValueError, match="divisible"):
            pipe(**inputs)

    def test_attention_slicing_forward_pass(self, expected_max_diff=1e-3):
        # Run on CPU: sliced attention is compared against a full-attention run of the same pipeline.
        pipe = self.get_pipeline()

        output_without_slicing = self.run_pipe(pipe)

        pipe.enable_attention_slicing(slice_size=1)
        output_with_slicing_1 = self.run_pipe(pipe)

        pipe.enable_attention_slicing(slice_size=2)
        output_with_slicing_2 = self.run_pipe(pipe)

        assert_tensors_close(
            output_with_slicing_1,
            output_without_slicing,
            atol=expected_max_diff,
            msg="Attention slicing (slice_size=1) changed the output.",
        )
        assert_tensors_close(
            output_with_slicing_2,
            output_without_slicing,
            atol=expected_max_diff,
            msg="Attention slicing (slice_size=2) changed the output.",
        )

    def test_encode_prompt_works_in_isolation(self):
        extra_required_param_value_dict = {
            "device": torch.device(torch_device).type,
            "do_classifier_free_guidance": self.get_dummy_inputs().get("guidance_scale", 1.0) > 1.0,
        }
        return super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict)


class TestPRXPixelPipelineMemory(PRXPixelPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the PRXPixel pipeline."""
