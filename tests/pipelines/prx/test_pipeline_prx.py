import pytest
import torch
from transformers import AutoTokenizer
from transformers.models.t5gemma.configuration_t5gemma import T5GemmaConfig, T5GemmaModuleConfig
from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder

from diffusers.models import AutoencoderDC, AutoencoderKL
from diffusers.models.transformers.transformer_prx import PRXTransformer2DModel
from diffusers.pipelines.prx.pipeline_prx import PRXPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from ...testing_utils import assert_tensors_close
from ..pipeline_params import TEXT_TO_IMAGE_PARAMS
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


# `T5GemmaEncoder` is instantiated here from a hand-built config rather than loaded from a repo, and transformers v5
# cannot round-trip that through `save_pretrained`/`from_pretrained`, so every test that reloads the pipeline from
# disk is skipped.
T5GEMMA_SERIALIZATION_SKIP_REASON = "Custom T5GemmaEncoder not compatible with transformers v5."

# Both PRX pipelines read `callback_on_step_end`'s inputs out of `locals()` but throw away what the callback
# returns, so a callback that rewrites `latents` (or `prompt_embeds`) has no effect on the denoising loop. Every
# other diffusers pipeline pops those keys back off the returned dict. Fixing it is a `src/` change and out of
# scope for this test migration, so the one shared test that exercises the write-back is marked `xfail`: whoever
# adds the pop-back will see it XPASS and can drop this marker.
CALLBACK_OUTPUTS_IGNORED = pytest.mark.xfail(
    reason="`PRX` pipelines discard the dict `callback_on_step_end` returns, so callback edits to `latents` are lost.",
    strict=True,
)


class PRXPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = PRXPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_input_params = frozenset(["prompt", "negative_prompt", "num_images_per_prompt"])
    output_shape = (3, 32, 32)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = PRXTransformer2DModel(
            patch_size=1,
            in_channels=4,
            context_in_dim=8,
            hidden_size=8,
            mlp_ratio=2.0,
            num_heads=2,
            depth=1,
            axes_dim=[2, 2],
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
            shift_factor=0.0,
            scaling_factor=1.0,
        ).eval()

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler()

        torch.manual_seed(0)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/dummy-gemma")
        tokenizer.model_max_length = 64

        torch.manual_seed(0)

        encoder_params = {
            "vocab_size": tokenizer.vocab_size,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 64,
            "layer_types": ["full_attention"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "dropout_rate": 0.0,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-06,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "query_pre_attn_scalar": 4,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
        }
        encoder_config = T5GemmaModuleConfig(**encoder_params)
        text_encoder_config = T5GemmaConfig(encoder=encoder_config, is_encoder_decoder=False, **encoder_params)
        text_encoder = T5GemmaEncoder(text_encoder_config.encoder)

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
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
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "use_resolution_binning": False,
        }


class TestPRXPipeline(PRXPipelineTesterConfig, PipelineTesterMixin):
    @CALLBACK_OUTPUTS_IGNORED
    def test_callback_inputs(self):
        super().test_callback_inputs()

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

    def test_inference_with_autoencoder_dc(self):
        """PRXPipeline should also work with an `AutoencoderDC` (DCAE) in place of the `AutoencoderKL`."""
        components = self.get_dummy_components()

        torch.manual_seed(0)
        vae_dc = AutoencoderDC(
            in_channels=3,
            latent_channels=4,
            attention_head_dim=2,
            encoder_block_types=(
                "ResBlock",
                "EfficientViTBlock",
            ),
            decoder_block_types=(
                "ResBlock",
                "EfficientViTBlock",
            ),
            encoder_block_out_channels=(8, 8),
            decoder_block_out_channels=(8, 8),
            encoder_qkv_multiscales=((), (5,)),
            decoder_qkv_multiscales=((), (5,)),
            encoder_layers_per_block=(1, 1),
            decoder_layers_per_block=(1, 1),
            upsample_block_type="interpolate",
            downsample_block_type="stride_conv",
            decoder_norm_types="rms_norm",
            decoder_act_fns="silu",
        ).eval()

        components["vae"] = vae_dc
        pipe = self.get_pipeline(**components)

        assert pipe.vae_scale_factor == vae_dc.spatial_compression_ratio

        output = self.run_pipe(pipe)
        assert output[0].shape == self.output_shape
        assert torch.isfinite(output).all()

    @pytest.mark.skip(T5GEMMA_SERIALIZATION_SKIP_REASON)
    def test_loading_with_variants(self):
        pass

    @pytest.mark.skip(T5GEMMA_SERIALIZATION_SKIP_REASON)
    def test_save_load_local(self):
        pass

    @pytest.mark.skip(T5GEMMA_SERIALIZATION_SKIP_REASON)
    def test_save_load_float16(self):
        pass

    @pytest.mark.skip(T5GEMMA_SERIALIZATION_SKIP_REASON)
    def test_save_load_optional_components(self):
        pass

    @pytest.mark.skip(T5GEMMA_SERIALIZATION_SKIP_REASON)
    def test_torch_dtype_dict(self):
        pass


class TestPRXPipelineMemory(PRXPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the PRX pipeline."""

    @pytest.mark.skip(T5GEMMA_SERIALIZATION_SKIP_REASON)
    def test_pipeline_with_accelerator_device_map(self):
        pass
