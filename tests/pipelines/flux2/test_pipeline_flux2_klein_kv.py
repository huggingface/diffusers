import pytest
import torch
from PIL import Image
from transformers import Qwen2TokenizerFast, Qwen3Config, Qwen3ForCausalLM

from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinKVPipeline,
    Flux2Transformer2DModel,
)

from ...testing_utils import assert_tensors_close, torch_device
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
    check_qkv_fused_layers_exist,
)


class Flux2KleinKVPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = Flux2KleinKVPipeline
    required_input_params_in_call_signature = frozenset(["prompt", "height", "width", "prompt_embeds", "image"])
    batch_input_params = frozenset(["prompt"])

    def get_dummy_components(self, num_layers: int = 1, num_single_layers: int = 1):
        torch.manual_seed(0)
        transformer = Flux2Transformer2DModel(
            patch_size=1,
            in_channels=4,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=16,
            timestep_guidance_channels=256,
            axes_dims_rope=[4, 4, 4, 4],
            guidance_embeds=False,
        )

        # Create minimal Qwen3 config
        config = Qwen3Config(
            intermediate_size=16,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=151936,
            max_position_embeddings=512,
        )
        torch.manual_seed(0)
        text_encoder = Qwen3ForCausalLM(config)

        # Use a simple tokenizer for testing
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"
        )

        torch.manual_seed(0)
        vae = AutoencoderKLFlux2(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=1,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
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
        inputs = {
            "prompt": "a dog is dancing",
            "image": Image.new("RGB", (64, 64)),
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "height": 8,
            "width": 8,
            "max_sequence_length": 64,
            "text_encoder_out_layers": (1,),
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }
        return inputs


class TestFlux2KleinKVPipeline(Flux2KleinKVPipelineTesterConfig, PipelineTesterMixin):
    def test_fused_qkv_projections(self):
        # Run on CPU to keep the slice comparisons deterministic.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        original_image_slice = image[0, -1, -3:, -3:]

        pipe.transformer.fuse_qkv_projections()
        assert check_qkv_fused_layers_exist(pipe.transformer, ["to_qkv"]), (
            "Something wrong with the fused attention layers. Expected all the attention projections to be fused."
        )

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice_fused = image[0, -1, -3:, -3:]

        pipe.transformer.unfuse_qkv_projections()
        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice_disabled = image[0, -1, -3:, -3:]

        assert_tensors_close(
            original_image_slice,
            image_slice_fused,
            atol=1e-3,
            rtol=1e-3,
            msg="Fusion of QKV projections shouldn't affect the outputs.",
        )
        assert_tensors_close(
            image_slice_fused,
            image_slice_disabled,
            atol=1e-3,
            rtol=1e-3,
            msg="Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled.",
        )
        assert_tensors_close(
            original_image_slice,
            image_slice_disabled,
            atol=1e-2,
            rtol=1e-2,
            msg="Original outputs should match when fused QKV projections are disabled.",
        )

    def test_image_output_shape(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()

        height_width_pairs = [(32, 32), (72, 57)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            inputs.update({"height": height, "width": width})
            image = pipe(**inputs).images[0]
            _, output_height, output_width = image.shape
            assert (output_height, output_width) == (expected_height, expected_width), (
                f"Output shape {image.shape} does not match expected shape {(expected_height, expected_width)}"
            )

    def test_without_image(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        del inputs["image"]
        image = pipe(**inputs).images
        assert image.shape == (1, 3, 8, 8)

    @pytest.mark.skip("Needs to be revisited")
    def test_encode_prompt_works_in_isolation(self):
        pass


class TestFlux2KleinKVPipelineMemory(Flux2KleinKVPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Flux2 Klein KV pipeline."""
