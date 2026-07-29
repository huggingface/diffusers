import numpy as np
import torch
from transformers import AutoProcessor, Mistral3Config, Mistral3ForConditionalGeneration

from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2Pipeline,
    Flux2Transformer2DModel,
)

from ...testing_utils import torch_device
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
    check_qkv_fused_layers_exist,
)


class Flux2PipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = Flux2Pipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "prompt_embeds"]
    )
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
            timestep_guidance_channels=256,  # Hardcoded in original code
            axes_dims_rope=[4, 4, 4, 4],
        )

        config = Mistral3Config(
            text_config={
                "model_type": "mistral",
                "vocab_size": 32000,
                "hidden_size": 16,
                "intermediate_size": 37,
                "max_position_embeddings": 512,
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-05,
                "rope_theta": 1000000000.0,
                "sliding_window": None,
                "bos_token_id": 2,
                "eos_token_id": 3,
                "pad_token_id": 4,
            },
            vision_config={
                "model_type": "pixtral",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "intermediate_size": 37,
                "image_size": 30,
                "patch_size": 6,
                "num_channels": 3,
            },
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=4,
            model_dtype="mistral3",
            image_seq_length=4,
            vision_feature_layer=-1,
            image_token_index=1,
        )
        torch.manual_seed(0)
        text_encoder = Mistral3ForConditionalGeneration(config)
        tokenizer = AutoProcessor.from_pretrained(
            "hf-internal-testing/Mistral-Small-3.1-24B-Instruct-2503-only-processor"
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
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 8,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "text_encoder_out_layers": (1,),
        }
        return inputs


class TestFlux2Pipeline(Flux2PipelineTesterConfig, PipelineTesterMixin):
    def test_fused_qkv_projections(self):
        # Run on CPU so the torch tensor slices can be compared with `np.allclose`.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        original_image_slice = image[0, -1, -3:, -3:]

        # TODO (sayakpaul): will refactor this once `fuse_qkv_projections()` has been added
        # to the pipeline level.
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

        assert np.allclose(original_image_slice, image_slice_fused, atol=1e-3, rtol=1e-3), (
            "Fusion of QKV projections shouldn't affect the outputs."
        )
        assert np.allclose(image_slice_fused, image_slice_disabled, atol=1e-3, rtol=1e-3), (
            "Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled."
        )
        assert np.allclose(original_image_slice, image_slice_disabled, atol=1e-2, rtol=1e-2), (
            "Original outputs should match when fused QKV projections are disabled."
        )

    def test_flux_image_output_shape(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
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


class TestFlux2PipelineMemory(Flux2PipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Flux2 pipeline."""
