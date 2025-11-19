import unittest

import numpy as np
import torch
from transformers import AutoProcessor, Mistral3Config, Mistral3ForConditionalGeneration

from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2Pipeline,
    Flux2Transformer2DModel,
)

from ...testing_utils import (
    torch_device,
)
from ..test_pipelines_common import (
    PipelineTesterMixin,
    check_qkv_fused_layers_exist,
)


class Flux2PipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = Flux2Pipeline
    params = frozenset(["prompt", "height", "width", "guidance_scale", "prompt_embeds"])
    batch_params = frozenset(["prompt"])

    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    supports_dduf = False

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

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        inputs = {
            "prompt": "a dog is dancing",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 8,
            "output_type": "np",
            "text_encoder_out_layers": (1,),
        }
        return inputs

    def test_fused_qkv_projections(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        original_image_slice = image[0, -3:, -3:, -1]

        # TODO (sayakpaul): will refactor this once `fuse_qkv_projections()` has been added
        # to the pipeline level.
        pipe.transformer.fuse_qkv_projections()
        self.assertTrue(
            check_qkv_fused_layers_exist(pipe.transformer, ["to_qkv"]),
            ("Something wrong with the fused attention layers. Expected all the attention projections to be fused."),
        )

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice_fused = image[0, -3:, -3:, -1]

        pipe.transformer.unfuse_qkv_projections()
        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice_disabled = image[0, -3:, -3:, -1]

        self.assertTrue(
            np.allclose(original_image_slice, image_slice_fused, atol=1e-3, rtol=1e-3),
            ("Fusion of QKV projections shouldn't affect the outputs."),
        )
        self.assertTrue(
            np.allclose(image_slice_fused, image_slice_disabled, atol=1e-3, rtol=1e-3),
            ("Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled."),
        )
        self.assertTrue(
            np.allclose(original_image_slice, image_slice_disabled, atol=1e-2, rtol=1e-2),
            ("Original outputs should match when fused QKV projections are disabled."),
        )

    def test_flux_image_output_shape(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        height_width_pairs = [(32, 32), (72, 57)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            inputs.update({"height": height, "width": width})
            image = pipe(**inputs).images[0]
            output_height, output_width, _ = image.shape
            self.assertEqual(
                (output_height, output_width),
                (expected_height, expected_width),
                f"Output shape {image.shape} does not match expected shape {(expected_height, expected_width)}",
            )
