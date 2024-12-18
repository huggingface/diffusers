import unittest

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlInpaintPipeline,
    FluxTransformer2DModel,
)
from diffusers.utils.testing_utils import (
    torch_device,
)

from ..test_pipelines_common import (
    PipelineTesterMixin,
    check_qkv_fusion_matches_attn_procs_length,
    check_qkv_fusion_processors_exist,
)


class FluxControlInpaintPipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = FluxControlInpaintPipeline
    params = frozenset(["prompt", "height", "width", "guidance_scale", "prompt_embeds", "pooled_prompt_embeds"])
    batch_params = frozenset(["prompt"])

    # there is no xformers processor for Flux
    test_xformers_attention = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=8,
            out_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
        )
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )

        torch.manual_seed(0)
        text_encoder = CLIPTextModel(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_2 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=1,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "transformer": transformer,
            "vae": vae,
        }

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        image = Image.new("RGB", (8, 8), 0)
        control_image = Image.new("RGB", (8, 8), 0)
        mask_image = Image.new("RGB", (8, 8), 255)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "control_image": control_image,
            "generator": generator,
            "image": image,
            "mask_image": mask_image,
            "strength": 0.8,
            "num_inference_steps": 2,
            "guidance_scale": 30.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 48,
            "output_type": "np",
        }
        return inputs

    # def test_flux_different_prompts(self):
    #     pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)

    #     inputs = self.get_dummy_inputs(torch_device)
    #     output_same_prompt = pipe(**inputs).images[0]

    #     inputs = self.get_dummy_inputs(torch_device)
    #     inputs["prompt_2"] = "a different prompt"
    #     output_different_prompts = pipe(**inputs).images[0]

    #     max_diff = np.abs(output_same_prompt - output_different_prompts).max()

    #     # Outputs should be different here
    #     # For some reasons, they don't show large differences
    #     assert max_diff > 1e-6

    def test_flux_prompt_embeds(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        output_with_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = inputs.pop("prompt")

        (prompt_embeds, pooled_prompt_embeds, text_ids) = pipe.encode_prompt(
            prompt,
            prompt_2=None,
            device=torch_device,
            max_sequence_length=inputs["max_sequence_length"],
        )
        output_with_embeds = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            **inputs,
        ).images[0]

        max_diff = np.abs(output_with_prompt - output_with_embeds).max()
        assert max_diff < 1e-4

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
        assert check_qkv_fusion_processors_exist(
            pipe.transformer
        ), "Something wrong with the fused attention processors. Expected all the attention processors to be fused."
        assert check_qkv_fusion_matches_attn_procs_length(
            pipe.transformer, pipe.transformer.original_attn_processors
        ), "Something wrong with the attention processors concerning the fused QKV projections."

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice_fused = image[0, -3:, -3:, -1]

        pipe.transformer.unfuse_qkv_projections()
        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice_disabled = image[0, -3:, -3:, -1]

        assert np.allclose(
            original_image_slice, image_slice_fused, atol=1e-3, rtol=1e-3
        ), "Fusion of QKV projections shouldn't affect the outputs."
        assert np.allclose(
            image_slice_fused, image_slice_disabled, atol=1e-3, rtol=1e-3
        ), "Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled."
        assert np.allclose(
            original_image_slice, image_slice_disabled, atol=1e-2, rtol=1e-2
        ), "Original outputs should match when fused QKV projections are disabled."

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
            assert (output_height, output_width) == (expected_height, expected_width)
