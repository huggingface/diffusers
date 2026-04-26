import inspect
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPAGPipeline,
    SD3Transformer2DModel,
)
from diffusers.utils.testing_utils import torch_device

from ..test_pipelines_common import (
    PipelineTesterMixin,
    check_qkv_fusion_matches_attn_procs_length,
    check_qkv_fusion_processors_exist,
)


class FluxPAGPipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = FluxPAGPipeline
    params = frozenset([
        "prompt",
        "height",
        "width",
        "guidance_scale",
        "negative_prompt",
        "prompt_embeds",
        "negative_prompt_embeds",
    ])
    batch_params = frozenset(["prompt", "negative_prompt"])
    test_xformers_attention = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = SD3Transformer2DModel(
            sample_size=32,
            patch_size=1,
            in_channels=4,
            num_layers=2,
            attention_head_dim=8,
            num_attention_heads=4,
            caption_projection_dim=32,
            joint_attention_dim=32,
            pooled_projection_dim=64,
            out_channels=4,
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
        text_encoder = CLIPTextModelWithProjection(clip_text_encoder_config)
        text_encoder_2 = CLIPTextModelWithProjection(clip_text_encoder_config)
        text_encoder_3 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_3 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

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
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "text_encoder_3": text_encoder_3,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "tokenizer_3": tokenizer_3,
            "transformer": transformer,
            "vae": vae,
        }

    def get_dummy_inputs(self, device, seed=0):
        generator = torch.manual_seed(seed) if str(device).startswith("mps") else torch.Generator(device="cpu").manual_seed(seed)
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "true_pag_scale": 0.0,
        }

    def test_different_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        image1 = pipe(**inputs).images[0]

        inputs["prompt_2"] = "A completely different scene"
        image2 = pipe(**inputs).images[0]

        assert np.abs(image1 - image2).max() > 1e-2

    def test_different_negative_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        image1 = pipe(**inputs).images[0]

        inputs["negative_prompt"] = "ugly, blurry"
        image2 = pipe(**inputs).images[0]

        assert np.abs(image1 - image2).max() > 1e-2

    def test_pag_disable_equivalent_to_baseline(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        image_pag_disabled = pipe(**inputs).images[0]

        del inputs["true_pag_scale"]
        image_baseline = pipe(**inputs).images[0]

        assert np.abs(image_pag_disabled - image_baseline).max() < 1e-3

    def test_fused_qkv_projections(self):
        device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(device)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice_original = image[0, -3:, -3:, -1]

        pipe.transformer.fuse_qkv_projections()
        assert check_qkv_fusion_processors_exist(pipe.transformer)
        assert check_qkv_fusion_matches_attn_procs_length(pipe.transformer, pipe.transformer.original_attn_processors)

        image_fused = pipe(**inputs).images
        image_slice_fused = image_fused[0, -3:, -3:, -1]

        pipe.transformer.unfuse_qkv_projections()
        image_disabled = pipe(**inputs).images
        image_slice_disabled = image_disabled[0, -3:, -3:, -1]

        assert np.allclose(image_slice_original, image_slice_fused, atol=1e-3)
        assert np.allclose(image_slice_fused, image_slice_disabled, atol=1e-3)

    def test_pag_applied_layers(self):
        device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(device)

        all_attn_keys = [k for k in pipe.transformer.attn_processors if "attn" in k]
        original_procs = pipe.transformer.attn_processors.copy()

        pag_layers = ["blocks.0", "blocks.1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_attn_keys)

        pipe.transformer.set_attn_processor(original_procs)
        pag_layers = ["blocks.0"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        expected_keys = [k for k in all_attn_keys if k.startswith("transformer_blocks.0")]
        assert set(pipe.pag_attn_processors) == set(expected_keys)

        pipe.transformer.set_attn_processor(original_procs)
        pag_layers = [r"blocks\.(0|1)"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 2
    
    def test_forward_signature_consistency(self):
        sig = inspect.signature(self.pipeline_class.__call__)
        expected = set(self.params)
        found = set(sig.parameters.keys())
        missing = expected - found
        extra = found - expected
        assert not missing, f"Missing parameters in pipeline: {missing}"
        assert not extra - {'self'}, f"Unexpected parameters in pipeline: {extra}"

    def test_attention_mask_support(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)
        inputs["attention_mask"] = torch.ones((1, 77))
        try:
            pipe(**inputs)
        except Exception as e:
            assert "attention_mask" not in str(e), f"Pipeline should support attention_mask, but failed: {e}"
