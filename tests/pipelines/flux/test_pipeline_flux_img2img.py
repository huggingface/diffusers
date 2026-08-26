import random

import torch
from transformers import AutoConfig, AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxImg2ImgPipeline, FluxTransformer2DModel

from ...testing_utils import floats_tensor, torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin
from .ip_adapter_tester import FluxIPAdapterTesterMixin


class FluxImg2ImgPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = FluxImg2ImgPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "prompt_embeds", "pooled_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    output_shape = (3, 8, 8)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=4,
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
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder_2 = T5EncoderModel(config)

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
            "image_encoder": None,
            "feature_extractor": None,
        }

    def get_dummy_inputs(self):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(0)).to(torch_device)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 48,
            "strength": 0.8,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }
        return inputs


class TestFluxImg2ImgPipeline(FluxImg2ImgPipelineTesterConfig, PipelineTesterMixin):
    def test_flux_different_prompts(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs()
        inputs["prompt_2"] = "a different prompt"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = (output_same_prompt - output_different_prompts).abs().max()

        # Outputs should be different here
        # For some reasons, they don't show large differences
        assert max_diff > 1e-6, "Outputs should be different for different prompts."

    def test_flux_true_cfg_with_negative_embeds(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs.pop("generator")
        prompt = inputs.pop("prompt")

        prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt, prompt_2=None, device=torch_device, num_images_per_prompt=1, max_sequence_length=48
        )
        negative_prompt_embeds, negative_pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt="bad quality", prompt_2=None, device=torch_device, num_images_per_prompt=1, max_sequence_length=48
        )
        inputs.update(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )

        inputs["true_cfg_scale"] = 1.0
        cfg_off = pipe(**inputs, generator=torch.manual_seed(0)).images[0]
        inputs["true_cfg_scale"] = 2.0
        cfg_on = pipe(**inputs, generator=torch.manual_seed(0)).images[0]
        assert not torch.allclose(cfg_off, cfg_on), (
            "Precomputed negative embeds should enable true CFG when negative_prompt is None."
        )

    def test_flux_image_output_shape(self):
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


class TestFluxImg2ImgPipelineIPAdapter(FluxImg2ImgPipelineTesterConfig, FluxIPAdapterTesterMixin):
    """IP-Adapter tests for the Flux Img2Img pipeline."""


class TestFluxImg2ImgPipelineMemory(FluxImg2ImgPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Flux Img2Img pipeline."""
