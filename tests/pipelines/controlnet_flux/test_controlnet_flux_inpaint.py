import random

import torch
from transformers import AutoConfig, AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlNetInpaintPipeline,
    FluxControlNetModel,
    FluxTransformer2DModel,
)
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, floats_tensor, torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class FluxControlNetInpaintPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = FluxControlNetInpaintPipeline
    required_input_params_in_call_signature = frozenset(
        [
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "prompt_embeds",
            "pooled_prompt_embeds",
            "image",
            "mask_image",
            "control_image",
            "strength",
            "num_inference_steps",
            "controlnet_conditioning_scale",
        ]
    )
    batch_input_params = frozenset(["prompt", "image", "mask_image", "control_image"])
    output_shape = (3, 32, 32)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=8,
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
        # `eval()` because a directly constructed model stays in training mode, which leaves T5's
        # dropout active and makes the pipeline outputs non-deterministic across calls.
        text_encoder_2 = T5EncoderModel(config).eval()

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=2,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        torch.manual_seed(0)
        controlnet = FluxControlNetModel(
            patch_size=1,
            in_channels=8,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
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
            "controlnet": controlnet,
        }

    def get_dummy_inputs(self):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(0)).to(torch_device)
        mask_image = torch.ones((1, 1, 32, 32)).to(torch_device)
        control_image = floats_tensor((1, 3, 32, 32), rng=random.Random(0)).to(torch_device)

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "mask_image": mask_image,
            "control_image": control_image,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 48,
            "strength": 0.8,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestFluxControlNetInpaintPipeline(FluxControlNetInpaintPipelineTesterConfig, PipelineTesterMixin):
    def test_flux_controlnet_inpaint_with_num_images_per_prompt(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        inputs["num_images_per_prompt"] = 2
        images = pipe(**inputs).images

        assert images.shape == (2, *self.output_shape)

    def test_flux_controlnet_inpaint_with_controlnet_conditioning_scale(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        image_default = pipe(**inputs).images

        inputs["controlnet_conditioning_scale"] = 0.5
        image_scaled = pipe(**inputs).images

        # Ensure that changing the controlnet_conditioning_scale produces a different output
        assert not torch.allclose(image_default, image_scaled, atol=0.01), (
            "Changing `controlnet_conditioning_scale` should change the output."
        )

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=3e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_flux_image_output_shape(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()

        height_width_pairs = [(32, 32), (72, 56)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            inputs.update(
                {
                    "control_image": randn_tensor(
                        (1, 3, height, width),
                        device=torch_device,
                        dtype=torch.float16,
                    ),
                    "image": randn_tensor(
                        (1, 3, height, width),
                        device=torch_device,
                        dtype=torch.float16,
                    ),
                    "mask_image": torch.ones((1, 1, height, width)).to(torch_device),
                    "height": height,
                    "width": width,
                }
            )
            image = pipe(**inputs).images[0]
            _, output_height, output_width = image.shape
            assert (output_height, output_width) == (expected_height, expected_width), (
                f"Output shape {image.shape} does not match expected shape {(expected_height, expected_width)}"
            )


class TestFluxControlNetInpaintPipelineMemory(FluxControlNetInpaintPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Flux ControlNet inpaint pipeline."""
