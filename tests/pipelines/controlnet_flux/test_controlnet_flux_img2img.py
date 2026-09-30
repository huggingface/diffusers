import torch
from transformers import AutoConfig, AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlNetImg2ImgPipeline,
    FluxControlNetModel,
    FluxTransformer2DModel,
)
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import assert_tensors_close, torch_device
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
    check_qkv_fused_layers_exist,
)


class FluxControlNetImg2ImgPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = FluxControlNetImg2ImgPipeline
    required_input_params_in_call_signature = frozenset(
        [
            "prompt",
            "image",
            "control_image",
            "height",
            "width",
            "strength",
            "guidance_scale",
            "controlnet_conditioning_scale",
            "prompt_embeds",
            "pooled_prompt_embeds",
        ]
    )
    batch_input_params = frozenset(["prompt", "image", "control_image"])
    output_shape = (3, 32, 32)

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
            latent_channels=1,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        torch.manual_seed(0)
        controlnet = FluxControlNetModel(
            in_channels=4,
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
        # Seeded so that repeated `get_dummy_inputs()` calls hand the pipeline the same images — the shared tests
        # compare two runs against each other.
        image = randn_tensor((1, 3, 32, 32), generator=self.get_generator(0), device=torch.device(torch_device))
        control_image = randn_tensor(
            (1, 3, 32, 32), generator=self.get_generator(1), device=torch.device(torch_device)
        )

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "control_image": control_image,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "controlnet_conditioning_scale": 1.0,
            "strength": 0.8,
            "height": 32,
            "width": 32,
            "max_sequence_length": 48,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestFluxControlNetImg2ImgPipeline(FluxControlNetImg2ImgPipelineTesterConfig, PipelineTesterMixin):
    def test_flux_controlnet_different_prompts(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs()
        inputs["prompt_2"] = "a different prompt"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = (output_same_prompt - output_different_prompts).abs().max()

        assert max_diff > 1e-6, "Outputs should be different for different prompts."

    def test_fused_qkv_projections(self):
        # Run on CPU to keep the seeded generator deterministic across the three forward passes.
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
                    "height": height,
                    "width": width,
                }
            )
            image = pipe(**inputs).images[0]
            _, output_height, output_width = image.shape
            assert (output_height, output_width) == (expected_height, expected_width), (
                f"Output shape {image.shape} does not match expected shape {(expected_height, expected_width)}"
            )


class TestFluxControlNetImg2ImgPipelineMemory(FluxControlNetImg2ImgPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Flux ControlNet img2img pipeline."""
