import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlInpaintPipeline,
    FluxTransformer2DModel,
)

from ...testing_utils import assert_tensors_close, torch_device
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
    check_qkv_fused_layers_exist,
)


class FluxControlInpaintPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = FluxControlInpaintPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "prompt_embeds", "pooled_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    output_shape = (3, 8, 8)

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
        }

    def get_dummy_inputs(self):
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "control_image": Image.new("RGB", (8, 8), 0),
            "generator": self.get_generator(0),
            "image": Image.new("RGB", (8, 8), 0),
            "mask_image": Image.new("RGB", (8, 8), 255),
            "strength": 0.8,
            "num_inference_steps": 2,
            "guidance_scale": 30.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 48,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }
        return inputs


class TestFluxControlInpaintPipeline(FluxControlInpaintPipelineTesterConfig, PipelineTesterMixin):
    def test_fused_qkv_projections(self):
        pipe = self.get_pipeline().to(torch_device)

        image_slice = self.run_pipe(pipe)[0, -1, -3:, -3:]

        # TODO (sayakpaul): will refactor this once `fuse_qkv_projections()` has been added
        # to the pipeline level.
        pipe.transformer.fuse_qkv_projections()
        assert check_qkv_fused_layers_exist(pipe.transformer, ["to_qkv"]), (
            "Something wrong with the fused attention layers. Expected all the attention projections to be fused."
        )

        image_slice_fused = self.run_pipe(pipe)[0, -1, -3:, -3:]

        pipe.transformer.unfuse_qkv_projections()
        image_slice_disabled = self.run_pipe(pipe)[0, -1, -3:, -3:]

        assert_tensors_close(
            image_slice_fused,
            image_slice,
            atol=1e-3,
            rtol=1e-3,
            msg="Fusion of QKV projections shouldn't affect the outputs.",
        )
        assert_tensors_close(
            image_slice_disabled,
            image_slice_fused,
            atol=1e-3,
            rtol=1e-3,
            msg="Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled.",
        )
        assert_tensors_close(
            image_slice_disabled,
            image_slice,
            atol=1e-2,
            rtol=1e-2,
            msg="Original outputs should match when fused QKV projections are disabled.",
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


class TestFluxControlInpaintPipelineMemory(FluxControlInpaintPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Flux Control Inpaint pipeline."""
