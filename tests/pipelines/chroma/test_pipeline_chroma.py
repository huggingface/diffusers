import torch
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKL, ChromaPipeline, ChromaTransformer2DModel, FlowMatchEulerDiscreteScheduler

from ...testing_utils import assert_tensors_close, torch_device
from ..flux.testing_utils import FluxIPAdapterTesterMixin
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
    check_qkv_fused_layers_exist,
)


class ChromaPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = ChromaPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    output_shape = (3, 8, 8)

    def get_dummy_components(self, num_layers: int = 1, num_single_layers: int = 1):
        torch.manual_seed(0)
        transformer = ChromaTransformer2DModel(
            patch_size=1,
            in_channels=4,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            axes_dims_rope=[4, 4, 8],
            approximator_hidden_dim=32,
            approximator_layers=1,
            approximator_num_channels=16,
        )

        torch.manual_seed(0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        # `eval()` because a directly constructed model stays in training mode, which leaves T5's
        # dropout active and makes the pipeline outputs non-deterministic across calls.
        text_encoder = T5EncoderModel(config).eval()

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

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
            "tokenizer": tokenizer,
            "transformer": transformer,
            "vae": vae,
            "image_encoder": None,
            "feature_extractor": None,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "negative_prompt": "bad, ugly",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 48,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestChromaPipeline(ChromaPipelineTesterConfig, PipelineTesterMixin):
    def test_chroma_different_prompts(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = "a different prompt"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = (output_same_prompt - output_different_prompts).abs().max()

        # Outputs should be different here
        # For some reasons, they don't show large differences
        assert max_diff > 1e-6, "Outputs should be different for different prompts."

    def test_fused_qkv_projections(self):
        # Run on CPU to keep the seeded generator deterministic across the three forward passes.
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

    def test_chroma_image_output_shape(self):
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


class TestChromaPipelineIPAdapter(ChromaPipelineTesterConfig, FluxIPAdapterTesterMixin):
    """IP-Adapter tests for the Chroma pipeline."""


class TestChromaPipelineMemory(ChromaPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Chroma pipeline."""
