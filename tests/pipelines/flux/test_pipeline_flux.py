import gc
import inspect
from typing import Any

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.loaders import FluxIPAdapterMixin

from ...models.transformers.test_models_transformer_flux import create_flux_ip_adapter_state_dict
from ...testing_utils import (
    Expectations,
    assert_tensors_close,
    backend_empty_cache,
    is_ip_adapter,
    nightly,
    numpy_cosine_similarity_distance,
    require_big_accelerator,
    slow,
    torch_device,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    FasterCacheTesterMixin,
    FirstBlockCacheTesterMixin,
    MagCacheTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
    PyramidAttentionBroadcastTesterMixin,
    TaylorSeerCacheTesterMixin,
)


class FluxPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = FluxPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "prompt_embeds", "pooled_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])

    def get_dummy_components(self, num_layers: int = 1, num_single_layers: int = 1):
        torch.manual_seed(0)
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=4,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
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
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
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
        return inputs


class TestFluxPipeline(FluxPipelineTesterConfig, PipelineTesterMixin):
    def test_flux_different_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)

        inputs = self.get_dummy_inputs()
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs()
        inputs["prompt_2"] = "a different prompt"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = (output_same_prompt - output_different_prompts).abs().max()

        # Outputs should be different here
        # For some reasons, they don't show large differences
        assert max_diff > 1e-6, "Outputs should be different for different prompts."

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

    def test_flux_true_cfg(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs.pop("generator")

        no_true_cfg_out = pipe(**inputs, generator=torch.manual_seed(0)).images[0]
        inputs["negative_prompt"] = "bad quality"
        inputs["true_cfg_scale"] = 2.0
        true_cfg_out = pipe(**inputs, generator=torch.manual_seed(0)).images[0]
        assert not torch.allclose(no_true_cfg_out, true_cfg_out), (
            "Outputs should be different when true_cfg_scale is set."
        )

    def test_flux_negative_embeds_shape_check(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)

        base_inputs = {
            "prompt_embeds": torch.randn(1, 4, 32, device=torch_device),
            "pooled_prompt_embeds": torch.randn(1, 32, device=torch_device),
            "negative_prompt_embeds": torch.randn(1, 5, 32, device=torch_device),
            "negative_pooled_prompt_embeds": torch.randn(1, 32, device=torch_device),
            "height": 16,
            "width": 16,
            "num_inference_steps": 1,
            "output_type": "latent",
        }

        with pytest.raises(ValueError, match="must have the same shape when passed directly"):
            pipe(**base_inputs, true_cfg_scale=2.0, generator=torch.manual_seed(0))

        pipe(**base_inputs, true_cfg_scale=1.0, generator=torch.manual_seed(0))


@is_ip_adapter
class TestFluxPipelineIPAdapter(FluxPipelineTesterConfig):
    """IP-Adapter tests for the Flux pipeline."""

    def test_pipeline_signature(self):
        parameters = inspect.signature(self.pipeline_class.__call__).parameters

        assert issubclass(self.pipeline_class, FluxIPAdapterMixin)
        assert "ip_adapter_image" in parameters, (
            "`ip_adapter_image` argument must be supported by the `__call__` method"
        )
        assert "ip_adapter_image_embeds" in parameters, (
            "`ip_adapter_image_embeds` argument must be supported by the `__call__` method"
        )

    def _get_dummy_image_embeds(self, image_embed_dim: int = 768):
        return torch.randn((1, 1, image_embed_dim), device=torch_device)

    def _modify_inputs_for_ip_adapter_test(self, inputs: dict[str, Any]):
        inputs["negative_prompt"] = ""
        if "true_cfg_scale" in inspect.signature(self.pipeline_class.__call__).parameters:
            inputs["true_cfg_scale"] = 4.0
        # Request torch outputs so comparisons run on torch tensors directly (see `BasePipelineTesterConfig`).
        inputs["output_type"] = "pt"
        inputs["return_dict"] = False
        return inputs

    def test_ip_adapter(self, expected_max_diff: float = 1e-4, expected_pipe_slice=None):
        r"""Tests for IP-Adapter.

        The following scenarios are tested:
          - Single IP-Adapter with scale=0 should produce same output as no IP-Adapter.
          - Multi IP-Adapter with scale=0 should produce same output as no IP-Adapter.
          - Single IP-Adapter with scale!=0 should produce different output compared to no IP-Adapter.
          - Multi IP-Adapter with scale!=0 should produce different output compared to no IP-Adapter.
        """
        # Raising the tolerance for this test when it's run on a CPU because we compare against static slices and
        # that can be shaky (with a VVVV low probability).
        expected_max_diff = 9e-4 if torch_device == "cpu" else expected_max_diff

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        image_embed_dim = (
            pipe.transformer.config.pooled_projection_dim
            if hasattr(pipe.transformer.config, "pooled_projection_dim")
            else 768
        )

        # forward pass without ip adapter
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs())
        if expected_pipe_slice is None:
            output_without_adapter = pipe(**inputs)[0]
        else:
            output_without_adapter = expected_pipe_slice

        # 1. Single IP-Adapter test cases
        adapter_state_dict = create_flux_ip_adapter_state_dict(pipe.transformer)
        # Load through the pipeline's public IP-Adapter API. `image_encoder_pretrained_model_name_or_path=None`
        # skips fetching a CLIP image encoder since we feed pre-computed `ip_adapter_image_embeds` directly.
        pipe.load_ip_adapter(adapter_state_dict, weight_name="", image_encoder_pretrained_model_name_or_path=None)

        # forward pass with single ip adapter, but scale=0 which should have no effect
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs())
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)]
        inputs["negative_ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)]
        pipe.set_ip_adapter_scale(0.0)
        output_without_adapter_scale = pipe(**inputs)[0]
        if expected_pipe_slice is not None:
            output_without_adapter_scale = output_without_adapter_scale[0, -3:, -3:, -1].flatten()

        # forward pass with single ip adapter, but with scale of adapter weights
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs())
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)]
        inputs["negative_ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)]
        pipe.set_ip_adapter_scale(42.0)
        output_with_adapter_scale = pipe(**inputs)[0]
        if expected_pipe_slice is not None:
            output_with_adapter_scale = output_with_adapter_scale[0, -3:, -3:, -1].flatten()

        assert_tensors_close(
            output_without_adapter_scale,
            output_without_adapter,
            atol=expected_max_diff,
            msg="Output without ip-adapter must be same as normal inference",
        )
        max_diff_with_adapter_scale = (output_with_adapter_scale - output_without_adapter).abs().max()
        assert max_diff_with_adapter_scale > 1e-2, "Output with ip-adapter must be different from normal inference"

        # 2. Multi IP-Adapter test cases
        adapter_state_dict_1 = create_flux_ip_adapter_state_dict(pipe.transformer)
        adapter_state_dict_2 = create_flux_ip_adapter_state_dict(pipe.transformer)
        pipe.load_ip_adapter(
            [adapter_state_dict_1, adapter_state_dict_2],
            weight_name=["", ""],
            image_encoder_pretrained_model_name_or_path=None,
        )

        # forward pass with multi ip adapter, but scale=0 which should have no effect
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs())
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)] * 2
        inputs["negative_ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)] * 2
        pipe.set_ip_adapter_scale([0.0, 0.0])
        output_without_multi_adapter_scale = pipe(**inputs)[0]
        if expected_pipe_slice is not None:
            output_without_multi_adapter_scale = output_without_multi_adapter_scale[0, -3:, -3:, -1].flatten()

        # forward pass with multi ip adapter, but with scale of adapter weights
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs())
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)] * 2
        inputs["negative_ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)] * 2
        pipe.set_ip_adapter_scale([42.0, 42.0])
        output_with_multi_adapter_scale = pipe(**inputs)[0]
        if expected_pipe_slice is not None:
            output_with_multi_adapter_scale = output_with_multi_adapter_scale[0, -3:, -3:, -1].flatten()

        assert_tensors_close(
            output_without_multi_adapter_scale,
            output_without_adapter,
            atol=expected_max_diff,
            msg="Output without multi-ip-adapter must be same as normal inference",
        )
        max_diff_with_multi_adapter_scale = (output_with_multi_adapter_scale - output_without_adapter).abs().max()
        assert max_diff_with_multi_adapter_scale > 1e-2, (
            "Output with multi-ip-adapter scale must be different from normal inference"
        )


class TestFluxPipelineMemory(FluxPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Flux pipeline."""


class TestFluxPipelinePyramidAttentionBroadcast(FluxPipelineTesterConfig, PyramidAttentionBroadcastTesterMixin):
    """Pyramid Attention Broadcast cache tests for the Flux pipeline."""


class TestFluxPipelineFasterCache(FluxPipelineTesterConfig, FasterCacheTesterMixin):
    """FasterCache tests for the Flux pipeline."""

    # Flux is guidance-distilled, so the FasterCache tester must skip the low/high-frequency-delta state checks.
    FASTER_CACHE_CONFIG = {
        "spatial_attention_block_skip_range": 2,
        "spatial_attention_timestep_skip_range": (-1, 901),
        "unconditional_batch_skip_range": 2,
        "attention_weight_callback": lambda _: 0.5,
        "is_guidance_distilled": True,
    }


class TestFluxPipelineFirstBlockCache(FluxPipelineTesterConfig, FirstBlockCacheTesterMixin):
    """First Block Cache tests for the Flux pipeline."""


class TestFluxPipelineTaylorSeerCache(FluxPipelineTesterConfig, TaylorSeerCacheTesterMixin):
    """TaylorSeer cache tests for the Flux pipeline."""


class TestFluxPipelineMagCache(FluxPipelineTesterConfig, MagCacheTesterMixin):
    """MagCache tests for the Flux pipeline."""


@nightly
@require_big_accelerator
class TestFluxPipelineSlow:
    pipeline_class = FluxPipeline
    repo_id = "black-forest-labs/FLUX.1-schnell"

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, seed=0):
        generator = torch.Generator(device="cpu").manual_seed(seed)

        prompt_embeds = torch.load(
            hf_hub_download(repo_id="diffusers/test-slices", repo_type="dataset", filename="flux/prompt_embeds.pt")
        ).to(torch_device)
        pooled_prompt_embeds = torch.load(
            hf_hub_download(
                repo_id="diffusers/test-slices", repo_type="dataset", filename="flux/pooled_prompt_embeds.pt"
            )
        ).to(torch_device)
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "max_sequence_length": 256,
            "output_type": "np",
            "generator": generator,
        }

    def test_flux_inference(self):
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id, torch_dtype=torch.bfloat16, text_encoder=None, text_encoder_2=None
        ).to(torch_device)

        inputs = self.get_inputs(torch_device)

        image = pipe(**inputs).images[0]
        image_slice = image[0, :10, :10]
        # fmt: off

        expected_slices = Expectations(
            {
                ("cuda", None): np.array([0.3242, 0.3203, 0.3164, 0.3164, 0.3125, 0.3125, 0.3281, 0.3242, 0.3203, 0.3301, 0.3262, 0.3242, 0.3281, 0.3242, 0.3203, 0.3262, 0.3262, 0.3164, 0.3262, 0.3281, 0.3184, 0.3281, 0.3281, 0.3203, 0.3281, 0.3281, 0.3164, 0.3320, 0.3320, 0.3203], dtype=np.float32,),
                ("xpu", 3): np.array([0.3301, 0.3281, 0.3359, 0.3203, 0.3203, 0.3281, 0.3281, 0.3301, 0.3340, 0.3281, 0.3320, 0.3359, 0.3281, 0.3301, 0.3320, 0.3242, 0.3301, 0.3281, 0.3242, 0.3320, 0.3320, 0.3281, 0.3320, 0.3320, 0.3262, 0.3320, 0.3301, 0.3301, 0.3359, 0.3320], dtype=np.float32,),
            }
        )
        expected_slice = expected_slices.get_expectation()
        # fmt: on

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), image_slice.flatten())
        assert max_diff < 1e-4, f"Image slice is different from expected slice: {image_slice} != {expected_slice}"


@slow
@require_big_accelerator
class TestFluxIPAdapterPipelineSlow:
    pipeline_class = FluxPipeline
    repo_id = "black-forest-labs/FLUX.1-dev"
    image_encoder_pretrained_model_name_or_path = "openai/clip-vit-large-patch14"
    weight_name = "ip_adapter.safetensors"
    ip_adapter_repo_id = "XLabs-AI/flux-ip-adapter"

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        prompt_embeds = torch.load(
            hf_hub_download(repo_id="diffusers/test-slices", repo_type="dataset", filename="flux/prompt_embeds.pt")
        )
        pooled_prompt_embeds = torch.load(
            hf_hub_download(
                repo_id="diffusers/test-slices", repo_type="dataset", filename="flux/pooled_prompt_embeds.pt"
            )
        )
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        ip_adapter_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            "ip_adapter_image": ip_adapter_image,
            "num_inference_steps": 2,
            "guidance_scale": 3.5,
            "true_cfg_scale": 4.0,
            "max_sequence_length": 256,
            "output_type": "np",
            "generator": generator,
        }

    def test_flux_ip_adapter_inference(self):
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id, torch_dtype=torch.bfloat16, text_encoder=None, text_encoder_2=None
        )
        pipe.load_ip_adapter(
            self.ip_adapter_repo_id,
            weight_name=self.weight_name,
            image_encoder_pretrained_model_name_or_path=self.image_encoder_pretrained_model_name_or_path,
        )
        pipe.set_ip_adapter_scale(1.0)
        pipe.enable_model_cpu_offload()

        inputs = self.get_inputs(torch_device)

        image = pipe(**inputs).images[0]
        image_slice = image[0, :10, :10]

        # fmt: off
        expected_slice = np.array(
            [0.1855, 0.1680, 0.1406, 0.1953, 0.1699, 0.1465, 0.2012, 0.1738, 0.1484, 0.2051, 0.1797, 0.1523, 0.2012, 0.1719, 0.1445, 0.2070, 0.1777, 0.1465, 0.2090, 0.1836, 0.1484, 0.2129, 0.1875, 0.1523, 0.2090, 0.1816, 0.1484, 0.2110, 0.1836, 0.1543],
            dtype=np.float32,
        )
        # fmt: on

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), image_slice.flatten())
        assert max_diff < 1e-4, f"Image slice is different from expected slice: {image_slice} != {expected_slice}"
