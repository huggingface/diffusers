import gc
import unittest

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    FasterCacheConfig,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)

from ...testing_utils import (
    Expectations,
    backend_empty_cache,
    nightly,
    numpy_cosine_similarity_distance,
    require_big_accelerator,
    slow,
    torch_device,
)
from ..test_pipelines_common import (
    FasterCacheTesterMixin,
    FirstBlockCacheTesterMixin,
    FluxIPAdapterTesterMixin,
    PipelineTesterMixin,
    PyramidAttentionBroadcastTesterMixin,
    check_qkv_fused_layers_exist,
)


class FluxPipelineFastTests(
    PipelineTesterMixin,
    FluxIPAdapterTesterMixin,
    PyramidAttentionBroadcastTesterMixin,
    FasterCacheTesterMixin,
    FirstBlockCacheTesterMixin,
    unittest.TestCase,
):
    pipeline_class = FluxPipeline
    params = frozenset(["prompt", "height", "width", "guidance_scale", "prompt_embeds", "pooled_prompt_embeds"])
    batch_params = frozenset(["prompt"])

    # there is no xformers processor for Flux
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    faster_cache_config = FasterCacheConfig(
        spatial_attention_block_skip_range=2,
        spatial_attention_timestep_skip_range=(-1, 901),
        unconditional_batch_skip_range=2,
        attention_weight_callback=lambda _: 0.5,
        is_guidance_distilled=True,
    )

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
            "image_encoder": None,
            "feature_extractor": None,
        }

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 48,
            "output_type": "np",
        }
        return inputs

    def test_flux_different_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt_2"] = "a different prompt"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = np.abs(output_same_prompt - output_different_prompts).max()

        # Outputs should be different here
        # For some reasons, they don't show large differences
        self.assertGreater(max_diff, 1e-6, "Outputs should be different for different prompts.")

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

    def test_flux_true_cfg(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)
        inputs.pop("generator")

        no_true_cfg_out = pipe(**inputs, generator=torch.manual_seed(0)).images[0]
        inputs["negative_prompt"] = "bad quality"
        inputs["true_cfg_scale"] = 2.0
        true_cfg_out = pipe(**inputs, generator=torch.manual_seed(0)).images[0]
        self.assertFalse(
            np.allclose(no_true_cfg_out, true_cfg_out), "Outputs should be different when true_cfg_scale is set."
        )


@nightly
@require_big_accelerator
class FluxPipelineSlowTests(unittest.TestCase):
    pipeline_class = FluxPipeline
    repo_id = "black-forest-labs/FLUX.1-schnell"

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
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
        self.assertLess(
            max_diff, 1e-4, f"Image slice is different from expected slice: {image_slice} != {expected_slice}"
        )


@slow
@require_big_accelerator
class FluxIPAdapterPipelineSlowTests(unittest.TestCase):
    pipeline_class = FluxPipeline
    repo_id = "black-forest-labs/FLUX.1-dev"
    image_encoder_pretrained_model_name_or_path = "openai/clip-vit-large-patch14"
    weight_name = "ip_adapter.safetensors"
    ip_adapter_repo_id = "XLabs-AI/flux-ip-adapter"

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
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
        self.assertLess(
            max_diff, 1e-4, f"Image slice is different from expected slice: {image_slice} != {expected_slice}"
        )
