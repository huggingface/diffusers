import gc
import unittest

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.testing_utils import (
    numpy_cosine_similarity_distance,
    require_big_gpu_with_torch_cuda,
    require_torch_multi_gpu,
    slow,
    torch_device,
)

from ..test_pipelines_common import (
    PipelineTesterMixin,
    check_qkv_fusion_matches_attn_procs_length,
    check_qkv_fusion_processors_exist,
)


class FluxPipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = FluxPipeline
    params = frozenset(["prompt", "height", "width", "guidance_scale", "prompt_embeds", "pooled_prompt_embeds"])
    batch_params = frozenset(["prompt"])

    # there is no xformers processor for Flux
    test_xformers_attention = False

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
        assert max_diff > 1e-6

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


@slow
@require_big_gpu_with_torch_cuda
@pytest.mark.big_gpu_with_torch_cuda
class FluxPipelineSlowTests(unittest.TestCase):
    pipeline_class = FluxPipeline
    repo_id = "black-forest-labs/FLUX.1-schnell"

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

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
        )
        pipe.enable_model_cpu_offload()

        inputs = self.get_inputs(torch_device)

        image = pipe(**inputs).images[0]
        image_slice = image[0, :10, :10]
        expected_slice = np.array(
            [
                0.3242,
                0.3203,
                0.3164,
                0.3164,
                0.3125,
                0.3125,
                0.3281,
                0.3242,
                0.3203,
                0.3301,
                0.3262,
                0.3242,
                0.3281,
                0.3242,
                0.3203,
                0.3262,
                0.3262,
                0.3164,
                0.3262,
                0.3281,
                0.3184,
                0.3281,
                0.3281,
                0.3203,
                0.3281,
                0.3281,
                0.3164,
                0.3320,
                0.3320,
                0.3203,
            ],
            dtype=np.float32,
        )

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), image_slice.flatten())

        assert max_diff < 1e-4

    @require_torch_multi_gpu
    @torch.no_grad()
    def test_flux_component_sharding(self):
        """
        internal note: test was run on `audace`.
        """

        ckpt_id = "black-forest-labs/FLUX.1-dev"
        dtype = torch.bfloat16
        prompt = "a photo of a cat with tiger-like look"

        pipeline = FluxPipeline.from_pretrained(
            ckpt_id,
            transformer=None,
            vae=None,
            device_map="balanced",
            max_memory={0: "16GB", 1: "16GB"},
            torch_dtype=dtype,
        )
        prompt_embeds, pooled_prompt_embeds, _ = pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=512
        )

        del pipeline.text_encoder
        del pipeline.text_encoder_2
        del pipeline.tokenizer
        del pipeline.tokenizer_2
        del pipeline

        gc.collect()
        torch.cuda.empty_cache()

        transformer = FluxTransformer2DModel.from_pretrained(
            ckpt_id, subfolder="transformer", device_map="auto", max_memory={0: "16GB", 1: "16GB"}, torch_dtype=dtype
        )
        pipeline = FluxPipeline.from_pretrained(
            ckpt_id,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            vae=None,
            transformer=transformer,
            torch_dtype=dtype,
        )

        height, width = 768, 1360
        # No need to wrap it up under `torch.no_grad()` as pipeline call method
        # is already wrapped under that.
        latents = pipeline(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=10,
            guidance_scale=3.5,
            height=height,
            width=width,
            output_type="latent",
            generator=torch.manual_seed(0),
        ).images
        latent_slice = latents[0, :3, :3].flatten().float().cpu().numpy()
        expected_slice = np.array([-0.377, -0.3008, -0.5117, -0.252, 0.0615, -0.3477, -0.1309, -0.1914, 0.1533])

        assert numpy_cosine_similarity_distance(latent_slice, expected_slice) < 1e-4

        del pipeline.transformer
        del pipeline

        gc.collect()
        torch.cuda.empty_cache()

        vae = AutoencoderKL.from_pretrained(ckpt_id, subfolder="vae", torch_dtype=dtype).to(torch_device)
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

        latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

        image = vae.decode(latents, return_dict=False)[0]
        image = image_processor.postprocess(image, output_type="np")
        image_slice = image[0, :3, :3, -1].flatten()
        expected_slice = np.array([0.127, 0.1113, 0.1055, 0.1172, 0.1172, 0.1074, 0.1191, 0.1191, 0.1152])

        assert numpy_cosine_similarity_distance(image_slice, expected_slice) < 1e-4

    @require_torch_multi_gpu
    @torch.no_grad()
    def test_flux_component_sharding_with_lora(self):
        """
        internal note: test was run on `audace`.
        """

        ckpt_id = "black-forest-labs/FLUX.1-dev"
        dtype = torch.bfloat16
        prompt = "jon snow eating pizza."

        pipeline = FluxPipeline.from_pretrained(
            ckpt_id,
            transformer=None,
            vae=None,
            device_map="balanced",
            max_memory={0: "16GB", 1: "16GB"},
            torch_dtype=dtype,
        )
        prompt_embeds, pooled_prompt_embeds, _ = pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=512
        )

        del pipeline.text_encoder
        del pipeline.text_encoder_2
        del pipeline.tokenizer
        del pipeline.tokenizer_2
        del pipeline

        gc.collect()
        torch.cuda.empty_cache()

        transformer = FluxTransformer2DModel.from_pretrained(
            ckpt_id, subfolder="transformer", device_map="auto", max_memory={0: "16GB", 1: "16GB"}, torch_dtype=dtype
        )
        pipeline = FluxPipeline.from_pretrained(
            ckpt_id,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            vae=None,
            transformer=transformer,
            torch_dtype=dtype,
        )
        pipeline.load_lora_weights("TheLastBen/Jon_Snow_Flux_LoRA", weight_name="jon_snow.safetensors")

        height, width = 768, 1360
        # No need to wrap it up under `torch.no_grad()` as pipeline call method
        # is already wrapped under that.
        latents = pipeline(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=10,
            guidance_scale=3.5,
            height=height,
            width=width,
            output_type="latent",
            generator=torch.manual_seed(0),
        ).images
        latent_slice = latents[0, :3, :3].flatten().float().cpu().numpy()
        expected_slice = np.array([-0.6523, -0.4961, -0.9141, -0.5, -0.2129, -0.6914, -0.375, -0.5664, -0.1699])

        assert numpy_cosine_similarity_distance(latent_slice, expected_slice) < 1e-4

        del pipeline.transformer
        del pipeline

        gc.collect()
        torch.cuda.empty_cache()

        vae = AutoencoderKL.from_pretrained(ckpt_id, subfolder="vae", torch_dtype=dtype).to(torch_device)
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

        latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

        image = vae.decode(latents, return_dict=False)[0]
        image = image_processor.postprocess(image, output_type="np")
        image_slice = image[0, :3, :3, -1].flatten()
        expected_slice = np.array([0.1211, 0.1094, 0.1035, 0.1094, 0.1113, 0.1074, 0.1133, 0.1133, 0.1094])

        assert numpy_cosine_similarity_distance(image_slice, expected_slice) < 1e-4
