import gc

import numpy as np
import pytest
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
)

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, StableDiffusion3Pipeline

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    numpy_cosine_similarity_distance,
    require_big_accelerator,
    slow,
    torch_device,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


class StableDiffusion3PipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusion3Pipeline
    required_input_params_in_call_signature = frozenset(
        [
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (3, 32, 32)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = SD3Transformer2DModel(
            sample_size=32,
            patch_size=1,
            in_channels=4,
            num_layers=1,
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

        torch.manual_seed(0)
        text_encoder = CLIPTextModelWithProjection(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_2 = CLIPTextModelWithProjection(clip_text_encoder_config)

        torch.manual_seed(0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder_3 = T5EncoderModel(config)

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_3 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
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
            "image_encoder": None,
            "feature_extractor": None,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestStableDiffusion3Pipeline(StableDiffusion3PipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images[0]
        generated_slice = image.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])

        # fmt: off
        expected_slice = torch.tensor([0.5034, 0.5052, 0.5329, 0.3051, 0.3103, 0.2199, 0.2012, 0.2736, 0.4372, 0.5862, 0.5706, 0.5178, 0.5872, 0.5969, 0.5454, 0.4805])
        # fmt: on

        assert_tensors_close(generated_slice, expected_slice, atol=1e-3, msg="Output does not match expected slice.")

    def test_skip_guidance_layers(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()

        output_full = pipe(**inputs)[0]

        inputs_with_skip = inputs.copy()
        inputs_with_skip["skip_guidance_layers"] = [0]
        output_skip = pipe(**inputs_with_skip)[0]

        assert not torch.allclose(output_full, output_skip, atol=1e-5), "Outputs should differ when layers are skipped"
        assert output_full.shape == output_skip.shape, "Outputs should have the same shape"

        inputs["num_images_per_prompt"] = 2
        output_full = pipe(**inputs)[0]

        inputs_with_skip = inputs.copy()
        inputs_with_skip["skip_guidance_layers"] = [0]
        output_skip = pipe(**inputs_with_skip)[0]

        assert not torch.allclose(output_full, output_skip, atol=1e-5), "Outputs should differ when layers are skipped"
        assert output_full.shape == output_skip.shape, "Outputs should have the same shape"


class TestStableDiffusion3PipelineMemory(StableDiffusion3PipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SD3 pipeline."""


@slow
@require_big_accelerator
class TestStableDiffusion3PipelineSlow:
    pipeline_class = StableDiffusion3Pipeline
    repo_id = "stabilityai/stable-diffusion-3-medium-diffusers"

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, seed=0):
        generator = torch.Generator(device="cpu").manual_seed(seed)

        return {
            "prompt": "A photo of a cat",
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "generator": generator,
        }

    def test_sd3_inference(self):
        pipe = self.pipeline_class.from_pretrained(self.repo_id, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload(device=torch_device)

        inputs = self.get_inputs(torch_device)

        image = pipe(**inputs).images[0]
        image_slice = image[0, :10, :10]
        # fmt: off
        expected_slice = np.array([0.4648, 0.4404, 0.4177, 0.5063, 0.4800, 0.4287, 0.5425, 0.5190, 0.4717, 0.5430, 0.5195, 0.4766, 0.5361, 0.5122, 0.4612, 0.4871, 0.4749, 0.4058, 0.4756, 0.4678, 0.3804, 0.4832, 0.4822, 0.3799, 0.5103, 0.5034, 0.3953, 0.5073, 0.4839, 0.3884])
        # fmt: on

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), image_slice.flatten())

        assert max_diff < 1e-4
