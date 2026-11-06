import gc
import random

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

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Img2ImgPipeline,
)
from diffusers.utils import load_image

from ...testing_utils import (
    Expectations,
    assert_tensors_close,
    backend_empty_cache,
    floats_tensor,
    numpy_cosine_similarity_distance,
    require_big_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


class StableDiffusion3Img2ImgPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusion3Img2ImgPipeline
    # `TEXT_GUIDED_IMAGE_VARIATION_PARAMS` without `height` / `width`: the output resolution follows the image.
    required_input_params_in_call_signature = frozenset(
        ["prompt", "image", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
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
            joint_attention_dim=32,
            caption_projection_dim=32,
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
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(0))
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "strength": 0.8,
        }


class TestStableDiffusion3Img2ImgPipeline(StableDiffusion3Img2ImgPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images[0]
        generated_slice = image.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])

        # fmt: off
        expected_slice = torch.tensor([0.4301, 0.5094, 0.4512, 0.4631, 0.3923, 0.3261, 0.3774, 0.6466, 0.5143, 0.5031, 0.5651, 0.5698, 0.3936, 0.5979, 0.5575, 0.5804])
        # fmt: on

        assert_tensors_close(generated_slice, expected_slice, atol=1e-3, msg="Output does not match expected slice.")


class TestStableDiffusion3Img2ImgPipelineMemory(StableDiffusion3Img2ImgPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SD3 img2img pipeline."""


@slow
@require_big_accelerator
class TestStableDiffusion3Img2ImgPipelineSlow:
    pipeline_class = StableDiffusion3Img2ImgPipeline
    repo_id = "stabilityai/stable-diffusion-3-medium-diffusers"

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, seed=0):
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/sketch-mountains-input.png"
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)

        return {
            "prompt": "A photo of a cat",
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "generator": generator,
            "image": init_image,
        }

    def test_sd3_img2img_inference(self):
        torch.manual_seed(0)
        pipe = self.pipeline_class.from_pretrained(self.repo_id, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload(device=torch_device)
        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images[0]
        image_slice = image[0, :10, :10]

        # fmt: off
        expected_slices = Expectations(
            {
                ("xpu", 3): np.array([0.5117, 0.4421, 0.3852, 0.5044, 0.4219, 0.3262, 0.5024, 0.4329, 0.3276, 0.4978, 0.4412, 0.3355, 0.4983, 0.4338, 0.3279, 0.4893, 0.4241, 0.3129, 0.4875, 0.4253, 0.3030, 0.4961, 0.4267, 0.2988, 0.5029, 0.4255, 0.3054, 0.5132, 0.4248, 0.3222]),
                ("cuda", 7): np.array([0.5435, 0.4673, 0.5732, 0.4438, 0.3557, 0.4912, 0.4331, 0.3491, 0.4915, 0.4287, 0.347, 0.4849, 0.4355, 0.3469, 0.4871, 0.4431, 0.3538, 0.4912, 0.4521, 0.3643, 0.5059, 0.4587, 0.373, 0.5166, 0.4685, 0.3845, 0.5264, 0.4746, 0.3914, 0.5342]),
                ("cuda", 8): np.array([0.5146, 0.4385, 0.3826, 0.5098, 0.4150, 0.3218, 0.5142, 0.4312, 0.3298, 0.5127, 0.4431, 0.3411, 0.5171, 0.4424, 0.3374, 0.5088, 0.4348, 0.3242, 0.5073, 0.4380, 0.3174, 0.5132, 0.4397, 0.3115, 0.5132, 0.4343, 0.3118, 0.5219, 0.4328, 0.3256]),
            }
        )
        # fmt: on

        expected_slice = expected_slices.get_expectation()

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), image_slice.flatten())

        assert max_diff < 1e-4, f"Outputs are not close enough, got {max_diff}"
