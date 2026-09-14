import gc
import random

import pytest
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableUnCLIPImg2ImgPipeline, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    backend_max_memory_allocated,
    backend_reset_max_memory_allocated,
    backend_reset_peak_memory_stats,
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    nightly,
    require_torch_accelerator,
    skip_mps,
    torch_device,
)
from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ..test_pipelines_common import assert_mean_pixel_difference
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class StableUnCLIPImg2ImgPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableUnCLIPImg2ImgPipeline
    required_input_params_in_call_signature = TEXT_GUIDED_IMAGE_VARIATION_PARAMS
    batch_input_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    output_shape = (3, 32, 32)

    def get_dummy_components(self):
        embedder_hidden_size = 32
        embedder_projection_dim = embedder_hidden_size

        # image encoding components

        feature_extractor = CLIPImageProcessor(crop_size=32, size=32)

        torch.manual_seed(0)
        image_encoder = CLIPVisionModelWithProjection(
            CLIPVisionConfig(
                hidden_size=embedder_hidden_size,
                projection_dim=embedder_projection_dim,
                num_hidden_layers=5,
                num_attention_heads=4,
                image_size=32,
                intermediate_size=37,
                patch_size=1,
            )
        )

        # regular denoising components

        torch.manual_seed(0)
        image_normalizer = StableUnCLIPImageNormalizer(embedding_dim=embedder_hidden_size)
        image_noising_scheduler = DDPMScheduler(beta_schedule="squaredcos_cap_v2")

        torch.manual_seed(0)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        torch.manual_seed(0)
        text_encoder = CLIPTextModel(
            CLIPTextConfig(
                bos_token_id=0,
                eos_token_id=2,
                hidden_size=embedder_hidden_size,
                projection_dim=32,
                intermediate_size=37,
                layer_norm_eps=1e-05,
                num_attention_heads=4,
                num_hidden_layers=5,
                pad_token_id=1,
                vocab_size=1000,
            )
        )

        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            block_out_channels=(32, 64),
            attention_head_dim=(2, 4),
            class_embed_type="projection",
            # The class embeddings are the noise augmented image embeddings.
            # I.e. the image embeddings concated with the noised embeddings of the same dimension
            projection_class_embeddings_input_dim=embedder_projection_dim * 2,
            cross_attention_dim=embedder_hidden_size,
            layers_per_block=1,
            upcast_attention=True,
            use_linear_projection=True,
        )

        torch.manual_seed(0)
        scheduler = DDIMScheduler(
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012,
            prediction_type="v_prediction",
            set_alpha_to_one=False,
            steps_offset=1,
        )

        torch.manual_seed(0)
        vae = AutoencoderKL()

        return {
            # image encoding components
            "feature_extractor": feature_extractor,
            "image_encoder": image_encoder.eval(),
            # image noising components
            "image_normalizer": image_normalizer.eval(),
            "image_noising_scheduler": image_noising_scheduler,
            # regular denoising components
            "tokenizer": tokenizer,
            "text_encoder": text_encoder.eval(),
            "unet": unet.eval(),
            "scheduler": scheduler,
            "vae": vae.eval(),
        }

    def get_dummy_inputs(self, pil_image=True):
        input_image = floats_tensor((1, 3, 32, 32), rng=random.Random(0))

        if pil_image:
            input_image = input_image * 0.5 + 0.5
            input_image = input_image.clamp(0, 1)
            input_image = input_image.permute(0, 2, 3, 1).float().numpy()
            input_image = DiffusionPipeline.numpy_to_pil(input_image)[0]

        return {
            "prompt": "An anime racoon running a marathon",
            "image": input_image,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestStableUnCLIPImg2ImgPipeline(StableUnCLIPImg2ImgPipelineTesterConfig, PipelineTesterMixin):
    @skip_mps
    def test_image_embeds_none(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs.update({"image_embeds": None})
        image = sd_pipe(**inputs).images
        image_slice = image[0, -1, -3:, -3:]

        assert image.shape == (1, *self.output_shape)
        # fmt: off
        expected_slice = torch.tensor([0.4661, 0.8109, 0.6731, 0.4309, 0.6674, 0.7538, 0.3841, 0.4951, 0.5243])
        # fmt: on
        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-3)

    # Overriding PipelineTesterMixin::test_inference_batch_single_identical
    # because undeterminism requires a looser check.
    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    @pytest.mark.skip("Test not supported at the moment.")
    def test_encode_prompt_works_in_isolation(self):
        pass


class TestStableUnCLIPImg2ImgPipelineMemory(StableUnCLIPImg2ImgPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the unCLIP img2img pipeline."""


@nightly
@require_torch_accelerator
class TestStableUnCLIPImg2ImgPipelineIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        # clean up the VRAM before and after each test
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_stable_unclip_l_img2img(self):
        input_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/turtle.png"
        )

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/stable_unclip_2_1_l_img2img_anime_turtle_fp16.npy"
        )

        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            "fusing/stable-unclip-2-1-l-img2img", torch_dtype=torch.float16
        )
        pipe.set_progress_bar_config(disable=None)
        # stable unclip will oom when integration tests are run on a V100,
        # so turn on memory savings
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()

        generator = torch.Generator(device="cpu").manual_seed(0)
        output = pipe(input_image, "anime turtle", generator=generator, output_type="np")

        image = output.images[0]

        assert image.shape == (768, 768, 3)

        assert_mean_pixel_difference(image, expected_image)

    def test_stable_unclip_h_img2img(self):
        input_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/turtle.png"
        )

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/stable_unclip_2_1_h_img2img_anime_turtle_fp16.npy"
        )

        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            "fusing/stable-unclip-2-1-h-img2img", torch_dtype=torch.float16
        )
        pipe.set_progress_bar_config(disable=None)
        # stable unclip will oom when integration tests are run on a V100,
        # so turn on memory savings
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()

        generator = torch.Generator(device="cpu").manual_seed(0)
        output = pipe(input_image, "anime turtle", generator=generator, output_type="np")

        image = output.images[0]

        assert image.shape == (768, 768, 3)

        assert_mean_pixel_difference(image, expected_image)

    def test_stable_unclip_img2img_pipeline_with_sequential_cpu_offloading(self):
        input_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/turtle.png"
        )

        backend_empty_cache(torch_device)
        backend_reset_max_memory_allocated(torch_device)
        backend_reset_peak_memory_stats(torch_device)

        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            "fusing/stable-unclip-2-1-h-img2img", torch_dtype=torch.float16
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()

        _ = pipe(
            input_image,
            "anime turtle",
            num_inference_steps=2,
            output_type="np",
        )

        mem_bytes = backend_max_memory_allocated(torch_device)
        # make sure that less than 7 GB is allocated
        assert mem_bytes < 7 * 10**9
