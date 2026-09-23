import gc

import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    PriorTransformer,
    StableUnCLIPPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer

from ...testing_utils import (
    backend_empty_cache,
    backend_max_memory_allocated,
    backend_reset_max_memory_allocated,
    backend_reset_peak_memory_stats,
    enable_full_determinism,
    load_numpy,
    nightly,
    require_torch_accelerator,
    torch_device,
)
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import assert_mean_pixel_difference
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class StableUnCLIPPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableUnCLIPPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    output_shape = (3, 32, 32)

    def get_dummy_components(self):
        embedder_hidden_size = 32
        embedder_projection_dim = embedder_hidden_size

        # prior components

        torch.manual_seed(0)
        prior_tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        torch.manual_seed(0)
        prior_text_encoder = CLIPTextModelWithProjection(
            CLIPTextConfig(
                bos_token_id=0,
                eos_token_id=2,
                hidden_size=embedder_hidden_size,
                projection_dim=embedder_projection_dim,
                intermediate_size=37,
                layer_norm_eps=1e-05,
                num_attention_heads=4,
                num_hidden_layers=5,
                pad_token_id=1,
                vocab_size=1000,
            )
        )

        torch.manual_seed(0)
        prior = PriorTransformer(
            num_attention_heads=2,
            attention_head_dim=12,
            embedding_dim=embedder_projection_dim,
            num_layers=1,
        )

        torch.manual_seed(0)
        prior_scheduler = DDPMScheduler(
            variance_type="fixed_small_log",
            prediction_type="sample",
            num_train_timesteps=1000,
            clip_sample=True,
            clip_sample_range=5.0,
            beta_schedule="squaredcos_cap_v2",
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
            # prior components
            "prior_tokenizer": prior_tokenizer,
            "prior_text_encoder": prior_text_encoder,
            "prior": prior,
            "prior_scheduler": prior_scheduler,
            # image noising components
            "image_normalizer": image_normalizer,
            "image_noising_scheduler": image_noising_scheduler,
            # regular denoising components
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "prior_num_inference_steps": 2,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestStableUnCLIPPipeline(StableUnCLIPPipelineTesterConfig, PipelineTesterMixin):
    # Overriding PipelineTesterMixin::test_inference_batch_single_identical
    # because UnCLIP undeterminism requires a looser check.
    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    @pytest.mark.skip("Test not supported because of the use of `_encode_prior_prompt()`.")
    def test_encode_prompt_works_in_isolation(self):
        pass


class TestStableUnCLIPPipelineMemory(StableUnCLIPPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Stable unCLIP pipeline."""


@nightly
@require_torch_accelerator
class TestStableUnCLIPPipelineIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        # clean up the VRAM before and after each test
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_stable_unclip(self):
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/stable_unclip_2_1_l_anime_turtle_fp16.npy"
        )

        pipe = StableUnCLIPPipeline.from_pretrained("fusing/stable-unclip-2-1-l", torch_dtype=torch.float16)
        pipe.set_progress_bar_config(disable=None)
        # stable unclip will oom when integration tests are run on a V100,
        # so turn on memory savings
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()

        generator = torch.Generator(device="cpu").manual_seed(0)
        output = pipe("anime turtle", generator=generator, output_type="np")

        image = output.images[0]

        assert image.shape == (768, 768, 3)

        assert_mean_pixel_difference(image, expected_image)

    def test_stable_unclip_pipeline_with_sequential_cpu_offloading(self):
        backend_empty_cache(torch_device)
        backend_reset_max_memory_allocated(torch_device)
        backend_reset_peak_memory_stats(torch_device)

        pipe = StableUnCLIPPipeline.from_pretrained("fusing/stable-unclip-2-1-l", torch_dtype=torch.float16)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()

        _ = pipe(
            "anime turtle",
            prior_num_inference_steps=2,
            num_inference_steps=2,
            output_type="np",
        )

        mem_bytes = backend_max_memory_allocated(torch_device)
        # make sure that less than 7 GB is allocated
        assert mem_bytes < 7 * 10**9
