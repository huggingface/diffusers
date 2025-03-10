import gc
import random
import unittest

import numpy as np
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
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    nightly,
    require_torch_gpu,
    skip_mps,
    torch_device,
)

from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ..test_pipelines_common import (
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
    assert_mean_pixel_difference,
)


enable_full_determinism()


class StableUnCLIPImg2ImgPipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableUnCLIPImg2ImgPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = frozenset(
        []
    )  # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess
    image_latents_params = frozenset([])

    supports_dduf = False

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

        components = {
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

        return components

    def get_dummy_inputs(self, device, seed=0, pil_image=True):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        input_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)

        if pil_image:
            input_image = input_image * 0.5 + 0.5
            input_image = input_image.clamp(0, 1)
            input_image = input_image.cpu().permute(0, 2, 3, 1).float().numpy()
            input_image = DiffusionPipeline.numpy_to_pil(input_image)[0]

        return {
            "prompt": "An anime racoon running a marathon",
            "image": input_image,
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "np",
        }

    @skip_mps
    def test_image_embeds_none(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableUnCLIPImg2ImgPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs.update({"image_embeds": None})
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4397, 0.7080, 0.5590, 0.4255, 0.7181, 0.5938, 0.4051, 0.3720, 0.5116])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    # Overriding PipelineTesterMixin::test_attention_slicing_forward_pass
    # because GPU undeterminism requires a looser check.
    def test_attention_slicing_forward_pass(self):
        test_max_difference = torch_device in ["cpu", "mps"]

        self._test_attention_slicing_forward_pass(test_max_difference=test_max_difference)

    # Overriding PipelineTesterMixin::test_inference_batch_single_identical
    # because undeterminism requires a looser check.
    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=1e-3)

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(test_max_difference=False)

    @unittest.skip("Test not supported at the moment.")
    def test_encode_prompt_works_in_isolation(self):
        pass


@nightly
@require_torch_gpu
class StableUnCLIPImg2ImgPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

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
        output = pipe(input_image, "anime turle", generator=generator, output_type="np")

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
        output = pipe(input_image, "anime turle", generator=generator, output_type="np")

        image = output.images[0]

        assert image.shape == (768, 768, 3)

        assert_mean_pixel_difference(image, expected_image)

    def test_stable_unclip_img2img_pipeline_with_sequential_cpu_offloading(self):
        input_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/turtle.png"
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

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

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 7 GB is allocated
        assert mem_bytes < 7 * 10**9
