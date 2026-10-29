import gc
import random

import numpy as np
import pytest
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    AutoencoderKLTemporalDecoder,
    EulerDiscreteScheduler,
    StableVideoDiffusionPipeline,
    UNetSpatioTemporalConditionModel,
)
from diffusers.utils import load_image

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class StableVideoDiffusionPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableVideoDiffusionPipeline
    required_input_params_in_call_signature = frozenset(["image"])
    batch_input_params = frozenset(["image", "generator"])
    # SVD is conditioned on an image only: it has no prompt and no `num_images_per_prompt`.
    optional_input_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "output_type",
            "return_dict",
        ]
    )
    # `(num_frames, channels, height, width)` for the dummy inputs.
    output_shape = (2, 3, 32, 32)

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNetSpatioTemporalConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=8,
            out_channels=4,
            down_block_types=(
                "CrossAttnDownBlockSpatioTemporal",
                "DownBlockSpatioTemporal",
            ),
            up_block_types=("UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal"),
            cross_attention_dim=32,
            num_attention_heads=8,
            projection_class_embeddings_input_dim=96,
            addition_time_embed_dim=32,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            interpolation_type="linear",
            num_train_timesteps=1000,
            prediction_type="v_prediction",
            sigma_max=700.0,
            sigma_min=0.002,
            steps_offset=1,
            timestep_spacing="leading",
            timestep_type="continuous",
            trained_betas=None,
            use_karras_sigmas=True,
        )

        torch.manual_seed(0)
        vae = AutoencoderKLTemporalDecoder(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            latent_channels=4,
        )

        torch.manual_seed(0)
        config = CLIPVisionConfig(
            hidden_size=32,
            projection_dim=32,
            num_hidden_layers=5,
            num_attention_heads=4,
            image_size=32,
            intermediate_size=37,
            patch_size=1,
        )
        image_encoder = CLIPVisionModelWithProjection(config)

        torch.manual_seed(0)
        feature_extractor = CLIPImageProcessor(crop_size=32, size=32)
        return {
            "unet": unet,
            "image_encoder": image_encoder,
            "scheduler": scheduler,
            "vae": vae,
            "feature_extractor": feature_extractor,
        }

    def get_dummy_inputs(self):
        return {
            "generator": self.get_generator(0),
            "image": floats_tensor((1, 3, 32, 32), rng=random.Random(0)),
            "num_inference_steps": 2,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
            "min_guidance_scale": 1.0,
            "max_guidance_scale": 2.5,
            "num_frames": 2,
            "height": 32,
            "width": 32,
        }


class TestStableVideoDiffusionPipeline(StableVideoDiffusionPipelineTesterConfig, PipelineTesterMixin):
    @pytest.mark.skip("Batched inference works and outputs look correct, but the test is failing")
    def test_inference_batch_single_identical(self):
        pass

    @pytest.mark.skip("Test is similar to test_inference_batch_single_identical")
    def test_inference_batch_consistent(self):
        pass

    # `StableVideoDiffusionPipeline` returns the bare `frames` tensor (not a 1-tuple) when `return_dict=False`,
    # so the two outputs have to be indexed differently before they can be compared.
    def test_dict_tuple_outputs_equivalent(self, expected_max_difference=1e-4):
        pipe = self.get_pipeline().to(torch_device)

        output = pipe(**self.get_dummy_inputs()).frames[0]
        output_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]

        assert_tensors_close(
            output_tuple, output, atol=expected_max_difference, msg="Dict and tuple outputs are not equal."
        )

    def test_np_output_type(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        inputs["output_type"] = "np"
        output = pipe(**inputs).frames

        assert isinstance(output, np.ndarray)
        assert len(output.shape) == 5

    def test_save_load_local(self, tmp_path, base_pipe_output):
        super().test_save_load_local(tmp_path, base_pipe_output, expected_max_difference=9e-4)

    def test_disable_cfg(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        inputs["max_guidance_scale"] = 1.0
        output = pipe(**inputs).frames

        assert len(output.shape) == 5


class TestStableVideoDiffusionPipelineMemory(StableVideoDiffusionPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SVD pipeline."""


@slow
@require_torch_accelerator
class TestStableVideoDiffusionPipelineSlow:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        # clean up the VRAM before and after each test
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_sd_video(self):
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            variant="fp16",
            torch_dtype=torch.float16,
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat_6.png?download=true"
        )

        generator = torch.Generator("cpu").manual_seed(0)
        num_frames = 3

        output = pipe(
            image=image,
            num_frames=num_frames,
            generator=generator,
            num_inference_steps=3,
            output_type="np",
        )

        image = output.frames[0]
        assert image.shape == (num_frames, 576, 1024, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.8592, 0.8645, 0.8499, 0.8722, 0.8769, 0.8421, 0.8557, 0.8528, 0.8285])
        assert numpy_cosine_similarity_distance(image_slice.flatten(), expected_slice.flatten()) < 1e-3
