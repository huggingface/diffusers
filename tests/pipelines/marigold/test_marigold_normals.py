# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
# Copyright 2024-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information and citation instructions are available on the
# Marigold project website: https://marigoldcomputervision.github.io
# --------------------------------------------------------------------------
import gc

import numpy as np
import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    AutoencoderTiny,
    LCMScheduler,
    MarigoldNormalsPipeline,
    UNet2DConditionModel,
)

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    load_image,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class MarigoldNormalsPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = MarigoldNormalsPipeline
    required_input_params_in_call_signature = frozenset(["image"])
    batch_input_params = frozenset(["image"])
    # Marigold predicts a normals map and takes no prompt: it exposes neither `num_images_per_prompt` nor
    # `num_videos_per_prompt`.
    optional_input_params = frozenset(["num_inference_steps", "generator", "latents", "output_type", "return_dict"])
    output_shape = (3, 32, 32)

    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            time_cond_proj_dim=time_cond_proj_dim,
            sample_size=32,
            in_channels=8,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        torch.manual_seed(0)
        scheduler = LCMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            prediction_type="v_prediction",
            set_alpha_to_one=False,
            steps_offset=1,
            beta_schedule="scaled_linear",
            clip_sample=False,
            thresholding=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        return {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "prediction_type": "normals",
            "use_full_z_range": True,
        }

    def get_dummy_tiny_autoencoder(self):
        return AutoencoderTiny(in_channels=3, out_channels=3, latent_channels=4)

    def get_dummy_inputs(self, seed: int = 0):
        # Marigold validates that the input image lies in [0, 1] (`MarigoldImageProcessor.check_image_values_range`),
        # so the Gaussian is squashed into that range rather than clipped against it.
        image = torch.randn((1, 3, 32, 32), generator=self.get_generator(seed)).sigmoid()
        return {
            "image": image,
            "num_inference_steps": 1,
            "processing_resolution": 0,
            "generator": self.get_generator(seed),
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestMarigoldNormalsPipeline(MarigoldNormalsPipelineTesterConfig, PipelineTesterMixin):
    def _test_marigold_normals(
        self,
        generator_seed: int = 0,
        expected_slice: torch.Tensor = None,
        atol: float = 1e-4,
        **pipe_kwargs,
    ):
        # Run on CPU: the expected slices below are CPU-specific.
        pipe = self.get_pipeline()

        pipe_inputs = self.get_dummy_inputs(seed=generator_seed)
        pipe_inputs.update(**pipe_kwargs)

        prediction = pipe(**pipe_inputs).prediction  # [N,3,H,W] for `output_type="pt"`

        prediction_slice = prediction[0, -1, -3:, -3:].flatten()

        if pipe_inputs.get("match_input_resolution", True):
            assert prediction.shape == (1, *self.output_shape), "Unexpected output resolution"
        else:
            assert prediction.shape[0] == 1 and prediction.shape[1] == 3, "Unexpected output dimensions"
            assert max(prediction.shape[2:4]) == pipe_inputs.get("processing_resolution", 768), (
                "Unexpected output resolution"
            )

        assert_tensors_close(prediction_slice, expected_slice, atol=atol)

    def test_marigold_depth_dummy_defaults(self):
        self._test_marigold_normals(
            expected_slice=torch.tensor(
                [-0.01402, 0.54840, -0.00052, -0.27905, -0.16117, -0.55048, 0.63950, 0.53618, -0.26825]
            ),
        )

    def test_marigold_depth_dummy_G0_S1_P32_E1_B1_M1(self):
        self._test_marigold_normals(
            generator_seed=0,
            expected_slice=torch.tensor(
                [-0.01402, 0.54840, -0.00052, -0.27905, -0.16117, -0.55048, 0.63950, 0.53618, -0.26825]
            ),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P16_E1_B1_M1(self):
        self._test_marigold_normals(
            generator_seed=0,
            expected_slice=torch.tensor(
                [-0.54494, -0.31659, -0.17026, -0.49534, -0.65212, -0.66506, -0.28120, -0.45898, -0.52408]
            ),
            num_inference_steps=1,
            processing_resolution=16,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G2024_S1_P32_E1_B1_M1(self):
        self._test_marigold_normals(
            generator_seed=2024,
            expected_slice=torch.tensor(
                [0.75286, -0.88962, -0.11049, 0.06276, -0.55335, -0.70896, 0.52707, -0.27555, -0.43498]
            ),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S2_P32_E1_B1_M1(self):
        self._test_marigold_normals(
            generator_seed=0,
            expected_slice=torch.tensor(
                [0.04780, -0.58508, -0.28968, 0.13094, 0.38533, 0.86582, 0.73544, 0.58218, 0.92315]
            ),
            num_inference_steps=2,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P64_E1_B1_M1(self):
        self._test_marigold_normals(
            generator_seed=0,
            expected_slice=torch.tensor(
                [-0.26170, 0.85460, 0.45221, 0.15963, 0.54384, -0.32731, 0.00334, -0.83391, -0.57067]
            ),
            num_inference_steps=1,
            processing_resolution=64,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P32_E3_B1_M1(self):
        self._test_marigold_normals(
            generator_seed=0,
            expected_slice=torch.tensor(
                [0.25150, -0.93332, -0.39775, 0.34287, 0.15370, -0.58052, 0.83557, 0.04513, -0.27762]
            ),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=3,
            ensembling_kwargs={"reduction": "mean"},
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P32_E4_B2_M1(self):
        self._test_marigold_normals(
            generator_seed=0,
            expected_slice=torch.tensor(
                [0.08004, -0.32468, -0.25072, 0.13662, 0.28124, -0.40264, 0.98766, 0.40109, -0.21820]
            ),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=4,
            ensembling_kwargs={"reduction": "mean"},
            batch_size=2,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P16_E1_B1_M0(self):
        self._test_marigold_normals(
            generator_seed=0,
            expected_slice=torch.tensor(
                [0.85842, 0.45535, -0.18574, 0.15936, -0.44240, 0.04431, 0.33110, -0.18396, -0.52408]
            ),
            num_inference_steps=1,
            processing_resolution=16,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=False,
        )

    def test_marigold_depth_dummy_no_num_inference_steps(self):
        with pytest.raises(ValueError, match="num_inference_steps"):
            self._test_marigold_normals(num_inference_steps=None, expected_slice=torch.tensor([0.0]))

    def test_marigold_depth_dummy_no_processing_resolution(self):
        with pytest.raises(ValueError, match="processing_resolution"):
            self._test_marigold_normals(processing_resolution=None, expected_slice=torch.tensor([0.0]))


class TestMarigoldNormalsPipelineMemory(MarigoldNormalsPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Marigold normals pipeline."""


@slow
@require_torch_accelerator
class TestMarigoldNormalsPipelineIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def _test_marigold_normals(
        self,
        is_fp16: bool = True,
        device: str = "cuda",
        generator_seed: int = 0,
        expected_slice: np.ndarray = None,
        model_id: str = "prs-eth/marigold-normals-lcm-v0-1",
        image_url: str = "https://marigoldmonodepth.github.io/images/einstein.jpg",
        atol: float = 1e-4,
        **pipe_kwargs,
    ):
        from_pretrained_kwargs = {}
        if is_fp16:
            from_pretrained_kwargs["variant"] = "fp16"
            from_pretrained_kwargs["torch_dtype"] = torch.float16

        pipe = MarigoldNormalsPipeline.from_pretrained(model_id, **from_pretrained_kwargs)
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(generator_seed)

        image = load_image(image_url)
        width, height = image.size

        prediction = pipe(image, generator=generator, **pipe_kwargs).prediction

        prediction_slice = prediction[0, -3:, -3:, -1].flatten()

        if pipe_kwargs.get("match_input_resolution", True):
            assert prediction.shape == (1, height, width, 3), "Unexpected output resolution"
        else:
            assert prediction.shape[0] == 1 and prediction.shape[3] == 3, "Unexpected output dimensions"
            assert max(prediction.shape[1:3]) == pipe_kwargs.get("processing_resolution", 768), (
                "Unexpected output resolution"
            )

        assert np.allclose(prediction_slice, expected_slice, atol=atol)

    def test_marigold_normals_einstein_f32_cpu_G0_S1_P32_E1_B1_M1(self):
        self._test_marigold_normals(
            is_fp16=False,
            device=torch_device,
            generator_seed=0,
            expected_slice=np.array([0.8971, 0.8971, 0.8971, 0.8971, 0.8971, 0.8971, 0.8971, 0.8971, 0.8971]),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_normals_einstein_f32_cuda_G0_S1_P768_E1_B1_M1(self):
        self._test_marigold_normals(
            is_fp16=False,
            device=torch_device,
            generator_seed=0,
            expected_slice=np.array([0.7980, 0.7952, 0.7914, 0.7931, 0.7871, 0.7816, 0.7844, 0.7710, 0.7601]),
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_normals_einstein_f16_cuda_G0_S1_P768_E1_B1_M1(self):
        self._test_marigold_normals(
            is_fp16=True,
            device=torch_device,
            generator_seed=0,
            expected_slice=np.array([0.7979, 0.7949, 0.7915, 0.7930, 0.7871, 0.7817, 0.7842, 0.7710, 0.7603]),
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_normals_einstein_f16_cuda_G2024_S1_P768_E1_B1_M1(self):
        self._test_marigold_normals(
            is_fp16=True,
            device=torch_device,
            generator_seed=2024,
            expected_slice=np.array([0.8428, 0.8428, 0.8433, 0.8369, 0.8325, 0.8315, 0.8271, 0.8135, 0.8057]),
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_normals_einstein_f16_cuda_G0_S2_P768_E1_B1_M1(self):
        self._test_marigold_normals(
            is_fp16=True,
            device=torch_device,
            generator_seed=0,
            expected_slice=np.array([0.7095, 0.7095, 0.7104, 0.7070, 0.7051, 0.7061, 0.7017, 0.6938, 0.6914]),
            num_inference_steps=2,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_normals_einstein_f16_cuda_G0_S1_P512_E1_B1_M1(self):
        self._test_marigold_normals(
            is_fp16=True,
            device=torch_device,
            generator_seed=0,
            expected_slice=np.array([0.7168, 0.7163, 0.7163, 0.7080, 0.7061, 0.7046, 0.7031, 0.7007, 0.6987]),
            num_inference_steps=1,
            processing_resolution=512,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_normals_einstein_f16_cuda_G0_S1_P768_E3_B1_M1(self):
        self._test_marigold_normals(
            is_fp16=True,
            device=torch_device,
            generator_seed=0,
            expected_slice=np.array([0.7114, 0.7124, 0.7144, 0.7085, 0.7070, 0.7080, 0.7051, 0.6958, 0.6924]),
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=3,
            ensembling_kwargs={"reduction": "mean"},
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_normals_einstein_f16_cuda_G0_S1_P768_E4_B2_M1(self):
        self._test_marigold_normals(
            is_fp16=True,
            device=torch_device,
            generator_seed=0,
            expected_slice=np.array([0.7412, 0.7441, 0.7490, 0.7383, 0.7388, 0.7437, 0.7329, 0.7271, 0.7300]),
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=4,
            ensembling_kwargs={"reduction": "mean"},
            batch_size=2,
            match_input_resolution=True,
        )

    def test_marigold_normals_einstein_f16_cuda_G0_S1_P512_E1_B1_M0(self):
        self._test_marigold_normals(
            is_fp16=True,
            device=torch_device,
            generator_seed=0,
            expected_slice=np.array([0.7188, 0.7144, 0.7134, 0.7178, 0.7207, 0.7222, 0.7231, 0.7041, 0.6987]),
            num_inference_steps=1,
            processing_resolution=512,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=False,
        )
