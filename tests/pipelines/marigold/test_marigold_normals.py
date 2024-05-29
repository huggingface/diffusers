# Copyright 2024 Marigold authors, PRS ETH Zurich. All rights reserved.
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
# Marigold project website: https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------
import gc
import random
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    AutoencoderTiny,
    LCMScheduler,
    MarigoldNormalsPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    require_torch_gpu,
    slow,
)

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class MarigoldNormalsPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = MarigoldNormalsPipeline
    params = frozenset(["image"])
    batch_params = frozenset(["image"])
    image_params = frozenset(["image"])
    image_latents_params = frozenset(["latents"])
    callback_cfg_params = frozenset([])
    test_xformers_attention = False
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "output_type",
        ]
    )

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

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "prediction_type": "normals",
            "use_full_z_range": True,
        }
        return components

    def get_dummy_tiny_autoencoder(self):
        return AutoencoderTiny(in_channels=3, out_channels=3, latent_channels=4)

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image / 2 + 0.5
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "image": image,
            "num_inference_steps": 1,
            "processing_resolution": 0,
            "generator": generator,
            "output_type": "np",
        }
        return inputs

    def _test_marigold_normals(
        self,
        generator_seed: int = 0,
        expected_slice: np.ndarray = None,
        atol: float = 1e-4,
        **pipe_kwargs,
    ):
        device = "cpu"
        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        pipe_inputs = self.get_dummy_inputs(device, seed=generator_seed)
        pipe_inputs.update(**pipe_kwargs)

        prediction = pipe(**pipe_inputs).prediction

        prediction_slice = prediction[0, -3:, -3:, -1].flatten()

        if pipe_inputs.get("match_input_resolution", True):
            self.assertEqual(prediction.shape, (1, 32, 32, 3), "Unexpected output resolution")
        else:
            self.assertTrue(prediction.shape[0] == 1 and prediction.shape[3] == 3, "Unexpected output dimensions")
            self.assertEqual(
                max(prediction.shape[1:3]),
                pipe_inputs.get("processing_resolution", 768),
                "Unexpected output resolution",
            )

        self.assertTrue(np.allclose(prediction_slice, expected_slice, atol=atol))

    def test_marigold_depth_dummy_defaults(self):
        self._test_marigold_normals(
            expected_slice=np.array([0.0967, 0.5234, 0.1448, -0.3155, -0.2550, -0.5578, 0.6854, 0.5657, -0.1263]),
        )

    def test_marigold_depth_dummy_G0_S1_P32_E1_B1_M1(self):
        self._test_marigold_normals(
            generator_seed=0,
            expected_slice=np.array([0.0967, 0.5234, 0.1448, -0.3155, -0.2550, -0.5578, 0.6854, 0.5657, -0.1263]),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P16_E1_B1_M1(self):
        self._test_marigold_normals(
            generator_seed=0,
            expected_slice=np.array([-0.4128, -0.5918, -0.6540, 0.2446, -0.2687, -0.4607, 0.2935, -0.0483, -0.2086]),
            num_inference_steps=1,
            processing_resolution=16,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G2024_S1_P32_E1_B1_M1(self):
        self._test_marigold_normals(
            generator_seed=2024,
            expected_slice=np.array([0.5731, -0.7631, -0.0199, 0.1609, -0.4628, -0.7044, 0.5761, -0.3471, -0.4498]),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S2_P32_E1_B1_M1(self):
        self._test_marigold_normals(
            generator_seed=0,
            expected_slice=np.array([0.1017, -0.6823, -0.2533, 0.1988, 0.3389, 0.8478, 0.7757, 0.5220, 0.8668]),
            num_inference_steps=2,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P64_E1_B1_M1(self):
        self._test_marigold_normals(
            generator_seed=0,
            expected_slice=np.array([-0.2391, 0.7969, 0.6224, 0.0698, 0.5669, -0.2167, -0.1362, -0.8945, -0.5501]),
            num_inference_steps=1,
            processing_resolution=64,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P32_E3_B1_M1(self):
        self._test_marigold_normals(
            generator_seed=0,
            expected_slice=np.array([0.3826, -0.9634, -0.3835, 0.3514, 0.0691, -0.6182, 0.8709, 0.1590, -0.2181]),
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
            expected_slice=np.array([0.2500, -0.3928, -0.2415, 0.1133, 0.2357, -0.4223, 0.9967, 0.4859, -0.1282]),
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
            expected_slice=np.array([0.9588, 0.3326, -0.0825, -0.0994, -0.3534, -0.4302, 0.3562, 0.4421, -0.2086]),
            num_inference_steps=1,
            processing_resolution=16,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=False,
        )

    def test_marigold_depth_dummy_no_num_inference_steps(self):
        with self.assertRaises(ValueError) as e:
            self._test_marigold_normals(
                num_inference_steps=None,
                expected_slice=np.array([0.0]),
            )
            self.assertIn("num_inference_steps", str(e))

    def test_marigold_depth_dummy_no_processing_resolution(self):
        with self.assertRaises(ValueError) as e:
            self._test_marigold_normals(
                processing_resolution=None,
                expected_slice=np.array([0.0]),
            )
            self.assertIn("processing_resolution", str(e))


@slow
@require_torch_gpu
class MarigoldNormalsPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

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
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(generator_seed)

        image = load_image(image_url)
        width, height = image.size

        prediction = pipe(image, generator=generator, **pipe_kwargs).prediction

        prediction_slice = prediction[0, -3:, -3:, -1].flatten()

        if pipe_kwargs.get("match_input_resolution", True):
            self.assertEqual(prediction.shape, (1, height, width, 3), "Unexpected output resolution")
        else:
            self.assertTrue(prediction.shape[0] == 1 and prediction.shape[3] == 3, "Unexpected output dimensions")
            self.assertEqual(
                max(prediction.shape[1:3]),
                pipe_kwargs.get("processing_resolution", 768),
                "Unexpected output resolution",
            )

        self.assertTrue(np.allclose(prediction_slice, expected_slice, atol=atol))

    def test_marigold_normals_einstein_f32_cpu_G0_S1_P32_E1_B1_M1(self):
        self._test_marigold_normals(
            is_fp16=False,
            device="cpu",
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
            device="cuda",
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
            device="cuda",
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
            device="cuda",
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
            device="cuda",
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
            device="cuda",
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
            device="cuda",
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
            device="cuda",
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
            device="cuda",
            generator_seed=0,
            expected_slice=np.array([0.7188, 0.7144, 0.7134, 0.7178, 0.7207, 0.7222, 0.7231, 0.7041, 0.6987]),
            num_inference_steps=1,
            processing_resolution=512,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=False,
        )
