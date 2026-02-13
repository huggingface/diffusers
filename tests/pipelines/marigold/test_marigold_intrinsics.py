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
import random
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    AutoencoderTiny,
    DDIMScheduler,
    MarigoldIntrinsicsPipeline,
    UNet2DConditionModel,
)

from ...testing_utils import (
    Expectations,
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    load_image,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..test_pipelines_common import PipelineTesterMixin, to_np


enable_full_determinism()


class MarigoldIntrinsicsPipelineTesterMixin(PipelineTesterMixin):
    def _test_inference_batch_single_identical(
        self,
        batch_size=2,
        expected_max_diff=1e-4,
        additional_params_copy_to_batched_inputs=["num_inference_steps"],
    ):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for components in pipe.components.values():
            if hasattr(components, "set_default_attn_processor"):
                components.set_default_attn_processor()

        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs(torch_device)
        # Reset generator in case it is has been used in self.get_dummy_inputs
        inputs["generator"] = self.get_generator(0)

        logger = diffusers.logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        # batchify inputs
        batched_inputs = {}
        batched_inputs.update(inputs)

        for name in self.batch_params:
            if name not in inputs:
                continue

            value = inputs[name]
            if name == "prompt":
                len_prompt = len(value)
                batched_inputs[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]
                batched_inputs[name][-1] = 100 * "very long"

            else:
                batched_inputs[name] = batch_size * [value]

        if "generator" in inputs:
            batched_inputs["generator"] = [self.get_generator(i) for i in range(batch_size)]

        if "batch_size" in inputs:
            batched_inputs["batch_size"] = batch_size

        for arg in additional_params_copy_to_batched_inputs:
            batched_inputs[arg] = inputs[arg]

        output = pipe(**inputs)
        output_batch = pipe(**batched_inputs)

        assert output_batch[0].shape[0] == batch_size * output[0].shape[0]  # only changed here

        max_diff = np.abs(to_np(output_batch[0][0]) - to_np(output[0][0])).max()
        assert max_diff < expected_max_diff

    def _test_inference_batch_consistent(
        self, batch_sizes=[2], additional_params_copy_to_batched_inputs=["num_inference_steps"], batch_generator=True
    ):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["generator"] = self.get_generator(0)

        logger = diffusers.logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        # prepare batched inputs
        batched_inputs = []
        for batch_size in batch_sizes:
            batched_input = {}
            batched_input.update(inputs)

            for name in self.batch_params:
                if name not in inputs:
                    continue

                value = inputs[name]
                if name == "prompt":
                    len_prompt = len(value)
                    # make unequal batch sizes
                    batched_input[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]

                    # make last batch super long
                    batched_input[name][-1] = 100 * "very long"

                else:
                    batched_input[name] = batch_size * [value]

            if batch_generator and "generator" in inputs:
                batched_input["generator"] = [self.get_generator(i) for i in range(batch_size)]

            if "batch_size" in inputs:
                batched_input["batch_size"] = batch_size

            batched_inputs.append(batched_input)

        logger.setLevel(level=diffusers.logging.WARNING)
        for batch_size, batched_input in zip(batch_sizes, batched_inputs):
            output = pipe(**batched_input)
            assert len(output[0]) == batch_size * pipe.n_targets  # only changed here


class MarigoldIntrinsicsPipelineFastTests(MarigoldIntrinsicsPipelineTesterMixin, unittest.TestCase):
    pipeline_class = MarigoldIntrinsicsPipeline
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
            in_channels=12,
            out_channels=8,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        torch.manual_seed(0)
        scheduler = DDIMScheduler(
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
            "prediction_type": "intrinsics",
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

    def _test_marigold_intrinsics(
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
            self.assertEqual(prediction.shape, (2, 32, 32, 3), "Unexpected output resolution")
        else:
            self.assertTrue(prediction.shape[0] == 2 and prediction.shape[3] == 3, "Unexpected output dimensions")
            self.assertEqual(
                max(prediction.shape[1:3]),
                pipe_inputs.get("processing_resolution", 768),
                "Unexpected output resolution",
            )

        np.set_printoptions(precision=5, suppress=True)
        msg = f"{prediction_slice}"
        self.assertTrue(np.allclose(prediction_slice, expected_slice, atol=atol), msg)
        # self.assertTrue(np.allclose(prediction_slice, expected_slice, atol=atol))

    def test_marigold_depth_dummy_defaults(self):
        self._test_marigold_intrinsics(
            expected_slice=np.array([0.6423, 0.40664, 0.41185, 0.65832, 0.63935, 0.43971, 0.51786, 0.55216, 0.47683]),
        )

    def test_marigold_depth_dummy_G0_S1_P32_E1_B1_M1(self):
        self._test_marigold_intrinsics(
            generator_seed=0,
            expected_slice=np.array([0.6423, 0.40664, 0.41185, 0.65832, 0.63935, 0.43971, 0.51786, 0.55216, 0.47683]),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P16_E1_B1_M1(self):
        self._test_marigold_intrinsics(
            generator_seed=0,
            expected_slice=np.array([0.53132, 0.44487, 0.40164, 0.5326, 0.49073, 0.46979, 0.53324, 0.51366, 0.50387]),
            num_inference_steps=1,
            processing_resolution=16,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G2024_S1_P32_E1_B1_M1(self):
        self._test_marigold_intrinsics(
            generator_seed=2024,
            expected_slice=np.array([0.40257, 0.39468, 0.51373, 0.4161, 0.40162, 0.58535, 0.43581, 0.47834, 0.48951]),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S2_P32_E1_B1_M1(self):
        self._test_marigold_intrinsics(
            generator_seed=0,
            expected_slice=np.array([0.49636, 0.4518, 0.42722, 0.59044, 0.6362, 0.39011, 0.53522, 0.55153, 0.48699]),
            num_inference_steps=2,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P64_E1_B1_M1(self):
        self._test_marigold_intrinsics(
            generator_seed=0,
            expected_slice=np.array([0.55547, 0.43511, 0.4887, 0.56399, 0.63867, 0.56337, 0.47889, 0.52925, 0.49235]),
            num_inference_steps=1,
            processing_resolution=64,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P32_E3_B1_M1(self):
        self._test_marigold_intrinsics(
            generator_seed=0,
            expected_slice=np.array([0.57249, 0.49824, 0.54438, 0.57733, 0.52404, 0.5255, 0.56493, 0.56336, 0.48579]),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=3,
            ensembling_kwargs={"reduction": "mean"},
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P32_E4_B2_M1(self):
        self._test_marigold_intrinsics(
            generator_seed=0,
            expected_slice=np.array([0.6294, 0.5575, 0.53414, 0.61077, 0.57156, 0.53974, 0.52956, 0.55467, 0.48751]),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=4,
            ensembling_kwargs={"reduction": "mean"},
            batch_size=2,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P16_E1_B1_M0(self):
        self._test_marigold_intrinsics(
            generator_seed=0,
            expected_slice=np.array([0.63511, 0.68137, 0.48783, 0.46689, 0.58505, 0.36757, 0.58465, 0.54302, 0.50387]),
            num_inference_steps=1,
            processing_resolution=16,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=False,
        )

    def test_marigold_depth_dummy_no_num_inference_steps(self):
        with self.assertRaises(ValueError) as e:
            self._test_marigold_intrinsics(
                num_inference_steps=None,
                expected_slice=np.array([0.0]),
            )
            self.assertIn("num_inference_steps", str(e))

    def test_marigold_depth_dummy_no_processing_resolution(self):
        with self.assertRaises(ValueError) as e:
            self._test_marigold_intrinsics(
                processing_resolution=None,
                expected_slice=np.array([0.0]),
            )
            self.assertIn("processing_resolution", str(e))


@slow
@require_torch_accelerator
class MarigoldIntrinsicsPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def _test_marigold_intrinsics(
        self,
        is_fp16: bool = True,
        device: str = "cuda",
        generator_seed: int = 0,
        expected_slice: np.ndarray = None,
        model_id: str = "prs-eth/marigold-iid-appearance-v1-1",
        image_url: str = "https://marigoldmonodepth.github.io/images/einstein.jpg",
        atol: float = 1e-3,
        **pipe_kwargs,
    ):
        from_pretrained_kwargs = {}
        if is_fp16:
            from_pretrained_kwargs["variant"] = "fp16"
            from_pretrained_kwargs["torch_dtype"] = torch.float16

        pipe = MarigoldIntrinsicsPipeline.from_pretrained(model_id, **from_pretrained_kwargs)
        if device in ["cuda", "xpu"]:
            pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(generator_seed)

        image = load_image(image_url)
        width, height = image.size

        prediction = pipe(image, generator=generator, **pipe_kwargs).prediction

        prediction_slice = prediction[0, -3:, -3:, -1].flatten()

        if pipe_kwargs.get("match_input_resolution", True):
            self.assertEqual(prediction.shape, (2, height, width, 3), "Unexpected output resolution")
        else:
            self.assertTrue(prediction.shape[0] == 2 and prediction.shape[3] == 3, "Unexpected output dimensions")
            self.assertEqual(
                max(prediction.shape[1:3]),
                pipe_kwargs.get("processing_resolution", 768),
                "Unexpected output resolution",
            )

        msg = f"{prediction_slice}"
        self.assertTrue(np.allclose(prediction_slice, expected_slice, atol=atol), msg)
        # self.assertTrue(np.allclose(prediction_slice, expected_slice, atol=atol))

    def test_marigold_intrinsics_einstein_f32_cpu_G0_S1_P32_E1_B1_M1(self):
        self._test_marigold_intrinsics(
            is_fp16=False,
            device="cpu",
            generator_seed=0,
            expected_slice=np.array([0.9162, 0.9162, 0.9162, 0.9162, 0.9162, 0.9162, 0.9162, 0.9162, 0.9162]),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_intrinsics_einstein_f32_accelerator_G0_S1_P768_E1_B1_M1(self):
        self._test_marigold_intrinsics(
            is_fp16=False,
            device=torch_device,
            generator_seed=0,
            expected_slice=np.array([0.62127, 0.61906, 0.61687, 0.61946, 0.61903, 0.61961, 0.61808, 0.62099, 0.62894]),
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_intrinsics_einstein_f16_accelerator_G0_S1_P768_E1_B1_M1(self):
        self._test_marigold_intrinsics(
            is_fp16=True,
            device=torch_device,
            generator_seed=0,
            expected_slice=np.array([0.62109, 0.61914, 0.61719, 0.61963, 0.61914, 0.61963, 0.61816, 0.62109, 0.62891]),
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_intrinsics_einstein_f16_accelerator_G2024_S1_P768_E1_B1_M1(self):
        self._test_marigold_intrinsics(
            is_fp16=True,
            device=torch_device,
            generator_seed=2024,
            expected_slice=np.array([0.64111, 0.63916, 0.63623, 0.63965, 0.63916, 0.63965, 0.6377, 0.64062, 0.64941]),
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_intrinsics_einstein_f16_accelerator_G0_S2_P768_E1_B1_M1(self):
        self._test_marigold_intrinsics(
            is_fp16=True,
            device=torch_device,
            generator_seed=0,
            expected_slice=np.array([0.60254, 0.60059, 0.59961, 0.60156, 0.60107, 0.60205, 0.60254, 0.60449, 0.61133]),
            num_inference_steps=2,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_intrinsics_einstein_f16_accelerator_G0_S1_P512_E1_B1_M1(self):
        self._test_marigold_intrinsics(
            is_fp16=True,
            device=torch_device,
            generator_seed=0,
            expected_slice=np.array([0.64551, 0.64453, 0.64404, 0.64502, 0.64844, 0.65039, 0.64502, 0.65039, 0.65332]),
            num_inference_steps=1,
            processing_resolution=512,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_intrinsics_einstein_f16_accelerator_G0_S1_P768_E3_B1_M1(self):
        expected_slices = Expectations(
            {
                ("xpu", 3): np.array(
                    [
                        0.62655,
                        0.62477,
                        0.62161,
                        0.62452,
                        0.62454,
                        0.62454,
                        0.62255,
                        0.62647,
                        0.63379,
                    ]
                ),
                ("cuda", 7): np.array(
                    [
                        0.61572,
                        0.1377,
                        0.61182,
                        0.61426,
                        0.61377,
                        0.61426,
                        0.61279,
                        0.61572,
                        0.62354,
                    ]
                ),
            }
        )
        self._test_marigold_intrinsics(
            is_fp16=True,
            device=torch_device,
            generator_seed=0,
            expected_slice=expected_slices.get_expectation(),
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=3,
            ensembling_kwargs={"reduction": "mean"},
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_intrinsics_einstein_f16_accelerator_G0_S1_P768_E4_B2_M1(self):
        expected_slices = Expectations(
            {
                ("xpu", 3): np.array(
                    [
                        0.62988,
                        0.62792,
                        0.62548,
                        0.62841,
                        0.62792,
                        0.62792,
                        0.62646,
                        0.62939,
                        0.63721,
                    ]
                ),
                ("cuda", 7): np.array(
                    [
                        0.61914,
                        0.6167,
                        0.61475,
                        0.61719,
                        0.61719,
                        0.61768,
                        0.61572,
                        0.61914,
                        0.62695,
                    ]
                ),
            }
        )
        self._test_marigold_intrinsics(
            is_fp16=True,
            device=torch_device,
            generator_seed=0,
            expected_slice=expected_slices.get_expectation(),
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=4,
            ensembling_kwargs={"reduction": "mean"},
            batch_size=2,
            match_input_resolution=True,
        )

    def test_marigold_intrinsics_einstein_f16_accelerator_G0_S1_P512_E1_B1_M0(self):
        self._test_marigold_intrinsics(
            is_fp16=True,
            device=torch_device,
            generator_seed=0,
            expected_slice=np.array([0.65332, 0.64697, 0.64648, 0.64844, 0.64697, 0.64111, 0.64941, 0.64209, 0.65332]),
            num_inference_steps=1,
            processing_resolution=512,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=False,
        )
