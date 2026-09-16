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
    DDIMScheduler,
    MarigoldIntrinsicsPipeline,
    UNet2DConditionModel,
)

from ...testing_utils import (
    Expectations,
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


class MarigoldIntrinsicsPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = MarigoldIntrinsicsPipeline
    required_input_params_in_call_signature = frozenset(["image"])
    batch_input_params = frozenset(["image"])
    # Marigold predicts intrinsic image maps and takes no prompt: it exposes neither `num_images_per_prompt` nor
    # `num_videos_per_prompt`.
    optional_input_params = frozenset(["num_inference_steps", "generator", "latents", "output_type", "return_dict"])
    # The pipeline returns one map per target, all stacked along the batch axis.
    output_shape = (3, 32, 32)

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

        return {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "prediction_type": "intrinsics",
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


class TestMarigoldIntrinsicsPipeline(MarigoldIntrinsicsPipelineTesterConfig, PipelineTesterMixin):
    """The pipeline returns `n_targets` predictions per input image, all stacked along the batch axis, which is why
    the two batch tests below assert against `batch_size * n_targets` instead of `batch_size`."""

    def test_inference_batch_consistent(self, batch_sizes=[2], batch_generator=True):
        pipe = self.get_pipeline().to(torch_device)

        for batch_size in batch_sizes:
            inputs = self.get_dummy_inputs()
            for name in self.batch_input_params:
                if name in inputs:
                    inputs[name] = batch_size * [inputs[name]]
            if batch_generator and "generator" in inputs:
                inputs["generator"] = [self.get_generator(i) for i in range(batch_size)]

            output = pipe(**inputs)
            assert len(output[0]) == batch_size * pipe.n_targets

    def test_inference_batch_single_identical(self, batch_size=2, expected_max_diff=1e-4):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        batched_inputs = dict(inputs)
        for name in self.batch_input_params:
            if name in inputs:
                batched_inputs[name] = batch_size * [inputs[name]]
        batched_inputs["generator"] = [self.get_generator(i) for i in range(batch_size)]

        output = pipe(**inputs)
        output_batch = pipe(**batched_inputs)

        assert output_batch[0].shape[0] == batch_size * output[0].shape[0]
        assert_tensors_close(
            output_batch[0][0], output[0][0], atol=expected_max_diff, msg="Batched output differs from single."
        )

    def _test_marigold_intrinsics(
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

        prediction = pipe(**pipe_inputs).prediction  # [n_targets,3,H,W] for `output_type="pt"`

        prediction_slice = prediction[0, -1, -3:, -3:].flatten()

        if pipe_inputs.get("match_input_resolution", True):
            assert prediction.shape == (pipe.n_targets, *self.output_shape), "Unexpected output resolution"
        else:
            assert prediction.shape[0] == pipe.n_targets and prediction.shape[1] == 3, "Unexpected output dimensions"
            assert max(prediction.shape[2:4]) == pipe_inputs.get("processing_resolution", 768), (
                "Unexpected output resolution"
            )

        assert_tensors_close(prediction_slice, expected_slice, atol=atol)

    def test_marigold_depth_dummy_defaults(self):
        self._test_marigold_intrinsics(
            expected_slice=torch.tensor(
                [0.6423, 0.40664, 0.41185, 0.65832, 0.63935, 0.43971, 0.51786, 0.55216, 0.47683]
            ),
        )

    def test_marigold_depth_dummy_G0_S1_P32_E1_B1_M1(self):
        self._test_marigold_intrinsics(
            generator_seed=0,
            expected_slice=torch.tensor(
                [0.6423, 0.40664, 0.41185, 0.65832, 0.63935, 0.43971, 0.51786, 0.55216, 0.47683]
            ),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P16_E1_B1_M1(self):
        self._test_marigold_intrinsics(
            generator_seed=0,
            expected_slice=torch.tensor(
                [0.53132, 0.44487, 0.40164, 0.5326, 0.49073, 0.46979, 0.53324, 0.51366, 0.50387]
            ),
            num_inference_steps=1,
            processing_resolution=16,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G2024_S1_P32_E1_B1_M1(self):
        self._test_marigold_intrinsics(
            generator_seed=2024,
            expected_slice=torch.tensor(
                [0.40250, 0.39464, 0.51378, 0.41603, 0.40150, 0.58531, 0.43581, 0.47833, 0.48946]
            ),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S2_P32_E1_B1_M1(self):
        self._test_marigold_intrinsics(
            generator_seed=0,
            expected_slice=torch.tensor(
                [0.52219, 0.45487, 0.42093, 0.58746, 0.63236, 0.38438, 0.52289, 0.54885, 0.48601]
            ),
            num_inference_steps=2,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P64_E1_B1_M1(self):
        self._test_marigold_intrinsics(
            generator_seed=0,
            expected_slice=torch.tensor(
                [0.55574, 0.43518, 0.48871, 0.56418, 0.63882, 0.56345, 0.47897, 0.52932, 0.49240]
            ),
            num_inference_steps=1,
            processing_resolution=64,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P32_E3_B1_M1(self):
        self._test_marigold_intrinsics(
            generator_seed=0,
            expected_slice=torch.tensor(
                [0.57244, 0.49813, 0.54442, 0.57727, 0.52388, 0.52545, 0.56492, 0.56334, 0.48579]
            ),
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
            expected_slice=torch.tensor(
                [0.62939, 0.55744, 0.53417, 0.61068, 0.57141, 0.53967, 0.52955, 0.55467, 0.48751]
            ),
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
            expected_slice=torch.tensor(
                [0.63543, 0.68147, 0.48780, 0.46715, 0.58511, 0.36761, 0.58482, 0.54309, 0.50388]
            ),
            num_inference_steps=1,
            processing_resolution=16,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=False,
        )

    def test_marigold_depth_dummy_no_num_inference_steps(self):
        with pytest.raises(ValueError, match="num_inference_steps"):
            self._test_marigold_intrinsics(num_inference_steps=None, expected_slice=torch.tensor([0.0]))

    def test_marigold_depth_dummy_no_processing_resolution(self):
        with pytest.raises(ValueError, match="processing_resolution"):
            self._test_marigold_intrinsics(processing_resolution=None, expected_slice=torch.tensor([0.0]))


class TestMarigoldIntrinsicsPipelineMemory(MarigoldIntrinsicsPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for Marigold intrinsics."""


@slow
@require_torch_accelerator
class TestMarigoldIntrinsicsPipelineIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
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
            assert prediction.shape == (2, height, width, 3), "Unexpected output resolution"
        else:
            assert prediction.shape[0] == 2 and prediction.shape[3] == 3, "Unexpected output dimensions"
            assert max(prediction.shape[1:3]) == pipe_kwargs.get("processing_resolution", 768), (
                "Unexpected output resolution"
            )

        assert np.allclose(prediction_slice, expected_slice, atol=atol), f"{prediction_slice}"

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
        expected_slices = Expectations(
            {
                ("xpu", 5): np.array(
                    [0.62477, 0.62261, 0.62027, 0.62281, 0.62241, 0.62271, 0.62184, 0.62479, 0.63226]
                ),
                ("xpu", 3): np.array(
                    [0.62127, 0.61906, 0.61687, 0.61946, 0.61903, 0.61961, 0.61808, 0.62099, 0.62894]
                ),
                (None, None): np.array(
                    [0.62127, 0.61906, 0.61687, 0.61946, 0.61903, 0.61961, 0.61808, 0.62099, 0.62894]
                ),
            }
        )
        self._test_marigold_intrinsics(
            is_fp16=False,
            device=torch_device,
            generator_seed=0,
            expected_slice=expected_slices.get_expectation(),
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_intrinsics_einstein_f16_accelerator_G0_S1_P768_E1_B1_M1(self):
        expected_slices = Expectations(
            {
                ("xpu", 5): np.array(
                    [0.62451, 0.62256, 0.62012, 0.62256, 0.62207, 0.62256, 0.62158, 0.62451, 0.63184]
                ),
                ("xpu", 3): np.array(
                    [0.62109, 0.61914, 0.61719, 0.61963, 0.61914, 0.61963, 0.61816, 0.62109, 0.62891]
                ),
                (None, None): np.array(
                    [0.62109, 0.61914, 0.61719, 0.61963, 0.61914, 0.61963, 0.61816, 0.62109, 0.62891]
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
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_intrinsics_einstein_f16_accelerator_G2024_S1_P768_E1_B1_M1(self):
        expected_slices = Expectations(
            {
                ("xpu", 5): np.array(
                    [0.63330, 0.63135, 0.62793, 0.63184, 0.63135, 0.63135, 0.63037, 0.63379, 0.64160]
                ),
                ("xpu", 3): np.array([0.64111, 0.63916, 0.63623, 0.63965, 0.63916, 0.63965, 0.6377, 0.64062, 0.64941]),
                (None, None): np.array(
                    [0.64111, 0.63916, 0.63623, 0.63965, 0.63916, 0.63965, 0.6377, 0.64062, 0.64941]
                ),
            }
        )
        self._test_marigold_intrinsics(
            is_fp16=True,
            device=torch_device,
            generator_seed=2024,
            expected_slice=expected_slices.get_expectation(),
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_intrinsics_einstein_f16_accelerator_G0_S2_P768_E1_B1_M1(self):
        expected_slices = Expectations(
            {
                ("xpu", 5): np.array([0.61475, 0.61328, 0.6123, 0.61426, 0.61328, 0.61475, 0.61475, 0.61621, 0.62402]),
                ("xpu", 3): np.array(
                    [0.60254, 0.60059, 0.59961, 0.60156, 0.60107, 0.60205, 0.60254, 0.60449, 0.61133]
                ),
                (None, None): np.array(
                    [0.60254, 0.60059, 0.59961, 0.60156, 0.60107, 0.60205, 0.60254, 0.60449, 0.61133]
                ),
            }
        )
        self._test_marigold_intrinsics(
            is_fp16=True,
            device=torch_device,
            generator_seed=0,
            expected_slice=expected_slices.get_expectation(),
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
                ("xpu", 5): np.array(
                    [
                        0.62354,
                        0.62158,
                        0.61963,
                        0.62207,
                        0.62158,
                        0.62207,
                        0.62109,
                        0.62354,
                        0.63135,
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
                ("xpu", 5): np.array(
                    [
                        0.62207,
                        0.62012,
                        0.61865,
                        0.62061,
                        0.62061,
                        0.62158,
                        0.62012,
                        0.62305,
                        0.63086,
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
