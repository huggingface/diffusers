# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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


import torch

from diffusers import (
    AutoencoderKL,
)

from ..testing_utils import (
    enable_full_determinism,
    load_hf_numpy,
    numpy_cosine_similarity_distance,
    torch_device,
)
from .single_file_testing_utils import SingleFileModelTesterMixin


enable_full_determinism()


class TestAutoencoderKLSingleFile(SingleFileModelTesterMixin):
    model_class = AutoencoderKL
    ckpt_path = (
        "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
    )
    repo_id = "stabilityai/sd-vae-ft-mse"
    main_input_name = "sample"
    base_precision = 1e-2

    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def get_sd_image(self, seed=0, shape=(4, 3, 512, 512), fp16=False):
        dtype = torch.float16 if fp16 else torch.float32
        image = torch.from_numpy(load_hf_numpy(self.get_file_format(seed, shape))).to(torch_device).to(dtype)
        return image

    def test_single_file_inference_same_as_pretrained(self):
        model_1 = self.model_class.from_pretrained(self.repo_id).to(torch_device)
        model_2 = self.model_class.from_single_file(self.ckpt_path, config=self.repo_id).to(torch_device)

        image = self.get_sd_image(33)

        generator = torch.Generator(torch_device)

        with torch.no_grad():
            sample_1 = model_1(image, generator=generator.manual_seed(0)).sample
            sample_2 = model_2(image, generator=generator.manual_seed(0)).sample

        assert sample_1.shape == sample_2.shape

        output_slice_1 = sample_1.flatten().float().cpu()
        output_slice_2 = sample_2.flatten().float().cpu()

        assert numpy_cosine_similarity_distance(output_slice_1, output_slice_2) < 1e-4

    def test_single_file_arguments(self):
        model_default = self.model_class.from_single_file(self.ckpt_path, config=self.repo_id)

        assert model_default.config.scaling_factor == 0.18215
        assert model_default.config.sample_size == 256
        assert model_default.dtype == torch.float32

        scaling_factor = 2.0
        sample_size = 512
        torch_dtype = torch.float16

        model = self.model_class.from_single_file(
            self.ckpt_path,
            config=self.repo_id,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            torch_dtype=torch_dtype,
        )
        assert model.config.scaling_factor == scaling_factor
        assert model.config.sample_size == sample_size
        assert model.dtype == torch_dtype
