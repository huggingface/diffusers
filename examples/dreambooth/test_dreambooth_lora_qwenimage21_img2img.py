# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

import json
import logging
import os
import sys
import tempfile

import numpy as np
import safetensors
from PIL import Image

from diffusers.loaders.lora_base import LORA_ADAPTER_METADATA_KEY


sys.path.append("..")
from test_examples_utils import ExamplesTestsAccelerate, run_command  # noqa: E402


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class TestDreamBoothLoRAQwenImage21Img2Img(ExamplesTestsAccelerate):
    instance_prompt = "photo"
    pretrained_model_name_or_path = "hf-internal-testing/tiny-qwenimage21-pipe"
    script_path = "examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py"
    transformer_layer_type = "transformer_blocks.0.attn.to_k"
    # 256 rather than the 64 the text-to-image tests use: the vision-language processor upsamples images below its
    # minimum pixel count, and a condition image it resizes produces more vision tokens than the transformer has
    # slots for. 256 is the smallest size where the two line up.
    resolution = 256

    def _paired_dataset(self, directory):
        """Write a two-row dataset with a target image, a condition image and a caption."""
        from datasets import Dataset, Features, Value
        from datasets import Image as ImageFeature

        rng = np.random.default_rng(0)

        def image():
            return Image.fromarray(rng.integers(0, 255, (self.resolution, self.resolution, 3), dtype=np.uint8))

        rows = {
            "image": [image() for _ in range(2)],
            "cond_image": [image() for _ in range(2)],
            "caption": [self.instance_prompt] * 2,
        }
        dataset = Dataset.from_dict(
            rows,
            features=Features({"image": ImageFeature(), "cond_image": ImageFeature(), "caption": Value("string")}),
        )
        path = os.path.join(directory, "dataset")
        os.makedirs(path, exist_ok=True)
        dataset.to_parquet(os.path.join(path, "data.parquet"))
        return path

    def _base_args(self, dataset_dir, tmpdir):
        return f"""
            {self.script_path}
            --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
            --dataset_name {dataset_dir}
            --cond_image_column cond_image
            --caption_column caption
            --instance_prompt {self.instance_prompt}
            --resolution {self.resolution}
            --train_batch_size 1
            --gradient_accumulation_steps 1
            --max_train_steps 2
            --learning_rate 5.0e-04
            --lr_scheduler constant
            --lr_warmup_steps 0
            --output_dir {tmpdir}
            """

    def test_dreambooth_lora_qwenimage21_img2img(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = self._paired_dataset(tmpdir)
            run_command(self._launch_args + self._base_args(dataset_dir, tmpdir).split())

            # save_pretrained smoke test
            assert os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))

            # make sure the state_dict has the correct naming in the parameters.
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            assert is_lora

            # when not training the text encoder, all the parameters in the state dict should start
            # with `"transformer"` in their names.
            starts_with_transformer = all(key.startswith("transformer") for key in lora_state_dict.keys())
            assert starts_with_transformer

    def test_dreambooth_lora_qwenimage21_img2img_latent_caching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = self._paired_dataset(tmpdir)
            test_args = self._base_args(dataset_dir, tmpdir).split() + ["--cache_latents"]
            run_command(self._launch_args + test_args)

            assert os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            assert is_lora
            starts_with_transformer = all(key.startswith("transformer") for key in lora_state_dict.keys())
            assert starts_with_transformer

    def test_dreambooth_lora_qwenimage21_img2img_with_metadata(self):
        # Use a `lora_alpha` that is different from `rank`.
        lora_alpha = 8
        rank = 4
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = self._paired_dataset(tmpdir)
            test_args = self._base_args(dataset_dir, tmpdir).split() + [
                f"--lora_alpha={lora_alpha}",
                f"--rank={rank}",
            ]
            run_command(self._launch_args + test_args)

            state_dict_file = os.path.join(tmpdir, "pytorch_lora_weights.safetensors")
            assert os.path.isfile(state_dict_file)

            # Check if the metadata was properly serialized.
            with safetensors.torch.safe_open(state_dict_file, framework="pt", device="cpu") as f:
                metadata = f.metadata() or {}

            metadata.pop("format", None)
            raw = metadata.get(LORA_ADAPTER_METADATA_KEY)
            if raw:
                raw = json.loads(raw)

            loaded_lora_alpha = raw["transformer.lora_alpha"]
            assert loaded_lora_alpha == lora_alpha
            loaded_lora_rank = raw["transformer.r"]
            assert loaded_lora_rank == rank
