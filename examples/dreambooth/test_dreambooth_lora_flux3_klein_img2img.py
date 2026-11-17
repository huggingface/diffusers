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

import logging
import os
import sys
import tempfile

import safetensors


sys.path.append("..")
from test_examples_utils import ExamplesTestsAccelerate, run_command  # noqa: E402


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

# Tiny randomly-initialized FLUX.3 image modular repo used for fast CI. The env var is an escape hatch for
# running against a local build (see `integration-notes/make_tiny_flux3_repo.py`).
FLUX3_TINY_REPO = os.environ.get("FLUX3_TINY_REPO", "hf-internal-testing/tiny-flux3-image-modular")


class DreamBoothLoRAFlux3KleinImg2Img(ExamplesTestsAccelerate):
    # image-to-image training needs a paired conditioning image, so it reads a dataset with an image +
    # conditioning_image column rather than a plain folder
    dataset_name = "hf-internal-testing/fill10"
    image_column = "image"
    cond_image_column = "conditioning_image"
    caption_column = "text"
    instance_prompt = "dog"
    pretrained_model_name_or_path = FLUX3_TINY_REPO
    script_path = "examples/dreambooth/train_dreambooth_lora_flux3_klein_img2img.py"

    def test_dreambooth_lora_flux3_klein_img2img(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --dataset_name {self.dataset_name}
                --image_column {self.image_column}
                --cond_image_column {self.cond_image_column}
                --caption_column {self.caption_column}
                --instance_prompt {self.instance_prompt}
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --lr_scheduler constant
                --lr_warmup_steps 0
                --max_sequence_length 8
                --text_encoder_out_layers 1
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

            # make sure the state_dict has the correct naming in the parameters.
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)

            starts_with_transformer = all(key.startswith("transformer") for key in lora_state_dict.keys())
            self.assertTrue(starts_with_transformer)

            # the reference image is consumed by its own ("image_cond") mode blocks, so training an img2img adapter
            # with the default targets reaches both the generated-image and the reference mode blocks
            self.assertTrue(any("mode_blocks.image." in key for key in lora_state_dict.keys()))
            self.assertTrue(any("mode_blocks.image_cond." in key for key in lora_state_dict.keys()))

    def test_dreambooth_lora_flux3_klein_img2img_latent_caching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --dataset_name {self.dataset_name}
                --image_column {self.image_column}
                --cond_image_column {self.cond_image_column}
                --caption_column {self.caption_column}
                --instance_prompt {self.instance_prompt}
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --cache_latents
                --learning_rate 5.0e-04
                --lr_scheduler constant
                --lr_warmup_steps 0
                --max_sequence_length 8
                --text_encoder_out_layers 1
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)

            # the cached conditioning latents must still produce an adapter on the reference mode blocks
            self.assertTrue(any("image_cond" in key for key in lora_state_dict.keys()))
