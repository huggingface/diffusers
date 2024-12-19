# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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


class DreamBoothLoRASANA(ExamplesTestsAccelerate):
    instance_data_dir = "docs/source/en/imgs"
    pretrained_model_name_or_path = "hf-internal-testing/tiny-sana-pipe"
    script_path = "examples/dreambooth/train_dreambooth_lora_sana.py"
    transformer_layer_type = "transformer_blocks.0.attn1.to_k"

    def test_dreambooth_lora_sana(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --instance_data_dir {self.instance_data_dir}
                --resolution 32
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --max_sequence_length 16
                """.split()

            test_args.extend(["--instance_prompt", ""])
            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

            # make sure the state_dict has the correct naming in the parameters.
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)

            # when not training the text encoder, all the parameters in the state dict should start
            # with `"transformer"` in their names.
            starts_with_transformer = all(key.startswith("transformer") for key in lora_state_dict.keys())
            self.assertTrue(starts_with_transformer)

    def test_dreambooth_lora_latent_caching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --instance_data_dir {self.instance_data_dir}
                --resolution 32
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --cache_latents
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --max_sequence_length 16
                """.split()

            test_args.extend(["--instance_prompt", ""])
            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

            # make sure the state_dict has the correct naming in the parameters.
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)

            # when not training the text encoder, all the parameters in the state dict should start
            # with `"transformer"` in their names.
            starts_with_transformer = all(key.startswith("transformer") for key in lora_state_dict.keys())
            self.assertTrue(starts_with_transformer)

    def test_dreambooth_lora_layers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --instance_data_dir {self.instance_data_dir}
                --resolution 32
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --cache_latents
                --learning_rate 5.0e-04
                --scale_lr
                --lora_layers {self.transformer_layer_type}
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --max_sequence_length 16
                """.split()

            test_args.extend(["--instance_prompt", ""])
            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

            # make sure the state_dict has the correct naming in the parameters.
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)

            # when not training the text encoder, all the parameters in the state dict should start
            # with `"transformer"` in their names. In this test, we only params of
            # `self.transformer_layer_type` should be in the state dict.
            starts_with_transformer = all(self.transformer_layer_type in key for key in lora_state_dict)
            self.assertTrue(starts_with_transformer)

    def test_dreambooth_lora_sana_checkpointing_checkpoints_total_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            {self.script_path}
            --pretrained_model_name_or_path={self.pretrained_model_name_or_path}
            --instance_data_dir={self.instance_data_dir}
            --output_dir={tmpdir}
            --resolution=32
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=6
            --checkpoints_total_limit=2
            --checkpointing_steps=2
            --max_sequence_length 16
            """.split()

            test_args.extend(["--instance_prompt", ""])
            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_dreambooth_lora_sana_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            {self.script_path}
            --pretrained_model_name_or_path={self.pretrained_model_name_or_path}
            --instance_data_dir={self.instance_data_dir}
            --output_dir={tmpdir}
            --resolution=32
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=4
            --checkpointing_steps=2
            --max_sequence_length 166
            """.split()

            test_args.extend(["--instance_prompt", ""])
            run_command(self._launch_args + test_args)

            self.assertEqual({x for x in os.listdir(tmpdir) if "checkpoint" in x}, {"checkpoint-2", "checkpoint-4"})

            resume_run_args = f"""
            {self.script_path}
            --pretrained_model_name_or_path={self.pretrained_model_name_or_path}
            --instance_data_dir={self.instance_data_dir}
            --output_dir={tmpdir}
            --resolution=32
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=8
            --checkpointing_steps=2
            --resume_from_checkpoint=checkpoint-4
            --checkpoints_total_limit=2
            --max_sequence_length 16
            """.split()

            resume_run_args.extend(["--instance_prompt", ""])
            run_command(self._launch_args + resume_run_args)

            self.assertEqual({x for x in os.listdir(tmpdir) if "checkpoint" in x}, {"checkpoint-6", "checkpoint-8"})
