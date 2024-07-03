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


sys.path.append("..")
from test_examples_utils import ExamplesTestsAccelerate, run_command  # noqa: E402


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class TextualInversion(ExamplesTestsAccelerate):
    def test_textual_inversion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/textual_inversion/textual_inversion.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --train_data_dir docs/source/en/imgs
                --learnable_property object
                --placeholder_token <cat-toy>
                --initializer_token a
                --save_steps 1
                --num_vectors 2
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "learned_embeds.safetensors")))

    def test_textual_inversion_checkpointing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/textual_inversion/textual_inversion.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --train_data_dir docs/source/en/imgs
                --learnable_property object
                --placeholder_token <cat-toy>
                --initializer_token a
                --save_steps 1
                --num_vectors 2
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 3
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=1
                --checkpoints_total_limit=2
                """.split()

            run_command(self._launch_args + test_args)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-3"},
            )

    def test_textual_inversion_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/textual_inversion/textual_inversion.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --train_data_dir docs/source/en/imgs
                --learnable_property object
                --placeholder_token <cat-toy>
                --initializer_token a
                --save_steps 1
                --num_vectors 2
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=1
                """.split()

            run_command(self._launch_args + test_args)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-1", "checkpoint-2"},
            )

            resume_run_args = f"""
                examples/textual_inversion/textual_inversion.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --train_data_dir docs/source/en/imgs
                --learnable_property object
                --placeholder_token <cat-toy>
                --initializer_token a
                --save_steps 1
                --num_vectors 2
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=1
                --resume_from_checkpoint=checkpoint-2
                --checkpoints_total_limit=2
                """.split()

            run_command(self._launch_args + resume_run_args)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-3"},
            )
