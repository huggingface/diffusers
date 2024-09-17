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
import shutil
import sys
import tempfile

import pytest
from huggingface_hub import snapshot_download

from diffusers import CogVideoXTransformer3DModel, DiffusionPipeline


sys.path.append("..")
from test_examples_utils import ExamplesTestsAccelerate, run_command  # noqa: E402


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class CogVideoXLoRA(ExamplesTestsAccelerate):
    dataset_name = "hf-internal-testing/tiny-video-dataset"
    instance_data_dir = "videos/"
    caption_column = "captions.txt"
    video_column = "videos.txt"
    instance_prompt = "A hiker standing at the peak of mountain"
    max_num_frames = 9

    pretrained_model_name_or_path = "hf-internal-testing/tiny-cogvideox-pipe"
    script_path = "examples/cogvideo/train_cogvideox_lora.py"

    dataset_path = None

    @pytest.fixture(scope="class", autouse=True)
    def prepare_dummy_inputs(self, request):
        tmpdir = tempfile.mkdtemp()

        try:
            if request.cls.dataset_path is None:
                request.cls.dataset_path = snapshot_download(self.dataset_name, repo_type="dataset", cache_dir=tmpdir)

            yield
        finally:
            shutil.rmtree(tmpdir)

    def test_lora(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --instance_data_root {self.dataset_path}
                --caption_column {self.caption_column}
                --video_column {self.video_column}
                --rank 1
                --lora_alpha 1
                --mixed_precision fp16
                --height 32
                --width 32
                --fps 8
                --max_num_frames {self.max_num_frames}
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 1e-3
                --lr_scheduler constant
                --lr_warmup_steps 0
                --enable_tiling
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args)
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

    def test_lora_checkpointing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 4, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4

            initial_run_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --instance_data_root {self.dataset_path}
                --caption_column {self.caption_column}
                --video_column {self.video_column}
                --rank 1
                --lora_alpha 1
                --mixed_precision fp16
                --height 32
                --width 32
                --fps 8
                --max_num_frames {self.max_num_frames}
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --learning_rate 1e-3
                --lr_scheduler constant
                --lr_warmup_steps 0
                --enable_tiling
                --output_dir {tmpdir}
                --seed 0
                --max_train_steps 4
                --checkpointing_steps 2
            """.split()

            run_command(self._launch_args + initial_run_args)

            # check can run the original fully trained output pipeline
            pipe = DiffusionPipeline.from_pretrained(self.pretrained_model_name_or_path)
            pipe.load_lora_weights(tmpdir)
            pipe(
                self.instance_prompt,
                num_inference_steps=1,
                num_frames=5,
                max_sequence_length=pipe.transformer.config.max_text_seq_length,
            )

            # check checkpoint directories exist
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "checkpoint-2")))
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "checkpoint-4")))

            # check can run an intermediate checkpoint
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="transformer"
            )
            pipe = DiffusionPipeline.from_pretrained(self.pretrained_model_name_or_path, transformer=transformer)
            pipe.load_lora_weights(os.path.join(tmpdir, "checkpoint-2"))
            pipe(
                self.instance_prompt,
                num_inference_steps=1,
                num_frames=5,
                max_sequence_length=pipe.transformer.config.max_text_seq_length,
            )

            # Remove checkpoint 2 so that we can check only later checkpoints exist after resuming
            shutil.rmtree(os.path.join(tmpdir, "checkpoint-2"))

            # Run training script for 7 total steps resuming from checkpoint 4

            resume_run_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --instance_data_root {self.dataset_path}
                --caption_column {self.caption_column}
                --video_column {self.video_column}
                --rank 1
                --lora_alpha 1
                --mixed_precision fp16
                --height 32
                --width 32
                --fps 8
                --max_num_frames {self.max_num_frames}
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --learning_rate 1e-3
                --lr_scheduler constant
                --lr_warmup_steps 0
                --enable_tiling
                --output_dir {tmpdir}
                --seed=0
                --max_train_steps 6
                --checkpointing_steps 2
                --resume_from_checkpoint checkpoint-4
            """.split()

            run_command(self._launch_args + resume_run_args)

            # check can run new fully trained pipeline
            pipe = DiffusionPipeline.from_pretrained(self.pretrained_model_name_or_path)
            pipe(
                self.instance_prompt,
                num_inference_steps=1,
                num_frames=5,
                max_sequence_length=pipe.transformer.config.max_text_seq_length,
            )

            # check old checkpoints do not exist
            self.assertFalse(os.path.isdir(os.path.join(tmpdir, "checkpoint-2")))

            # check new checkpoints exist
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "checkpoint-4")))
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "checkpoint-6")))

    def test_lora_checkpointing_checkpoints_total_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --instance_data_root {self.dataset_path}
                --caption_column {self.caption_column}
                --video_column {self.video_column}
                --rank 1
                --lora_alpha 1
                --mixed_precision fp16
                --height 32
                --width 32
                --fps 8
                --max_num_frames {self.max_num_frames}
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --learning_rate 1e-3
                --lr_scheduler constant
                --lr_warmup_steps 0
                --enable_tiling
                --output_dir {tmpdir}
                --max_train_steps 6
                --checkpointing_steps 2
                --checkpoints_total_limit 2
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_lora_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --instance_data_root {self.dataset_path}
                --caption_column {self.caption_column}
                --video_column {self.video_column}
                --rank 1
                --lora_alpha 1
                --mixed_precision fp16
                --height 32
                --width 32
                --fps 8
                --max_num_frames {self.max_num_frames}
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --learning_rate 1e-3
                --lr_scheduler constant
                --lr_warmup_steps 0
                --enable_tiling
                --output_dir {tmpdir}
                --max_train_steps 4
                --checkpointing_steps=2
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4"},
            )

            resume_run_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --instance_data_root {self.dataset_path}
                --caption_column {self.caption_column}
                --video_column {self.video_column}
                --rank 1
                --lora_alpha 1
                --mixed_precision fp16
                --height 32
                --width 32
                --fps 8
                --max_num_frames {self.max_num_frames}
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --learning_rate 1e-3
                --lr_scheduler constant
                --lr_warmup_steps 0
                --enable_tiling
                --output_dir {tmpdir}
                --max_train_steps 8
                --checkpointing_steps 2
                --resume_from_checkpoint checkpoint-4
                --checkpoints_total_limit 2
            """.split()

            run_command(self._launch_args + resume_run_args)

            self.assertEqual({x for x in os.listdir(tmpdir) if "checkpoint" in x}, {"checkpoint-6", "checkpoint-8"})
