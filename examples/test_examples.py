# coding=utf-8
# Copyright 2023 HuggingFace Inc..
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
import subprocess
import sys
import tempfile
import unittest
from typing import List

import safetensors
from accelerate.utils import write_basic_config

from diffusers import DiffusionPipeline, UNet2DConditionModel


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


# These utils relate to ensuring the right error message is received when running scripts
class SubprocessCallException(Exception):
    pass


def run_command(command: List[str], return_stdout=False):
    """
    Runs `command` with `subprocess.check_output` and will potentially return the `stdout`. Will also properly capture
    if an error occurred while running `command`
    """
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        if return_stdout:
            if hasattr(output, "decode"):
                output = output.decode("utf-8")
            return output
    except subprocess.CalledProcessError as e:
        raise SubprocessCallException(
            f"Command `{' '.join(command)}` failed with the following error:\n\n{e.output.decode()}"
        ) from e


stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class ExamplesTestsAccelerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.mkdtemp()
        cls.configPath = os.path.join(cls._tmpdir, "default_config.yml")

        write_basic_config(save_location=cls.configPath)
        cls._launch_args = ["accelerate", "launch", "--config_file", cls.configPath]

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(cls._tmpdir)

    def test_train_unconditional(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/unconditional_image_generation/train_unconditional.py
                --dataset_name hf-internal-testing/dummy_image_class_data
                --model_config_name_or_path diffusers/ddpm_dummy
                --resolution 64
                --output_dir {tmpdir}
                --train_batch_size 2
                --num_epochs 1
                --gradient_accumulation_steps 1
                --ddpm_num_inference_steps 2
                --learning_rate 1e-3
                --lr_warmup_steps 5
                """.split()

            run_command(self._launch_args + test_args, return_stdout=True)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))

    def test_textual_inversion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/textual_inversion/textual_inversion.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --train_data_dir docs/source/en/imgs
                --learnable_property object
                --placeholder_token <cat-toy>
                --initializer_token a
                --validation_prompt <cat-toy>
                --validation_steps 1
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

    def test_dreambooth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/dreambooth/train_dreambooth.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --instance_data_dir docs/source/en/imgs
                --instance_prompt photo
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
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))

    def test_dreambooth_if(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/dreambooth/train_dreambooth.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-if-pipe
                --instance_data_dir docs/source/en/imgs
                --instance_prompt photo
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --pre_compute_text_embeddings
                --tokenizer_max_length=77
                --text_encoder_use_attention_mask
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))

    def test_dreambooth_checkpointing(self):
        instance_prompt = "photo"
        pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 5, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4

            initial_run_args = f"""
                examples/dreambooth/train_dreambooth.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --instance_data_dir docs/source/en/imgs
                --instance_prompt {instance_prompt}
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 5
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            # check can run the original fully trained output pipeline
            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(instance_prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "checkpoint-2")))
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "checkpoint-4")))

            # check can run an intermediate checkpoint
            unet = UNet2DConditionModel.from_pretrained(tmpdir, subfolder="checkpoint-2/unet")
            pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, unet=unet, safety_checker=None)
            pipe(instance_prompt, num_inference_steps=2)

            # Remove checkpoint 2 so that we can check only later checkpoints exist after resuming
            shutil.rmtree(os.path.join(tmpdir, "checkpoint-2"))

            # Run training script for 7 total steps resuming from checkpoint 4

            resume_run_args = f"""
                examples/dreambooth/train_dreambooth.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --instance_data_dir docs/source/en/imgs
                --instance_prompt {instance_prompt}
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --resume_from_checkpoint=checkpoint-4
                --seed=0
                """.split()

            run_command(self._launch_args + resume_run_args)

            # check can run new fully trained pipeline
            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(instance_prompt, num_inference_steps=2)

            # check old checkpoints do not exist
            self.assertFalse(os.path.isdir(os.path.join(tmpdir, "checkpoint-2")))

            # check new checkpoints exist
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "checkpoint-4")))
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "checkpoint-6")))

    def test_dreambooth_lora(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/dreambooth/train_dreambooth_lora.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --instance_data_dir docs/source/en/imgs
                --instance_prompt photo
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
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

            # make sure the state_dict has the correct naming in the parameters.
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)

            # when not training the text encoder, all the parameters in the state dict should start
            # with `"unet"` in their names.
            starts_with_unet = all(key.startswith("unet") for key in lora_state_dict.keys())
            self.assertTrue(starts_with_unet)

    def test_dreambooth_lora_with_text_encoder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/dreambooth/train_dreambooth_lora.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --instance_data_dir docs/source/en/imgs
                --instance_prompt photo
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --train_text_encoder
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

            # check `text_encoder` is present at all.
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            keys = lora_state_dict.keys()
            is_text_encoder_present = any(k.startswith("text_encoder") for k in keys)
            self.assertTrue(is_text_encoder_present)

            # the names of the keys of the state dict should either start with `unet`
            # or `text_encoder`.
            is_correct_naming = all(k.startswith("unet") or k.startswith("text_encoder") for k in keys)
            self.assertTrue(is_correct_naming)

    def test_dreambooth_lora_if_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/dreambooth/train_dreambooth_lora.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-if-pipe
                --instance_data_dir docs/source/en/imgs
                --instance_prompt photo
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --pre_compute_text_embeddings
                --tokenizer_max_length=77
                --text_encoder_use_attention_mask
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

            # make sure the state_dict has the correct naming in the parameters.
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)

            # when not training the text encoder, all the parameters in the state dict should start
            # with `"unet"` in their names.
            starts_with_unet = all(key.startswith("unet") for key in lora_state_dict.keys())
            self.assertTrue(starts_with_unet)

    def test_dreambooth_lora_sdxl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/dreambooth/train_dreambooth_lora_sdxl.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-xl-pipe
                --instance_data_dir docs/source/en/imgs
                --instance_prompt photo
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
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

            # make sure the state_dict has the correct naming in the parameters.
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)

            # when not training the text encoder, all the parameters in the state dict should start
            # with `"unet"` in their names.
            starts_with_unet = all(key.startswith("unet") for key in lora_state_dict.keys())
            self.assertTrue(starts_with_unet)

    def test_dreambooth_lora_sdxl_with_text_encoder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/dreambooth/train_dreambooth_lora_sdxl.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-xl-pipe
                --instance_data_dir docs/source/en/imgs
                --instance_prompt photo
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --train_text_encoder
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

            # make sure the state_dict has the correct naming in the parameters.
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)

            # when not training the text encoder, all the parameters in the state dict should start
            # with `"unet"` or `"text_encoder"` or `"text_encoder_2"` in their names.
            keys = lora_state_dict.keys()
            starts_with_unet = all(
                k.startswith("unet") or k.startswith("text_encoder") or k.startswith("text_encoder_2") for k in keys
            )
            self.assertTrue(starts_with_unet)

    def test_dreambooth_lora_sdxl_checkpointing_checkpoints_total_limit(self):
        pipeline_path = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"

        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/dreambooth/train_dreambooth_lora_sdxl.py
                --pretrained_model_name_or_path {pipeline_path}
                --instance_data_dir docs/source/en/imgs
                --instance_prompt photo
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
                --checkpointing_steps=2
                --checkpoints_total_limit=2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args)

            pipe = DiffusionPipeline.from_pretrained(pipeline_path)
            pipe.load_lora_weights(tmpdir)
            pipe("a prompt", num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                # checkpoint-2 should have been deleted
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_dreambooth_lora_sdxl_text_encoder_checkpointing_checkpoints_total_limit(self):
        pipeline_path = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"

        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/dreambooth/train_dreambooth_lora_sdxl.py
                --pretrained_model_name_or_path {pipeline_path}
                --instance_data_dir docs/source/en/imgs
                --instance_prompt photo
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
                --checkpointing_steps=2
                --checkpoints_total_limit=2
                --train_text_encoder
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args)

            pipe = DiffusionPipeline.from_pretrained(pipeline_path)
            pipe.load_lora_weights(tmpdir)
            pipe("a prompt", num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                # checkpoint-2 should have been deleted
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_custom_diffusion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/custom_diffusion/train_custom_diffusion.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --instance_data_dir docs/source/en/imgs
                --instance_prompt <new1>
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 1.0e-05
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --modifier_token <new1>
                --no_safe_serialization
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_custom_diffusion_weights.bin")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "<new1>.bin")))

    def test_text_to_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
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
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))

    def test_text_to_image_checkpointing(self):
        pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"
        prompt = "a prompt"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 5, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4

            initial_run_args = f"""
                examples/text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 5
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4"},
            )

            # check can run an intermediate checkpoint
            unet = UNet2DConditionModel.from_pretrained(tmpdir, subfolder="checkpoint-2/unet")
            pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, unet=unet, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            # Remove checkpoint 2 so that we can check only later checkpoints exist after resuming
            shutil.rmtree(os.path.join(tmpdir, "checkpoint-2"))

            # Run training script for 7 total steps resuming from checkpoint 4

            resume_run_args = f"""
                examples/text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --resume_from_checkpoint=checkpoint-4
                --seed=0
                """.split()

            run_command(self._launch_args + resume_run_args)

            # check can run new fully trained pipeline
            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {
                    # no checkpoint-2 -> check old checkpoints do not exist
                    # check new checkpoints exist
                    "checkpoint-4",
                    "checkpoint-6",
                },
            )

    def test_text_to_image_checkpointing_use_ema(self):
        pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"
        prompt = "a prompt"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 5, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4

            initial_run_args = f"""
                examples/text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 5
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --use_ema
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4"},
            )

            # check can run an intermediate checkpoint
            unet = UNet2DConditionModel.from_pretrained(tmpdir, subfolder="checkpoint-2/unet")
            pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, unet=unet, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            # Remove checkpoint 2 so that we can check only later checkpoints exist after resuming
            shutil.rmtree(os.path.join(tmpdir, "checkpoint-2"))

            # Run training script for 7 total steps resuming from checkpoint 4

            resume_run_args = f"""
                examples/text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --resume_from_checkpoint=checkpoint-4
                --use_ema
                --seed=0
                """.split()

            run_command(self._launch_args + resume_run_args)

            # check can run new fully trained pipeline
            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {
                    # no checkpoint-2 -> check old checkpoints do not exist
                    # check new checkpoints exist
                    "checkpoint-4",
                    "checkpoint-6",
                },
            )

    def test_text_to_image_checkpointing_checkpoints_total_limit(self):
        pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"
        prompt = "a prompt"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 7, checkpointing_steps == 2, checkpoints_total_limit == 2
            # Should create checkpoints at steps 2, 4, 6
            # with checkpoint at step 2 deleted

            initial_run_args = f"""
                examples/text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --checkpoints_total_limit=2
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                # checkpoint-2 should have been deleted
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_text_to_image_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"
        prompt = "a prompt"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 9, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4, 6, 8

            initial_run_args = f"""
                examples/text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 9
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4", "checkpoint-6", "checkpoint-8"},
            )

            # resume and we should try to checkpoint at 10, where we'll have to remove
            # checkpoint-2 and checkpoint-4 instead of just a single previous checkpoint

            resume_run_args = f"""
                examples/text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 11
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --resume_from_checkpoint=checkpoint-8
                --checkpoints_total_limit=3
                --seed=0
                """.split()

            run_command(self._launch_args + resume_run_args)

            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-6", "checkpoint-8", "checkpoint-10"},
            )

    def test_text_to_image_sdxl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/text_to_image/train_text_to_image_sdxl.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-xl-pipe
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
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
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))

    def test_text_to_image_lora_checkpointing_checkpoints_total_limit(self):
        pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"
        prompt = "a prompt"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 7, checkpointing_steps == 2, checkpoints_total_limit == 2
            # Should create checkpoints at steps 2, 4, 6
            # with checkpoint at step 2 deleted

            initial_run_args = f"""
                examples/text_to_image/train_text_to_image_lora.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --checkpoints_total_limit=2
                --seed=0
                --num_validation_images=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-pipe", safety_checker=None
            )
            pipe.load_lora_weights(tmpdir)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                # checkpoint-2 should have been deleted
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_text_to_image_lora_sdxl_checkpointing_checkpoints_total_limit(self):
        prompt = "a prompt"
        pipeline_path = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 7, checkpointing_steps == 2, checkpoints_total_limit == 2
            # Should create checkpoints at steps 2, 4, 6
            # with checkpoint at step 2 deleted

            initial_run_args = f"""
                examples/text_to_image/train_text_to_image_lora_sdxl.py
                --pretrained_model_name_or_path {pipeline_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --checkpoints_total_limit=2
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(pipeline_path)
            pipe.load_lora_weights(tmpdir)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                # checkpoint-2 should have been deleted
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_text_to_image_lora_sdxl_text_encoder_checkpointing_checkpoints_total_limit(self):
        prompt = "a prompt"
        pipeline_path = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 7, checkpointing_steps == 2, checkpoints_total_limit == 2
            # Should create checkpoints at steps 2, 4, 6
            # with checkpoint at step 2 deleted

            initial_run_args = f"""
                examples/text_to_image/train_text_to_image_lora_sdxl.py
                --pretrained_model_name_or_path {pipeline_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --train_text_encoder
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --checkpoints_total_limit=2
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(pipeline_path)
            pipe.load_lora_weights(tmpdir)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                # checkpoint-2 should have been deleted
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_text_to_image_lora_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"
        prompt = "a prompt"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 9, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4, 6, 8

            initial_run_args = f"""
                examples/text_to_image/train_text_to_image_lora.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 9
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --seed=0
                --num_validation_images=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-pipe", safety_checker=None
            )
            pipe.load_lora_weights(tmpdir)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4", "checkpoint-6", "checkpoint-8"},
            )

            # resume and we should try to checkpoint at 10, where we'll have to remove
            # checkpoint-2 and checkpoint-4 instead of just a single previous checkpoint

            resume_run_args = f"""
                examples/text_to_image/train_text_to_image_lora.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 11
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --resume_from_checkpoint=checkpoint-8
                --checkpoints_total_limit=3
                --seed=0
                --num_validation_images=0
                """.split()

            run_command(self._launch_args + resume_run_args)

            pipe = DiffusionPipeline.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-pipe", safety_checker=None
            )
            pipe.load_lora_weights(tmpdir)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-6", "checkpoint-8", "checkpoint-10"},
            )

    def test_unconditional_checkpointing_checkpoints_total_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            initial_run_args = f"""
                examples/unconditional_image_generation/train_unconditional.py
                --dataset_name hf-internal-testing/dummy_image_class_data
                --model_config_name_or_path diffusers/ddpm_dummy
                --resolution 64
                --output_dir {tmpdir}
                --train_batch_size 1
                --num_epochs 1
                --gradient_accumulation_steps 1
                --ddpm_num_inference_steps 2
                --learning_rate 1e-3
                --lr_warmup_steps 5
                --checkpointing_steps=2
                --checkpoints_total_limit=2
                """.split()

            run_command(self._launch_args + initial_run_args)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                # checkpoint-2 should have been deleted
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_unconditional_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            initial_run_args = f"""
                examples/unconditional_image_generation/train_unconditional.py
                --dataset_name hf-internal-testing/dummy_image_class_data
                --model_config_name_or_path diffusers/ddpm_dummy
                --resolution 64
                --output_dir {tmpdir}
                --train_batch_size 1
                --num_epochs 1
                --gradient_accumulation_steps 1
                --ddpm_num_inference_steps 2
                --learning_rate 1e-3
                --lr_warmup_steps 5
                --checkpointing_steps=1
                """.split()

            run_command(self._launch_args + initial_run_args)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-1", "checkpoint-2", "checkpoint-3", "checkpoint-4", "checkpoint-5", "checkpoint-6"},
            )

            resume_run_args = f"""
                examples/unconditional_image_generation/train_unconditional.py
                --dataset_name hf-internal-testing/dummy_image_class_data
                --model_config_name_or_path diffusers/ddpm_dummy
                --resolution 64
                --output_dir {tmpdir}
                --train_batch_size 1
                --num_epochs 2
                --gradient_accumulation_steps 1
                --ddpm_num_inference_steps 2
                --learning_rate 1e-3
                --lr_warmup_steps 5
                --resume_from_checkpoint=checkpoint-6
                --checkpointing_steps=2
                --checkpoints_total_limit=3
                """.split()

            run_command(self._launch_args + resume_run_args)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-8", "checkpoint-10", "checkpoint-12"},
            )

    def test_textual_inversion_checkpointing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/textual_inversion/textual_inversion.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --train_data_dir docs/source/en/imgs
                --learnable_property object
                --placeholder_token <cat-toy>
                --initializer_token a
                --validation_prompt <cat-toy>
                --validation_steps 1
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
                --validation_prompt <cat-toy>
                --validation_steps 1
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
                """.split()

            run_command(self._launch_args + test_args)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-1", "checkpoint-2", "checkpoint-3"},
            )

            resume_run_args = f"""
                examples/textual_inversion/textual_inversion.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --train_data_dir docs/source/en/imgs
                --learnable_property object
                --placeholder_token <cat-toy>
                --initializer_token a
                --validation_prompt <cat-toy>
                --validation_steps 1
                --save_steps 1
                --num_vectors 2
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 4
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=1
                --resume_from_checkpoint=checkpoint-3
                --checkpoints_total_limit=2
                """.split()

            run_command(self._launch_args + resume_run_args)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-3", "checkpoint-4"},
            )

    def test_instruct_pix2pix_checkpointing_checkpoints_total_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/instruct_pix2pix/train_instruct_pix2pix.py
                --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
                --dataset_name=hf-internal-testing/instructpix2pix-10-samples
                --resolution=64
                --random_flip
                --train_batch_size=1
                --max_train_steps=7
                --checkpointing_steps=2
                --checkpoints_total_limit=2
                --output_dir {tmpdir}
                --seed=0
                """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_instruct_pix2pix_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/instruct_pix2pix/train_instruct_pix2pix.py
                --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
                --dataset_name=hf-internal-testing/instructpix2pix-10-samples
                --resolution=64
                --random_flip
                --train_batch_size=1
                --max_train_steps=9
                --checkpointing_steps=2
                --output_dir {tmpdir}
                --seed=0
                """.split()

            run_command(self._launch_args + test_args)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4", "checkpoint-6", "checkpoint-8"},
            )

            resume_run_args = f"""
                examples/instruct_pix2pix/train_instruct_pix2pix.py
                --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
                --dataset_name=hf-internal-testing/instructpix2pix-10-samples
                --resolution=64
                --random_flip
                --train_batch_size=1
                --max_train_steps=11
                --checkpointing_steps=2
                --output_dir {tmpdir}
                --seed=0
                --resume_from_checkpoint=checkpoint-8
                --checkpoints_total_limit=3
                """.split()

            run_command(self._launch_args + resume_run_args)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-6", "checkpoint-8", "checkpoint-10"},
            )

    def test_dreambooth_checkpointing_checkpoints_total_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            examples/dreambooth/train_dreambooth.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --instance_data_dir=docs/source/en/imgs
            --output_dir={tmpdir}
            --instance_prompt=prompt
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=6
            --checkpoints_total_limit=2
            --checkpointing_steps=2
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_dreambooth_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            examples/dreambooth/train_dreambooth.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --instance_data_dir=docs/source/en/imgs
            --output_dir={tmpdir}
            --instance_prompt=prompt
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=9
            --checkpointing_steps=2
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4", "checkpoint-6", "checkpoint-8"},
            )

            resume_run_args = f"""
            examples/dreambooth/train_dreambooth.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --instance_data_dir=docs/source/en/imgs
            --output_dir={tmpdir}
            --instance_prompt=prompt
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=11
            --checkpointing_steps=2
            --resume_from_checkpoint=checkpoint-8
            --checkpoints_total_limit=3
            """.split()

            run_command(self._launch_args + resume_run_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-6", "checkpoint-8", "checkpoint-10"},
            )

    def test_dreambooth_lora_checkpointing_checkpoints_total_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            examples/dreambooth/train_dreambooth_lora.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --instance_data_dir=docs/source/en/imgs
            --output_dir={tmpdir}
            --instance_prompt=prompt
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=6
            --checkpoints_total_limit=2
            --checkpointing_steps=2
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_dreambooth_lora_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            examples/dreambooth/train_dreambooth_lora.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --instance_data_dir=docs/source/en/imgs
            --output_dir={tmpdir}
            --instance_prompt=prompt
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=9
            --checkpointing_steps=2
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4", "checkpoint-6", "checkpoint-8"},
            )

            resume_run_args = f"""
            examples/dreambooth/train_dreambooth_lora.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --instance_data_dir=docs/source/en/imgs
            --output_dir={tmpdir}
            --instance_prompt=prompt
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=11
            --checkpointing_steps=2
            --resume_from_checkpoint=checkpoint-8
            --checkpoints_total_limit=3
            """.split()

            run_command(self._launch_args + resume_run_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-6", "checkpoint-8", "checkpoint-10"},
            )

    def test_controlnet_checkpointing_checkpoints_total_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            examples/controlnet/train_controlnet.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --dataset_name=hf-internal-testing/fill10
            --output_dir={tmpdir}
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=6
            --checkpoints_total_limit=2
            --checkpointing_steps=2
            --controlnet_model_name_or_path=hf-internal-testing/tiny-controlnet
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_controlnet_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            examples/controlnet/train_controlnet.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --dataset_name=hf-internal-testing/fill10
            --output_dir={tmpdir}
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --controlnet_model_name_or_path=hf-internal-testing/tiny-controlnet
            --max_train_steps=9
            --checkpointing_steps=2
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4", "checkpoint-6", "checkpoint-8"},
            )

            resume_run_args = f"""
            examples/controlnet/train_controlnet.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --dataset_name=hf-internal-testing/fill10
            --output_dir={tmpdir}
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --controlnet_model_name_or_path=hf-internal-testing/tiny-controlnet
            --max_train_steps=11
            --checkpointing_steps=2
            --resume_from_checkpoint=checkpoint-8
            --checkpoints_total_limit=3
            """.split()

            run_command(self._launch_args + resume_run_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-8", "checkpoint-10", "checkpoint-12"},
            )

    def test_controlnet_sdxl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            examples/controlnet/train_controlnet_sdxl.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-xl-pipe
            --dataset_name=hf-internal-testing/fill10
            --output_dir={tmpdir}
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --controlnet_model_name_or_path=hf-internal-testing/tiny-controlnet-sdxl
            --max_train_steps=9
            --checkpointing_steps=2
            """.split()

            run_command(self._launch_args + test_args)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "diffusion_pytorch_model.safetensors")))

    def test_t2i_adapter_sdxl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            examples/t2i_adapter/train_t2i_adapter_sdxl.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-xl-pipe
            --adapter_model_name_or_path=hf-internal-testing/tiny-adapter
            --dataset_name=hf-internal-testing/fill10
            --output_dir={tmpdir}
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=9
            --checkpointing_steps=2
            """.split()

            run_command(self._launch_args + test_args)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "diffusion_pytorch_model.safetensors")))

    def test_custom_diffusion_checkpointing_checkpoints_total_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            examples/custom_diffusion/train_custom_diffusion.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --instance_data_dir=docs/source/en/imgs
            --output_dir={tmpdir}
            --instance_prompt=<new1>
            --resolution=64
            --train_batch_size=1
            --modifier_token=<new1>
            --dataloader_num_workers=0
            --max_train_steps=6
            --checkpoints_total_limit=2
            --checkpointing_steps=2
            --no_safe_serialization
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_custom_diffusion_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            examples/custom_diffusion/train_custom_diffusion.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --instance_data_dir=docs/source/en/imgs
            --output_dir={tmpdir}
            --instance_prompt=<new1>
            --resolution=64
            --train_batch_size=1
            --modifier_token=<new1>
            --dataloader_num_workers=0
            --max_train_steps=9
            --checkpointing_steps=2
            --no_safe_serialization
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4", "checkpoint-6", "checkpoint-8"},
            )

            resume_run_args = f"""
            examples/custom_diffusion/train_custom_diffusion.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --instance_data_dir=docs/source/en/imgs
            --output_dir={tmpdir}
            --instance_prompt=<new1>
            --resolution=64
            --train_batch_size=1
            --modifier_token=<new1>
            --dataloader_num_workers=0
            --max_train_steps=11
            --checkpointing_steps=2
            --resume_from_checkpoint=checkpoint-8
            --checkpoints_total_limit=3
            --no_safe_serialization
            """.split()

            run_command(self._launch_args + resume_run_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-6", "checkpoint-8", "checkpoint-10"},
            )

    def test_text_to_image_lora_sdxl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/text_to_image/train_text_to_image_lora_sdxl.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-xl-pipe
                --dataset_name hf-internal-testing/dummy_image_text_data
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
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

            # make sure the state_dict has the correct naming in the parameters.
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)

    def test_text_to_image_lora_sdxl_with_text_encoder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/text_to_image/train_text_to_image_lora_sdxl.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-xl-pipe
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --train_text_encoder
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

            # make sure the state_dict has the correct naming in the parameters.
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)

            # when not training the text encoder, all the parameters in the state dict should start
            # with `"unet"` or `"text_encoder"` or `"text_encoder_2"` in their names.
            keys = lora_state_dict.keys()
            starts_with_unet = all(
                k.startswith("unet") or k.startswith("text_encoder") or k.startswith("text_encoder_2") for k in keys
            )
            self.assertTrue(starts_with_unet)
