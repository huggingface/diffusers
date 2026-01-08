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

import logging
import os
import sys
import tempfile
import unittest

import safetensors
import torch


sys.path.append("..")
from test_examples_utils import ExamplesTestsAccelerate, run_command  # noqa: E402


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class DreamBoothFiboEditUnitTests(unittest.TestCase):
    """Unit tests for helper functions in train_dreambooth_fibo_edit.py"""

    def test_find_closest_resolution(self):
        """Test the find_closest_resolution function for aspect ratio selection."""
        # Import the function from the training script
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_dreambooth_fibo_edit import find_closest_resolution, RESOLUTIONS_1k

        # Test square image (1:1 aspect ratio)
        width, height = find_closest_resolution(1024, 1024)
        self.assertEqual((width, height), (1024, 1024))

        # Test landscape image
        width, height = find_closest_resolution(1920, 1080)
        # 1920/1080 = 1.778, closest to 1.750 which maps to (1344, 768)
        self.assertIn((width, height), list(RESOLUTIONS_1k.values()))
        self.assertGreater(width, height)  # Should be landscape

        # Test portrait image
        width, height = find_closest_resolution(1080, 1920)
        # 1080/1920 = 0.5625, closest to 0.67 which maps to (832, 1248)
        self.assertIn((width, height), list(RESOLUTIONS_1k.values()))
        self.assertLess(width, height)  # Should be portrait

    def test_clean_json_caption_valid(self):
        """Test clean_json_caption with valid JSON."""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_dreambooth_fibo_edit import clean_json_caption

        # Test valid JSON
        valid_json = '{"prompt": "a photo of a cat", "style": "realistic"}'
        result = clean_json_caption(valid_json)
        self.assertIsInstance(result, str)
        # Should be valid JSON after cleaning
        import json
        parsed = json.loads(result)
        self.assertEqual(parsed["prompt"], "a photo of a cat")

    def test_clean_json_caption_invalid(self):
        """Test clean_json_caption with invalid JSON raises ValueError."""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_dreambooth_fibo_edit import clean_json_caption

        # Test invalid JSON
        invalid_json = "not a valid json"
        with self.assertRaises(ValueError):
            clean_json_caption(invalid_json)

    def test_create_attention_matrix(self):
        """Test the create_attention_matrix function."""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_dreambooth_fibo_edit import create_attention_matrix

        # Create a simple attention mask
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
        result = create_attention_matrix(attention_mask)

        # Check output shape
        self.assertEqual(result.shape, (2, 3, 3))

        # Check that 1s map to 0 (keep) and 0s map to -inf (ignore)
        # First batch: [1,1,0] -> matrix where positions with mask 0 become -inf
        self.assertEqual(result[0, 0, 0].item(), 0.0)  # 1*1 = 1 -> 0
        self.assertEqual(result[0, 0, 2].item(), float("-inf"))  # 1*0 = 0 -> -inf
        self.assertEqual(result[0, 2, 0].item(), float("-inf"))  # 0*1 = 0 -> -inf

    def test_pad_embedding(self):
        """Test the pad_embedding function."""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_dreambooth_fibo_edit import pad_embedding

        # Create a sample embedding
        batch_size, seq_len, dim = 2, 5, 64
        embedding = torch.randn(batch_size, seq_len, dim)
        max_tokens = 10

        padded_embedding, attention_mask = pad_embedding(embedding, max_tokens)

        # Check shapes
        self.assertEqual(padded_embedding.shape, (batch_size, max_tokens, dim))
        self.assertEqual(attention_mask.shape, (batch_size, max_tokens))

        # Check attention mask values
        # First seq_len positions should be 1 (unmasked)
        self.assertTrue(torch.all(attention_mask[:, :seq_len] == 1))
        # Remaining positions should be 0 (masked/padded)
        self.assertTrue(torch.all(attention_mask[:, seq_len:] == 0))

        # Check that original content is preserved
        self.assertTrue(torch.allclose(padded_embedding[:, :seq_len, :], embedding))

    def test_shifted_logit_normal_timestep_sampler(self):
        """Test the ShiftedLogitNormalTimestepSampler."""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_dreambooth_fibo_edit import ShiftedLogitNormalTimestepSampler

        sampler = ShiftedLogitNormalTimestepSampler(std=1.0)

        batch_size = 10
        seq_length = 1024

        timesteps = sampler.sample(batch_size, seq_length)

        # Check output shape
        self.assertEqual(timesteps.shape, (batch_size,))

        # Check that all timesteps are in valid range [0, 1]
        self.assertTrue(torch.all(timesteps >= 0))
        self.assertTrue(torch.all(timesteps <= 1))

    def test_uniform_timestep_sampler(self):
        """Test the UniformTimestepSampler."""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_dreambooth_fibo_edit import UniformTimestepSampler

        sampler = UniformTimestepSampler(min_value=0.0, max_value=1.0)

        batch_size = 100

        timesteps = sampler.sample(batch_size)

        # Check output shape
        self.assertEqual(timesteps.shape, (batch_size,))

        # Check that all timesteps are in valid range [0, 1]
        self.assertTrue(torch.all(timesteps >= 0))
        self.assertTrue(torch.all(timesteps <= 1))

    def test_shifted_stretched_logit_normal_timestep_sampler(self):
        """Test the ShiftedStretchedLogitNormalTimestepSampler."""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_dreambooth_fibo_edit import ShiftedStretchedLogitNormalTimestepSampler

        sampler = ShiftedStretchedLogitNormalTimestepSampler(std=1.0, uniform_prob=0.1)

        batch_size = 100
        seq_length = 1024

        timesteps = sampler.sample(batch_size, seq_length)

        # Check output shape
        self.assertEqual(timesteps.shape, (batch_size,))

        # Check that all timesteps are in valid range [0, 1]
        self.assertTrue(torch.all(timesteps >= 0))
        self.assertTrue(torch.all(timesteps <= 1))

    def test_resolutions_1k_coverage(self):
        """Test that RESOLUTIONS_1k covers common aspect ratios."""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_dreambooth_fibo_edit import RESOLUTIONS_1k

        # Check that we have expected aspect ratios
        aspect_ratios = list(RESOLUTIONS_1k.keys())

        # Should have portrait (< 1), square (= 1), and landscape (> 1) ratios
        portrait_ratios = [r for r in aspect_ratios if r < 1]
        square_ratios = [r for r in aspect_ratios if r == 1]
        landscape_ratios = [r for r in aspect_ratios if r > 1]

        self.assertGreater(len(portrait_ratios), 0, "Should have portrait aspect ratios")
        self.assertEqual(len(square_ratios), 1, "Should have exactly one square ratio")
        self.assertGreater(len(landscape_ratios), 0, "Should have landscape aspect ratios")

        # All resolutions should be divisible by 16 (for VAE)
        for ratio, (w, h) in RESOLUTIONS_1k.items():
            self.assertEqual(w % 16, 0, f"Width {w} for ratio {ratio} should be divisible by 16")
            self.assertEqual(h % 16, 0, f"Height {h} for ratio {ratio} should be divisible by 16")


@unittest.skipUnless(
    os.environ.get("RUN_SLOW", "0") == "1",
    "Slow tests require RUN_SLOW=1 environment variable and a tiny test model",
)
class DreamBoothLoRAFiboEdit(ExamplesTestsAccelerate):
    """
    Integration tests for train_dreambooth_fibo_edit.py.

    NOTE: These tests require a tiny test model at 'hf-internal-testing/tiny-bria-fibo-edit-pipe'
    or the pretrained_model_name_or_path to be updated to point to an available tiny model.

    To run these tests, set RUN_SLOW=1 and ensure the test model is available.
    """

    # NOTE: Update this path once a tiny test model is available
    pretrained_model_name_or_path = "hf-internal-testing/tiny-bria-fibo-edit-pipe"
    script_path = "examples/dreambooth/train_dreambooth_fibo_edit.py"
    # For fibo-edit, we need a dataset with source images, target images, and JSON captions
    # Using a test dataset that should be set up for this purpose
    dataset_name = "hf-internal-testing/fibo-edit-test-dataset"

    def test_dreambooth_lora_fibo_edit(self):
        """Test basic LoRA training for Fibo Edit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --dataset_name {self.dataset_name}
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 1e-4
                --lr_scheduler constant
                --lr_warmup_steps 0
                --lora_rank 4
                --output_dir {tmpdir}
                --mixed_precision no
                --checkpointing_steps 1000
                """.split()

            run_command(self._launch_args + test_args)

            # Check that checkpoint was saved
            checkpoint_dirs = [d for d in os.listdir(tmpdir) if d.startswith("checkpoint")]
            # Either we have checkpoints or final output
            self.assertTrue(
                len(checkpoint_dirs) > 0 or os.path.exists(os.path.join(tmpdir, "checkpoint_final")),
                "Expected checkpoint directories to be created",
            )

    def test_dreambooth_lora_fibo_edit_with_gradient_checkpointing(self):
        """Test LoRA training with gradient checkpointing enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --dataset_name {self.dataset_name}
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 1e-4
                --lr_scheduler constant
                --lr_warmup_steps 0
                --lora_rank 4
                --gradient_checkpointing 1
                --output_dir {tmpdir}
                --mixed_precision no
                --checkpointing_steps 1000
                """.split()

            run_command(self._launch_args + test_args)

            # Check that checkpoint was saved
            checkpoint_dirs = [d for d in os.listdir(tmpdir) if d.startswith("checkpoint")]
            self.assertTrue(
                len(checkpoint_dirs) > 0 or os.path.exists(os.path.join(tmpdir, "checkpoint_final")),
                "Expected checkpoint directories to be created",
            )

    def test_dreambooth_lora_fibo_edit_checkpointing(self):
        """Test checkpointing functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --dataset_name {self.dataset_name}
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 4
                --learning_rate 1e-4
                --lr_scheduler constant
                --lr_warmup_steps 0
                --lora_rank 4
                --checkpointing_steps 2
                --output_dir {tmpdir}
                --mixed_precision no
                """.split()

            run_command(self._launch_args + test_args)

            # Check that intermediate checkpoints were created
            checkpoint_dirs = {d for d in os.listdir(tmpdir) if d.startswith("checkpoint")}
            # Should have checkpoint at step 2 (and possibly final)
            self.assertTrue(
                "checkpoint_2" in checkpoint_dirs or "checkpoint_final" in checkpoint_dirs,
                f"Expected checkpoints, found: {checkpoint_dirs}",
            )

    def test_dreambooth_lora_fibo_edit_resume_from_checkpoint(self):
        """Test resume from checkpoint functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First training run
            test_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --dataset_name {self.dataset_name}
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 1e-4
                --lr_scheduler constant
                --lr_warmup_steps 0
                --lora_rank 4
                --checkpointing_steps 2
                --output_dir {tmpdir}
                --mixed_precision no
                """.split()

            run_command(self._launch_args + test_args)

            # Check that checkpoint was created
            checkpoint_dirs = [d for d in os.listdir(tmpdir) if d.startswith("checkpoint")]
            self.assertGreater(len(checkpoint_dirs), 0, "Expected checkpoint to be created")

            # Resume training
            resume_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --dataset_name {self.dataset_name}
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 4
                --learning_rate 1e-4
                --lr_scheduler constant
                --lr_warmup_steps 0
                --lora_rank 4
                --checkpointing_steps 2
                --resume_from_checkpoint latest
                --output_dir {tmpdir}
                --mixed_precision no
                """.split()

            run_command(self._launch_args + resume_args)

            # Check that training continued and created more checkpoints
            final_checkpoint_dirs = [d for d in os.listdir(tmpdir) if d.startswith("checkpoint")]
            self.assertGreater(
                len(final_checkpoint_dirs),
                len(checkpoint_dirs),
                "Expected more checkpoints after resuming",
            )

    def test_dreambooth_lora_fibo_edit_different_lr_schedulers(self):
        """Test different learning rate schedulers."""
        schedulers = ["constant", "cosine_with_warmup"]

        for scheduler in schedulers:
            with tempfile.TemporaryDirectory() as tmpdir:
                test_args = f"""
                    {self.script_path}
                    --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                    --dataset_name {self.dataset_name}
                    --train_batch_size 1
                    --gradient_accumulation_steps 1
                    --max_train_steps 2
                    --learning_rate 1e-4
                    --lr_scheduler {scheduler}
                    --lr_warmup_steps 1
                    --lora_rank 4
                    --output_dir {tmpdir}
                    --mixed_precision no
                    --checkpointing_steps 1000
                    """.split()

                run_command(self._launch_args + test_args)

                # Check that training completed
                checkpoint_dirs = [d for d in os.listdir(tmpdir) if d.startswith("checkpoint")]
                self.assertTrue(
                    len(checkpoint_dirs) > 0 or os.path.exists(os.path.join(tmpdir, "checkpoint_final")),
                    f"Expected checkpoints with scheduler {scheduler}",
                )

    def test_dreambooth_lora_fibo_edit_lora_weights_structure(self):
        """Test that LoRA weights have the correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --dataset_name {self.dataset_name}
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 1e-4
                --lr_scheduler constant
                --lr_warmup_steps 0
                --lora_rank 4
                --output_dir {tmpdir}
                --mixed_precision no
                --checkpointing_steps 1
                """.split()

            run_command(self._launch_args + test_args)

            # Find the checkpoint directory
            checkpoint_dirs = [d for d in os.listdir(tmpdir) if d.startswith("checkpoint")]
            self.assertGreater(len(checkpoint_dirs), 0, "Expected checkpoint to be created")

            # Check for LoRA weights file in checkpoint
            checkpoint_path = os.path.join(tmpdir, checkpoint_dirs[0])
            lora_weights_path = os.path.join(checkpoint_path, "pytorch_lora_weights.safetensors")

            if os.path.exists(lora_weights_path):
                # Load and verify LoRA weights
                lora_state_dict = safetensors.torch.load_file(lora_weights_path)

                # Check that all keys contain "lora"
                is_lora = all("lora" in k for k in lora_state_dict.keys())
                self.assertTrue(is_lora, "All LoRA state dict keys should contain 'lora'")

                # Check that all keys start with "transformer"
                starts_with_transformer = all(key.startswith("transformer") for key in lora_state_dict.keys())
                self.assertTrue(starts_with_transformer, "All keys should start with 'transformer'")


if __name__ == "__main__":
    unittest.main()
