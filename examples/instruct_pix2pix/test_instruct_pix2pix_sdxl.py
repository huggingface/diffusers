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
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from PIL import Image
from safetensors.torch import load_file
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLInstructPix2PixPipeline, UNet2DConditionModel


@pytest.fixture
def pretrained_model_path(tmp_path):
    torch.manual_seed(0)
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"<|startoftext|>": 0, "<|endoftext|>": 1, "a</w>": 2, "b</w>": 3}))
    merges_path = tmp_path / "merges.txt"
    merges_path.write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(str(vocab_path), str(merges_path), model_max_length=8)
    text_config = CLIPTextConfig(
        vocab_size=4,
        hidden_size=32,
        intermediate_size=37,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=8,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
    )
    pipeline = StableDiffusionXLInstructPix2PixPipeline(
        text_encoder=CLIPTextModel(text_config),
        text_encoder_2=CLIPTextModelWithProjection(text_config),
        tokenizer=tokenizer,
        tokenizer_2=tokenizer,
        vae=AutoencoderKL(
            block_out_channels=(32, 64),
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
            latent_channels=4,
            sample_size=16,
        ),
        unet=UNet2DConditionModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            block_out_channels=(32, 64),
            layers_per_block=1,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            attention_head_dim=(2, 4),
            cross_attention_dim=64,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            projection_class_embeddings_input_dim=80,
        ),
        scheduler=DDPMScheduler(num_train_timesteps=10),
        add_watermarker=False,
    )
    model_path = tmp_path / "model"
    pipeline.save_pretrained(model_path)
    return model_path


@pytest.mark.parametrize("conditioning_dropout_prob", [0.0, 0.5])
def test_text_conditioning_dropout(tmp_path, pretrained_model_path, conditioning_dropout_prob):
    trained_weights = []
    for prompt in ("a", "b"):
        dataset_path = tmp_path / f"data-{prompt}"
        dataset_path.mkdir()
        dataset = Dataset.from_dict(
            {
                "original_image": [Image.new("RGB", (16, 16), "red")] * 2,
                "edited_image": [Image.new("RGB", (16, 16), "blue")] * 2,
                "edit_prompt": [prompt] * 2,
            }
        )
        dataset.to_parquet(dataset_path / "train.parquet")
        output_path = tmp_path / f"output-{prompt}"
        command = [
            sys.executable,
            str(Path(__file__).with_name("train_instruct_pix2pix_sdxl.py")),
            f"--pretrained_model_name_or_path={pretrained_model_path}",
            f"--dataset_name={dataset_path}",
            "--original_image_column=original_image",
            "--edited_image_column=edited_image",
            "--edit_prompt_column=edit_prompt",
            "--resolution=16",
            "--center_crop",
            "--train_batch_size=2",
            "--max_train_steps=1",
            "--learning_rate=0.001",
            f"--conditioning_dropout_prob={conditioning_dropout_prob}",
            "--mixed_precision=no",
            "--report_to=tensorboard",
            "--seed=0",
            f"--output_dir={output_path}",
        ]
        result = subprocess.run(
            command,
            env={**os.environ, "ACCELERATE_USE_CPU": "true", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        trained_weights.append(load_file(output_path / "unet" / "diffusion_pytorch_model.safetensors"))

    if conditioning_dropout_prob == 0.5:
        for name in trained_weights[0]:
            torch.testing.assert_close(trained_weights[0][name], trained_weights[1][name], rtol=0, atol=0)
    else:
        assert any(not torch.equal(trained_weights[0][name], trained_weights[1][name]) for name in trained_weights[0])
