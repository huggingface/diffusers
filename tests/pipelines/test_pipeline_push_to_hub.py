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

import json
import os
import tempfile
import unittest
import uuid

import torch
from huggingface_hub import ModelCard, delete_repo
from huggingface_hub.utils import is_jinja_available
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from ..others.test_utils import TOKEN, USER, is_staging_test


# Standalone, pipeline-agnostic Hub integration test. It does not compose the `BasePipelineTesterConfig`
# fixtures (it builds its own fixed SD components) and relies on `@is_staging_test` (a `unittest.skip`-based
# decorator), so it stays a `unittest.TestCase` rather than a config + mixin test.
@is_staging_test
class TestPipelinePushToHub(unittest.TestCase):
    identifier = uuid.uuid4()
    repo_id = f"test-pipeline-{identifier}"
    org_repo_id = f"valid_org/{repo_id}-org"

    def get_pipeline_components(self):
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )

        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )

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

        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1, "!": 2}
            vocab_path = os.path.join(tmpdir, "vocab.json")
            with open(vocab_path, "w") as f:
                json.dump(dummy_vocab, f)

            merges = "Ġ t\nĠt h"
            merges_path = os.path.join(tmpdir, "merges.txt")
            with open(merges_path, "w") as f:
                f.writelines(merges)
            tokenizer = CLIPTokenizer(vocab_file=vocab_path, merges_file=merges_path)

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def test_push_to_hub(self):
        components = self.get_pipeline_components()
        pipeline = StableDiffusionPipeline(**components)
        pipeline.push_to_hub(self.repo_id, token=TOKEN)

        new_model = UNet2DConditionModel.from_pretrained(f"{USER}/{self.repo_id}", subfolder="unet")
        unet = components["unet"]
        for p1, p2 in zip(unet.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Push to hub via save_pretrained to a separate repo. Reusing `self.repo_id` after
        # deleting it makes the staging server's LFS GC reject the next commit with
        # "LFS pointer pointed to a file that does not exist" when the model bytes are identical.
        save_repo_id = f"{self.repo_id}-saved"
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir, repo_id=save_repo_id, push_to_hub=True, token=TOKEN)

        new_model = UNet2DConditionModel.from_pretrained(f"{USER}/{save_repo_id}", subfolder="unet")
        for p1, p2 in zip(unet.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repos
        delete_repo(token=TOKEN, repo_id=self.repo_id)
        delete_repo(save_repo_id, token=TOKEN)

    def test_push_to_hub_in_organization(self):
        components = self.get_pipeline_components()
        pipeline = StableDiffusionPipeline(**components)
        pipeline.push_to_hub(self.org_repo_id, token=TOKEN)

        new_model = UNet2DConditionModel.from_pretrained(self.org_repo_id, subfolder="unet")
        unet = components["unet"]
        for p1, p2 in zip(unet.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Push to hub via save_pretrained to a separate repo. Reusing `self.org_repo_id` after
        # deleting it makes the staging server's LFS GC reject the next commit with
        # "LFS pointer pointed to a file that does not exist" when the model bytes are identical.
        save_org_repo_id = f"{self.org_repo_id}-saved"
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir, push_to_hub=True, token=TOKEN, repo_id=save_org_repo_id)

        new_model = UNet2DConditionModel.from_pretrained(save_org_repo_id, subfolder="unet")
        for p1, p2 in zip(unet.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repos
        delete_repo(token=TOKEN, repo_id=self.org_repo_id)
        delete_repo(save_org_repo_id, token=TOKEN)

    @unittest.skipIf(
        not is_jinja_available(),
        reason="Model card tests cannot be performed without Jinja installed.",
    )
    def test_push_to_hub_library_name(self):
        components = self.get_pipeline_components()
        pipeline = StableDiffusionPipeline(**components)
        # Use a method-unique repo to avoid recycling a name that `test_push_to_hub` just deleted,
        # which the staging server rejects with an LFS pointer error.
        repo_id = f"test-pipeline-library-name-{uuid.uuid4()}"
        pipeline.push_to_hub(repo_id, token=TOKEN)

        model_card = ModelCard.load(f"{USER}/{repo_id}", token=TOKEN).data
        assert model_card.library_name == "diffusers"

        # Reset repo
        delete_repo(repo_id, token=TOKEN)
