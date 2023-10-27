# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
import tempfile
import unittest
import uuid

import numpy as np
import torch
from huggingface_hub import delete_repo, hf_hub_download
from test_utils import TOKEN, is_staging_test
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.constants import WORKFLOW_NAME
from diffusers.utils.testing_utils import torch_device
from diffusers.workflow_utils import populate_workflow_from_pipeline


class WorkflowFastTests(unittest.TestCase):
    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=1,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            norm_num_groups=2,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[4, 8],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=2,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=64,
            layer_norm_eps=1e-05,
            num_attention_heads=8,
            num_hidden_layers=3,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

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

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def test_workflow_with_stable_diffusion(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs, return_workflow=True)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        with tempfile.TemporaryDirectory() as tmpdirname:
            output.workflow.save_pretrained(tmpdirname)

            components = self.get_dummy_components()
            sd_pipe = StableDiffusionPipeline(**components)
            sd_pipe = sd_pipe.to(torch_device)
            sd_pipe.set_progress_bar_config(disable=None)
            sd_pipe.load_workflow(tmpdirname)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images
        workflow_image_slice = image[0, -3:, -3:, -1]

        self.assertTrue(np.allclose(image_slice, workflow_image_slice))


@is_staging_test
class WorkflowPushToHubTester(unittest.TestCase):
    identifier = uuid.uuid4()
    repo_id = f"test-workflow-{identifier}"
    org_repo_id = f"valid_org/{repo_id}-org"

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def test_push_to_hub(self):
        inputs = self.get_dummy_inputs(device="cpu")
        workflow = populate_workflow_from_pipeline(list(inputs.keys()), inputs, None)
        workflow.push_to_hub(self.repo_id, token=TOKEN)

        local_path = hf_hub_download(repo_id=self.repo_id, filename=WORKFLOW_NAME, token=TOKEN)
        with open(local_path) as f:
            locally_loaded_workflow = json.load(f)

        for k in workflow:
            assert workflow[k] == locally_loaded_workflow[k]

        # Reset repo
        delete_repo(token=TOKEN, repo_id=self.repo_id)

    def test_push_to_hub_in_organization(self):
        inputs = self.get_dummy_inputs(device="cpu")
        workflow = populate_workflow_from_pipeline(list(inputs.keys()), inputs, None)
        workflow.push_to_hub(self.org_repo_id, token=TOKEN)

        local_path = hf_hub_download(repo_id=self.org_repo_id, filename=WORKFLOW_NAME, token=TOKEN)
        with open(local_path) as f:
            locally_loaded_workflow = json.load(f)

        for k in workflow:
            assert workflow[k] == locally_loaded_workflow[k]

        # Reset repo
        delete_repo(token=TOKEN, repo_id=self.org_repo_id)
