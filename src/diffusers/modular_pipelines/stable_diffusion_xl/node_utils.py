# Copyright 2025 The HuggingFace Team. All rights reserved.
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


SDXL_NODE_TYPES_PARAMS_MAP = {
    "controlnet": {
        "inputs": [
            "control_image",
            "controlnet_conditioning_scale",
            "control_guidance_start",
            "control_guidance_end",
            "height",
            "width",
        ],
        "model_inputs": [
            "controlnet",
        ],
        "outputs": [
            "controlnet_out",
        ],
        "block_names": [None],
    },
    "denoise": {
        "inputs": [
            "embeddings",
            "width",
            "height",
            "seed",
            "num_inference_steps",
            "guidance_scale",
            "image_latents",
            "strength",
            # custom adapters coming in as inputs
            "controlnet",
            # ip_adapter is optional and custom; include if available
            "ip_adapter",
        ],
        "model_inputs": [
            "unet",
            "guider",
            "scheduler",
        ],
        "outputs": [
            "latents",
            "latents_preview",
        ],
        "block_names": ["denoise"],
    },
    "vae_encoder": {
        "inputs": [
            "image",
            "width",
            "height",
        ],
        "model_inputs": [
            "vae",
        ],
        "outputs": [
            "image_latents",
        ],
        "block_names": ["vae_encoder"],
    },
    "text_encoder": {
        "inputs": [
            "prompt",
            "negative_prompt",
        ],
        "model_inputs": [
            "text_encoders",
        ],
        "outputs": [
            "embeddings",
        ],
        "block_names": ["text_encoder"],
    },
    "decoder": {
        "inputs": [
            "latents",
        ],
        "model_inputs": [
            "vae",
        ],
        "outputs": [
            "images",
        ],
        "block_names": ["decode"],
    },
}
