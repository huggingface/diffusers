# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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


# mellon nodes
QwenImage_NODE_TYPES_PARAMS_MAP = {
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
            "vae",
        ],
        "outputs": [
            "controlnet_out",
        ],
        "block_names": ["controlnet_vae_encoder"],
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
            "controlnet",
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
    },
}
