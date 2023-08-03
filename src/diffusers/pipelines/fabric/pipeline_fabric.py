# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from .cross_attention import LoRACrossAttnProcessor
from .attention import BasicTransformerBlock
from .pipelines import StableDiffusionPipeline
from .scheduler import EulerAncestralDiscreateScheduler
from .embeddings import TimestepEmbedding, Timesteps
from .modeling_utils import ModelMixin
from .unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FabricModel(nn.Module):
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_ckpt: Optional[str] = None,
        stable_diffusion_version: str = "1.5",
        lora_weights: Optional[str] = None,
        torch_dtype=torch.float32,
    ):
        super().__init__()
        # Getting UNet from Stable diffusion 
        if stable_diffusion_version == "2.1":
            warnings.warn("StableDiffusion v2.x is not supported and may give unexpected results.")

        if model_name is None:
            if stable_diffusion_version == "1.5":
                model_name = "runwayml/stable-diffusion-v1-5"
            elif stable_diffusion_version == "2.1":
                model_name = "stabilityai/stable-diffusion-2-1"
            else:
                raise ValueError(
                    f"Unknown stable diffusion version: {stable_diffusion_version}. Version must be either '1.5' or '2.1'"
                )

        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")

        if model_ckpt is not None:
            pipe = StableDiffusionPipeline.from_ckpt(
                model_ckpt,
                scheduler=scheduler,
                torch_dtype=torch_dtype,
                safety_checker=None,
            )
            pipe.scheduler = scheduler
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                scheduler=scheduler,
                torch_dtype=torch_dtype,
                safety_checker=None,
            )

        if lora_weights:
            print(f"Applying LoRA weights from {lora_weights}")
            apply_unet_lora_weights(
                pipeline=pipe, unet_path=lora_weights
            )

        self.pipeline = pipe
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.scheduler = scheduler
        self.dtype = torch_dtype

    def forward(
        self, 
        prompt: Union[str, List[str]] = "a photo of an astronaut riding a horse on mars",
        negative_prompt: Optional[Union[str, List[str]]] = "",
        liked: Optional[List[Image.Image]] = [],
        disliked: Optional[List[Image.Image]] = [],
        seed: int = 42,
        n_images: int = 1,
        guidance_scale: float = 8.0,
        denoising_steps: int = 20,
        feedback_start: float = 0.33,
        feedback_end: float = 0.66,
        min_weight: float = 0.1,
        max_weight: float = 1.0,
        neg_scale: float = 0.5,
        pos_bottleneck_scale: float = 1.0,
        neg_bottleneck_scale: float = 1.0,
    )

        with tqdm(total=denoising_steps) as pbar:
            for i, t in enumerate(timestamp):
                
        pass




