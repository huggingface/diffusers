from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL
from ...models import AutoencoderKL, UNet2DConditionModel
from .modeling_ctx_clip import CtxCLIPTextModel
from transformers import CLIPTokenizer
from ...pipelines import DiffusionPipeline
import torch
from ...schedulers import DDIMScheduler, DDPMScheduler
from ...utils import (
    BaseOutput,
    is_accelerate_available,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from torch import nn
from transformers.activations import QuickGELUActivation as QuickGELU
from .modeling_blip2 import Blip2VisionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class ProjLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, drop_p=0.1, eps=1e-12):
        super().__init__()

        # Dense1 -> Act -> Dense2 -> Drop -> Res -> Norm
        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.act_fn = QuickGELU()
        self.dense2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(drop_p)

        self.LayerNorm = nn.LayerNorm(out_dim, eps=eps)

    def forward(self, x):
        x_in = x

        x = self.LayerNorm(x)
        x = self.dropout(self.dense2(self.act_fn(self.dense1(x)))) + x_in

        return x


# Create a class for the Blip Diffusion pipeline
class BlipDiffusionPipeline(DiffusionPipeline):
    
    def __init__(self, tokenizer: CLIPTokenizer, text_encoder: CtxCLIPTextModel, vae: AutoencoderKL, unet: UNet2DConditionModel, scheduler: DDPMScheduler, vision_encoder: Blip2VisionModel):
        super().__init__()
        
        self.register_modules(tokenizer=tokenizer, text_encoder=CtxCLIPTextModel,  vae=vae, unet=unet, scheduler=scheduler, vision_encoder=vision_encoder)

    def prepare_latents():
        pass

    def encode_prompt():
        pass
    
    def enable_sequential_cpu_offload():
        pass

    def enable_model_cpu_offload(self, gpu_id=0):
        pass

    def __call__(self):
        pass


