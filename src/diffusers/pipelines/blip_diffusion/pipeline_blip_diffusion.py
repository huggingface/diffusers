from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL
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

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Create a class for the Blip Diffusion pipeline
class BlipDiffusionPipeline(DiffusionPipeline):
    
    def __init__(self, scheduler: DDPMScheduler):
        super().__init__()
        
        self.register_modules(scheduler=scheduler)

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


