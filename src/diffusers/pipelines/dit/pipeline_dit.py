from typing import List, Optional, Tuple, Union

import torch

from ...configuration_utils import FrozenDict
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...utils import deprecate
from ...models import AutoencoderKL, DiT


class DiTPipeline(DiffusionPipeline):
    def __init__(self, dit: DiT, vae: AutoencoderKL, scheduler):
        super().__init__()
        self.register_modules(dit=dit, vae=vae, scheduler=scheduler)
