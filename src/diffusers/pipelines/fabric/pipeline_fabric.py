import warnings
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,
)
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.cross_attention import LoRACrossAttnProcessor
