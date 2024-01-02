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

# CURRENTLY WORK IN PROGRESS!

from math import pi
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from diffusers import DDPMScheduler, DiffusionPipeline, ImagePipelineOutput, UNet2DModel
from diffusers.utils.torch_utils import randn_tensor