# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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
import tempfile
import unittest
from typing import Dict, List, Tuple

import numpy as np
import torch

from diffusers import DDPMScheduler
from diffusers.utils import torch_device


torch.backends.cuda.matmul.allow_tf32 = False


class PNDMSchedulerIntegrationTest(unittest.TestCase):
    pass
