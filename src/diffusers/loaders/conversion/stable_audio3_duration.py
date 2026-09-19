# Copyright 2026 The HuggingFace Team. All rights reserved.
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


import math

import torch

from .core import Conversion, Rule
from .transforms import WithConstants


def stable_audio3_duration_conversion(config):
    prefix = "conditioner.conditioners.seconds_total.embedder.embedding.1"
    ramp = torch.linspace(0, 1, config["fourier_dim"] // 2, dtype=torch.float32)
    freqs = torch.exp(
        ramp * (math.log(config["max_freq"]) - math.log(config["min_freq"])) + math.log(config["min_freq"])
    )
    return Conversion(
        mapping={prefix + ".bias": "linear.bias"},
        rules=(Rule((prefix + ".weight",), ("linear.weight", "freqs"), WithConstants((freqs,))),),
    )
