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


import torch

from .core import Conversion, Rule
from .transforms import WithConstants


def hifigan_conversion(config):
    modules = [("conv_pre", "conv_pre"), ("conv_post", "conv_post")]
    count = len(config["upsample_rates"])
    modules.extend((f"ups.{i}", f"upsampler.{i}") for i in range(count))
    for i in range(count):
        for j, dilations in enumerate(config["resblock_dilation_sizes"]):
            index = i * len(config["resblock_kernel_sizes"]) + j
            modules.extend(
                (f"resblocks.{index}.convs{part}.{k}", f"resblocks.{index}.convs{part}.{k}")
                for part in (1, 2)
                for k in range(len(dilations))
            )
    mapping = {f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")}
    rules = ()
    if config["normalize_before"]:
        mapping.update({"mean": "mean", "scale": "scale"})
    else:
        del mapping["conv_pre.weight"]
        rules = (
            Rule(
                ("conv_pre.weight",),
                ("conv_pre.weight", "mean", "scale"),
                WithConstants((torch.zeros(config["model_in_dim"]), torch.ones(config["model_in_dim"]))),
            ),
        )
    return Conversion(mapping=mapping, rules=rules)
