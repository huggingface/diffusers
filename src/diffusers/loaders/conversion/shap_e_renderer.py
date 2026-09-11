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
from .shap_e_tables import create_mc_lookup_table
from .transforms import WithConstants


def shap_e_renderer_conversion(config):
    modules = [(f"renderer.nerstf.mlp.{i}", f"mlp.mlp.{i}") for i in range(config["n_hidden_layers"] + 1)]
    for name in config["param_names"]:
        name = name.replace(".", "__")
        modules.extend(
            (f"encoder.params_proj.projections.{name}.{leaf}", f"params_proj.projections.{name}.{leaf}")
            for leaf in ("proj", "norm")
        )
    mapping = {f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")}
    anchor = "renderer.nerstf.mlp.0.weight"
    target = mapping.pop(anchor)
    cases, masks = create_mc_lookup_table()
    background = torch.tensor(config["background"], device="cpu") / 255.0
    rule = Rule(
        (anchor,),
        (target, "void.background", "mesh_decoder.cases", "mesh_decoder.masks"),
        WithConstants((background, cases, masks)),
    )
    return Conversion(mapping=mapping, rules=(rule,))
