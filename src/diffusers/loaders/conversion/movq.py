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

from .core import Conversion
from .ldm_vae import ldm_vae_conversion


def movq_conversion(config):
    base = ldm_vae_conversion(config)
    mapping = {}
    for old, new in base.mapping.items():
        module, _, parameter = new.rpartition(".")
        if new.startswith("decoder.") and module.endswith((".norm1", ".norm2", ".group_norm", ".conv_norm_out")):
            source = old.rsplit(".", 1)[0]
            module = module.removesuffix("group_norm") + "spatial_norm" if module.endswith("group_norm") else module
            for name in ("norm_layer", "conv_y", "conv_b"):
                mapping[f"{source}.{name}.{parameter}"] = f"{module}.{name}.{parameter}"
        else:
            mapping[old] = new
    mapping["quantize.embedding.weight"] = "quantize.embedding.weight"
    return Conversion(mapping=mapping, rules=base.rules)
