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
from .ldm_unet import _ldm_unet_mapping


def gligen_conversion(config):
    mapping = _ldm_unet_mapping(config, controlnet=False)
    blocks = [
        (old.removesuffix(".norm1.weight"), new.removesuffix(".norm1.weight"))
        for old, new in mapping.items()
        if ".transformer_blocks." in new and new.endswith(".norm1.weight")
    ]
    for old, new in blocks:
        for name in ("linear", "norm1", "norm2", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"):
            mapping.update({f"{old}.fuser.{name}.{p}": f"{new}.fuser.{name}.{p}" for p in ("weight", "bias")})
        for part in ("q", "k", "v"):
            mapping[f"{old}.fuser.attn.to_{part}.weight"] = f"{new}.fuser.attn.to_{part}.weight"
        for name in ("alpha_attn", "alpha_dense"):
            mapping[f"{old}.fuser.{name}"] = f"{new}.fuser.{name}"
    image = config["attention_type"] == "gated-text-image"
    names = ("linears_text", "linears_image") if image else ("linears",)
    for name in names:
        for i in (0, 2, 4):
            for p in ("weight", "bias"):
                key = f"position_net.{name}.{i}.{p}"
                mapping[key] = key
    for name in (("null_text_feature", "null_image_feature") if image else ("null_positive_feature",)) + (
        "null_position_feature",
    ):
        mapping[f"position_net.{name}"] = f"position_net.{name}"
    return Conversion(mapping=mapping)
