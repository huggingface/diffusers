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


def versatile_text_unet_conversion(config):
    spatial_config = {
        **config,
        "down_block_types": [kind.replace("Flat", "2D") for kind in config["down_block_types"]],
        "up_block_types": [kind.replace("Flat", "2D") for kind in config["up_block_types"]],
    }
    base = _ldm_unet_mapping(spatial_config, controlnet=False)
    mapping = {}
    for old, new in base.items():
        if ".downsamplers." in new:
            old = old.replace(".0.op.", ".0.")
            new = new.replace(".downsamplers.0.conv.", ".downsamplers.0.")
        elif ".upsamplers." in new:
            old = old.replace(".conv.", ".")
            new = new.replace(".upsamplers.0.conv.", ".upsamplers.0.")
        prefix = "model.diffusion_model." if old.startswith("time_embed.") else "model.diffusion_model.unet_text."
        mapping[prefix + old] = new
    return Conversion(mapping=mapping)
