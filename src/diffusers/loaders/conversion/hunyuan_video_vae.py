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

from .core import Conversion
from .ldm_vae import ldm_vae_conversion


def hunyuan_video_vae_conversion(config):
    base = ldm_vae_conversion(config)
    keys = []
    for key in sorted(base.diffusers_keys):
        if ".downsamplers." in key or ".upsamplers." in key:
            continue
        parts = key.split(".")
        if parts[-2] in ("conv_in", "conv_out", "conv1", "conv2", "conv_shortcut"):
            parts.insert(-1, "conv")
        keys.append(".".join(parts))
    count = len(config["block_out_channels"])
    spatial = int(math.log2(config["spatial_compression_ratio"]))
    temporal = int(math.log2(config["temporal_compression_ratio"]))
    for i in range(count):
        resampling = i < spatial or (i >= count - 1 - temporal and i < count - 1)
        if resampling:
            keys.extend(
                f"{component}.{direction}_blocks.{i}.{sampler}.0.conv.conv.{p}"
                for component, direction, sampler in (
                    ("encoder", "down", "downsamplers"),
                    ("decoder", "up", "upsamplers"),
                )
                for p in ("weight", "bias")
            )
    return Conversion(mapping={key: key for key in keys})
