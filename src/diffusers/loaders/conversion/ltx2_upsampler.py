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


def ltx2_upsampler_conversion(config):
    modules = ["initial_conv", "initial_norm", "final_conv"]
    modules.extend(
        f"{group}.{i}.{name}"
        for group in ("res_blocks", "post_upsample_res_blocks")
        for i in range(config["num_blocks_per_stage"])
        for name in ("conv1", "conv2", "norm1", "norm2")
    )
    rational = config["spatial_upsample"] and not config["temporal_upsample"] and config["use_rational_resampler"]
    modules.append("upsampler.conv" if rational else "upsampler.0")
    mapping = {f"{name}.{p}": f"{name}.{p}" for name in modules for p in ("weight", "bias")}
    if rational:
        mapping["upsampler.blur_down.kernel"] = "upsampler.blur_down.kernel"
    return Conversion(mapping=mapping)
