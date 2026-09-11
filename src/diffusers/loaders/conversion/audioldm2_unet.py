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
from .ldm_unet import ldm_unet_conversion


def audioldm2_unet_conversion(config):
    base = ldm_unet_conversion(config)
    mapping = {}
    dimensions = config["cross_attention_dim"]
    if isinstance(dimensions, int):
        dimensions = [dimensions] * len(config["block_out_channels"])
    counts = [len(value) if isinstance(value, (tuple, list)) else 1 for value in dimensions]
    for old, new in base.mapping.items():
        if config["norm_num_groups"] is None and new.startswith("conv_norm_out."):
            continue
        if ".attentions." in new:
            target_prefix, suffix = new.split(".attentions.", 1)
            index, suffix = suffix.split(".", 1)
            if target_prefix == "mid_block":
                count = counts[-1]
                source_prefix, source_suffix = old.split(".", 2)[0], old.split(".", 2)[2]
                for j in range(count):
                    mapping[f"{source_prefix}.{j + 1}.{source_suffix}"] = f"mid_block.attentions.{j}.{suffix}"
            else:
                block_index = int(target_prefix.split(".")[1])
                if target_prefix.startswith("up_blocks"):
                    block_index = len(counts) - 1 - block_index
                count = counts[block_index]
                source_group, source_index, _, source_suffix = old.split(".", 3)
                for j in range(count):
                    mapping[f"{source_group}.{source_index}.{j + 1}.{source_suffix}"] = (
                        f"{target_prefix}.attentions.{int(index) * count + j}.{suffix}"
                    )
        elif new.startswith("mid_block.resnets.1."):
            mapping[old.replace("middle_block.2.", f"middle_block.{counts[-1] + 1}.", 1)] = new
        elif new.startswith("up_blocks.") and ".upsamplers." in new:
            i = int(new.split(".")[1])
            if config["up_block_types"][i] == "CrossAttnUpBlock2D":
                a, b, _, suffix = old.split(".", 3)
                old = f"{a}.{b}.{counts[len(counts) - 1 - i] + 1}.{suffix}"
            mapping[old] = new
        else:
            mapping[old] = new
    return Conversion(mapping=mapping)
