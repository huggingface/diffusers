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


from .core import Conversion, Rule
from .transforms import Split


def stable_cascade_conversion(config):
    modules = [("embedding.1", "embedding.1"), ("clf.1", "clf.1")]
    pooled = "clip_txt_pooled_mapper" if config["clip_text_in_channels"] is not None else "clip_mapper"
    modules.append((pooled, "clip_txt_pooled_mapper"))
    for field, name in (("clip_text_in_channels", "clip_txt_mapper"), ("clip_image_in_channels", "clip_img_mapper")):
        if config[field] is not None:
            modules.append((name, name))
    for field, name in (("effnet_in_channels", "effnet_mapper"), ("pixel_mapper_in_channels", "pixels_mapper")):
        if config[field] is not None:
            modules.extend((f"{name}.{i}", f"{name}.{i}") for i in (0, 2))
    mapping, rules = {}, []
    count = len(config["block_out_channels"])
    for direction in ("down", "up"):
        for i in range(count):
            config_index = i if direction == "down" else count - 1 - i
            channels = config["block_out_channels"][config_index]
            if (direction == "down" and i > 0) or (direction == "up" and i < count - 1):
                name = f"{direction}_{'downscalers' if direction == 'down' else 'upscalers'}.{i}.1"
                if config["switch_level"] is not None:
                    name += ".blocks.0" if direction == "down" else ".blocks.1"
                modules.append((name, name))
            repeats = config[f"{direction}_blocks_repeat_mappers"]
            for j in range(repeats[i] - 1):
                name = f"{direction}_repeat_mappers.{i}.{j}"
                modules.append((name, name))
            block_types = config["block_types_per_layer"][config_index]
            for j in range(config[f"{direction}_num_layers_per_block"][i]):
                for k, block_type in enumerate(block_types):
                    prefix = f"{direction}_blocks.{i}.{j * len(block_types) + k}"
                    if block_type == "SDCascadeResBlock":
                        modules.extend(
                            (f"{prefix}.{name}", f"{prefix}.{name}")
                            for name in ("depthwise", "channelwise.0", "channelwise.4")
                        )
                        for name in ("gamma", "beta"):
                            key = f"{prefix}.channelwise.2.{name}"
                            mapping[key] = key
                    elif block_type == "SDCascadeTimestepBlock":
                        for name in ["mapper"] + [f"mapper_{cond}" for cond in config["timestep_conditioning_type"]]:
                            modules.append((f"{prefix}.{name}", f"{prefix}.{name}"))
                    elif block_type == "SDCascadeAttnBlock":
                        modules.extend(
                            [
                                (prefix + ".kv_mapper.1", prefix + ".kv_mapper.1"),
                                (prefix + ".attention.attn.out_proj", prefix + ".attention.to_out.0"),
                            ]
                        )
                        for p in ("weight", "bias"):
                            rules.append(
                                Rule(
                                    (f"{prefix}.attention.attn.in_proj_{p}",),
                                    tuple(f"{prefix}.attention.to_{part}.{p}" for part in ("q", "k", "v")),
                                    Split((channels,) * 3),
                                )
                            )
                    else:
                        raise ValueError(f"Unknown Stable Cascade block type {block_type}.")
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
