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

from .core import Conversion, Rule
from .transforms import Reshape, Split
from .wan import wan_conversion


def wan_animate_conversion(config):
    mapping = dict(wan_conversion(config).mapping)
    modules = [("pose_patch_embedding", "pose_patch_embedding"), ("face_encoder.out_proj", "face_encoder.out_proj")]
    mapping["face_encoder.padding_tokens"] = "face_encoder.padding_tokens"
    mapping["motion_encoder.dec.direction.weight"] = "motion_encoder.motion_synthesis_weight"
    mapping["motion_encoder.enc.net_app.convs.0.0.weight"] = "motion_encoder.conv_in.weight"
    channels = config["motion_encoder_channel_sizes"] or {
        "4": 512,
        "8": 512,
        "16": 512,
        "32": 512,
        "64": 256,
        "128": 128,
        "256": 64,
        "512": 32,
        "1024": 16,
    }
    size = config["motion_encoder_size"]
    count = int(math.log2(size)) - 2
    rules = [
        Rule(
            ("motion_encoder.enc.net_app.convs.0.1.bias",),
            ("motion_encoder.conv_in.act_fn.bias",),
            Reshape((1, channels[str(size)], 1, 1), (channels[str(size)],)),
        )
    ]
    for i in range(count):
        old, new = f"motion_encoder.enc.net_app.convs.{i + 1}", f"motion_encoder.res_blocks.{i}"
        mapping.update(
            {
                f"{old}.{a}.weight": f"{new}.{b}.weight"
                for a, b in (("conv1.0", "conv1"), ("conv2.1", "conv2"), ("skip.1", "conv_skip"))
            }
        )
        for a, b, channel in (
            ("conv1.1", "conv1", channels[str(size // 2**i)]),
            ("conv2.2", "conv2", channels[str(size // 2 ** (i + 1))]),
        ):
            rules.append(
                Rule((f"{old}.{a}.bias",), (f"{new}.{b}.act_fn.bias",), Reshape((1, channel, 1, 1), (channel,)))
            )
    mapping[f"motion_encoder.enc.net_app.convs.{count + 1}.weight"] = "motion_encoder.conv_out.weight"
    modules.extend((f"motion_encoder.enc.fc.{i}", f"motion_encoder.motion_network.{i}") for i in range(5))
    modules.extend((f"face_encoder.{name}.conv", f"face_encoder.{name}") for name in ("conv1_local", "conv2", "conv3"))
    hidden = config["num_attention_heads"] * config["attention_head_dim"]
    for i in range(config["num_layers"] // config["inject_face_latents_blocks"]):
        old, new = f"face_adapter.fuser_blocks.{i}", f"face_adapter.{i}"
        modules.extend([(old + ".linear1_q", new + ".to_q"), (old + ".linear2", new + ".to_out")])
        for p in ("weight", "bias"):
            rules.append(
                Rule((f"{old}.linear1_kv.{p}",), (f"{new}.to_k.{p}", f"{new}.to_v.{p}"), Split((hidden,) * 2))
            )
        mapping.update({f"{old}.{part}_norm.weight": f"{new}.norm_{part}.weight" for part in ("q", "k")})
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
