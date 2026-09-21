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


def wuerstchen_prior_conversion(config):
    blocks = [
        (f"blocks.{3 * i + j}", kind, config["c"]) for i in range(config["depth"]) for j, kind in enumerate("CTA")
    ]
    mapping, rules = _wuerstchen_block_rules(blocks)
    mapping.update(
        {
            f"{name}.{p}": f"{name}.{p}"
            for name in ("projection", "cond_mapper.0", "cond_mapper.2", "out.1")
            for p in ("weight", "bias")
        }
    )
    return Conversion(mapping=mapping, rules=rules)


def _wuerstchen_block_rules(blocks):
    mapping, modules, rules = {}, [], []
    for prefix, kind, channels in blocks:
        if kind == "C":
            modules.extend(
                (f"{prefix}.{name}", f"{prefix}.{name}") for name in ("depthwise", "channelwise.0", "channelwise.4")
            )
            mapping.update({f"{prefix}.channelwise.2.{p}": f"{prefix}.channelwise.2.{p}" for p in ("gamma", "beta")})
        elif kind == "T":
            modules.append((prefix + ".mapper", prefix + ".mapper"))
        elif kind == "A":
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
            raise ValueError(f"Unknown Wuerstchen block type {kind}.")
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return mapping, tuple(rules)
