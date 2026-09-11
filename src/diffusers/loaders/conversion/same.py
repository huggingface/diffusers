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
from .transforms import ReorderChunks


def same_conversion(config):
    mapping = {
        "bottleneck.scaling_factor": "bottleneck.scale",
        "bottleneck.bias": "bottleneck.bias",
        "bottleneck.running_std": "bottleneck.running_std",
    }
    count = len(config["encoder_strides"])
    modules = [(f"encoder.layers.{count + 1}", "encoder.proj"), ("decoder.layers.1", "decoder.proj")]
    channels = [config["audio_channels"] * config["patch_size"]] + [
        config["encoder_channels"] * c for c in config["encoder_c_mults"]
    ]
    rules = []
    for component in ("encoder", "decoder"):
        for i in range(count):
            level = i if component == "encoder" else count - 1 - i
            old, new = f"{component}.layers.{i if component == 'encoder' else i + 3}", f"{component}.blocks.{i}"
            mapping[old + ".new_tokens"] = new + ".new_tokens"
            if channels[level] != channels[level + 1]:
                mapping.update({f"{old}.mapping.{p}": f"{new}.mapping.{p}" for p in ("weight_g", "weight_v", "bias")})
            for j in range(config["encoder_transformer_depths"][level]):
                a, b = f"{old}.transformers.{j}", f"{new}.transformers.{j}"
                for source, target in (
                    ("pre_norm", "norm_attn"),
                    ("ff_norm", "norm_ff"),
                    ("self_attn.q_norm", "attn.q_norm"),
                    ("self_attn.k_norm", "attn.k_norm"),
                ):
                    mapping.update({f"{a}.{source}.{p}": f"{b}.{target}.{p}" for p in ("alpha", "gamma", "beta")})
                source, target = a + ".self_attn.to_qkv.weight", b + ".attn.to_qkv.weight"
                if config["use_differential_attention"]:
                    rules.append(Rule((source,), (target,), ReorderChunks((0, 3, 1, 4, 2))))
                else:
                    mapping[source] = target
                mapping[a + ".self_attn.to_out.weight"] = b + ".attn.to_out.weight"
                modules.extend([(a + ".ff.ff.0.proj", b + ".ff.proj_in"), (a + ".ff.ff.2", b + ".ff.proj_out")])
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
