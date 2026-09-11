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
from .unet_3d import unet_3d_conversion


def i2vgen_xl_conversion(config):
    base = unet_3d_conversion(config)
    mapping, rules = dict(base.mapping), list(base.rules)
    modules = []
    for old, new, indices in (
        ("local_image_concat", "image_latents_proj_in", (0, 2, 4)),
        ("local_image_embedding", "image_latents_context_embedding", (0, 3, 5)),
        ("context_embedding", "context_embedding", (0, 2)),
        ("fps_embedding", "fps_embedding", (0, 2)),
    ):
        modules.extend((f"{old}.{i}", f"{new}.{i}") for i in indices)
    old, new = "local_temporal_encoder.layers.0", "image_latents_temporal_encoder"
    modules.extend(
        [
            (old + ".0.norm", new + ".norm1"),
            (old + ".0.fn.to_out.0", new + ".attn1.to_out.0"),
            (old + ".1.net.0.0", new + ".ff.net.0.proj"),
            (old + ".1.net.2", new + ".ff.net.2"),
        ]
    )
    rules.append(
        Rule(
            (old + ".0.fn.to_qkv.weight",),
            tuple(f"{new}.attn1.to_{part}.weight" for part in ("q", "k", "v")),
            Split((2 * config["in_channels"],) * 3),
        )
    )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
