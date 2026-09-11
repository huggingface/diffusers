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


def minimax_music3_vocoder_conversion(config):
    count = len(config["upsampling_ratios"])
    mapping = {
        "dec_in_proj.weight": "dec_in_proj.weight",
        "dec_in_proj.bias": "dec_in_proj.bias",
        f"decoder.model.{count + 1}.alpha": "snake_out.alpha",
    }
    convolutions = [("decoder.model.0", "conv_in"), (f"decoder.model.{count + 2}", "conv_out")]
    for i in range(count):
        old, new = f"decoder.model.{i + 1}.block", f"blocks.{i}"
        mapping[old + ".0.alpha"] = new + ".snake1.alpha"
        convolutions.append((old + ".1", new + ".conv_t1"))
        for j in range(3):
            a, b = f"{old}.{j + 2}.block", f"{new}.res_unit{j + 1}"
            mapping[a + ".0.alpha"] = b + ".snake1.alpha"
            mapping[a + ".2.alpha"] = b + ".snake2.alpha"
            convolutions.extend([(a + ".1", b + ".conv1"), (a + ".3", b + ".conv2")])
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in convolutions for p in ("weight_g", "weight_v", "bias")})
    return Conversion(mapping=mapping)
