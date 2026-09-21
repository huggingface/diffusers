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
from .transforms import Reshape


def oobleck_conversion(config):
    mapping, rules, convolutions, snakes = {}, [], [], []
    count = len(config["downsampling_ratios"])
    multiples = [1] + list(config["channel_multiples"])
    for component in ("encoder", "decoder"):
        base = config["encoder_hidden_size"] if component == "encoder" else config["decoder_channels"]
        convolutions.extend(
            [
                (f"{component}.layers.0", f"{component}.conv1", True),
                (f"{component}.layers.{count + 2}", f"{component}.conv2", component == "encoder"),
            ]
        )
        snakes.append(
            (
                f"{component}.layers.{count + 1}",
                f"{component}.snake1",
                base * (multiples[-1] if component == "encoder" else 1),
            )
        )
        for i in range(count):
            old, new = f"{component}.layers.{i + 1}", f"{component}.block.{i}"
            encoder = component == "encoder"
            input_dim = base * multiples[i if encoder else count - i]
            unit_dim = input_dim if encoder else base * multiples[count - i - 1]
            snakes.append((old + (".layers.3" if encoder else ".layers.0"), new + ".snake1", input_dim))
            convolutions.append(
                (old + (".layers.4" if encoder else ".layers.1"), new + (".conv1" if encoder else ".conv_t1"), True)
            )
            for j in range(3):
                a, b = f"{old}.layers.{j if encoder else j + 2}", f"{new}.res_unit{j + 1}"
                snakes.extend([(a + ".layers.0", b + ".snake1", unit_dim), (a + ".layers.2", b + ".snake2", unit_dim)])
                convolutions.extend([(a + ".layers.1", b + ".conv1", True), (a + ".layers.3", b + ".conv2", True)])
    for old, new, bias in convolutions:
        mapping.update(
            {
                f"{old}.{p}": f"{new}.{p}"
                for p in (("weight_g", "weight_v", "bias") if bias else ("weight_g", "weight_v"))
            }
        )
    for old, new, channels in snakes:
        for p in ("alpha", "beta"):
            rules.append(Rule((f"{old}.{p}",), (f"{new}.{p}",), Reshape((channels,), (1, channels, 1))))
    return Conversion(mapping=mapping, rules=tuple(rules))
