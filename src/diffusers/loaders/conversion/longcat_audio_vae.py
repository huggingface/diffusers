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


def longcat_audio_vae_conversion(config):
    count = len(config["c_mults"] or (1, 2, 4, 8, 16))
    snake = config["act_fn"] == "snake" if config["act_fn"] is not None else config["use_snake"] is not False
    convolutions = [
        ("encoder.layers.0", True),
        (f"encoder.layers.{count + 1}", True),
        ("decoder.layers.0", True),
        (f"decoder.layers.{count + 2}", False),
    ]
    snakes = [f"decoder.layers.{count + 1}"]
    for component in ("encoder", "decoder"):
        encoder = component == "encoder"
        for i in range(count):
            prefix = f"{component}.layers.{i + 1}.layers"
            snakes.append(prefix + (".3" if encoder else ".0"))
            convolutions.append((prefix + (".4" if encoder else ".1"), True))
            for j in range(3):
                unit = f"{prefix}.{j if encoder else j + 2}.layers"
                snakes.extend([unit + ".0", unit + ".2"])
                convolutions.extend([(unit + ".1", True), (unit + ".3", True)])
    keys = [
        f"{name}.{p}"
        for name, bias in convolutions
        for p in (("weight_g", "weight_v", "bias") if bias else ("weight_g", "weight_v"))
    ]
    if snake:
        keys.extend(f"{name}.{p}" for name in snakes for p in ("alpha", "beta"))
    return Conversion(mapping={key: key for key in keys})
