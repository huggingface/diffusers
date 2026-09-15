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


def tiny_vae_conversion(config):
    modules = []
    index = 0
    for i, count in enumerate(config["num_encoder_blocks"]):
        modules.append((f"encoder.{index}", f"encoder.layers.{index}", i == 0))
        index += 1
        for _ in range(count):
            modules.extend((f"encoder.{index}.conv.{j}", f"encoder.layers.{index}.conv.{j}", True) for j in (0, 2, 4))
            index += 1
    modules.append((f"encoder.{index}", f"encoder.layers.{index}", True))
    modules.append(("decoder.1", "decoder.layers.0", True))
    index = 2
    for i, count in enumerate(config["num_decoder_blocks"]):
        for _ in range(count):
            modules.extend(
                (f"decoder.{index + 1}.conv.{j}", f"decoder.layers.{index}.conv.{j}", True) for j in (0, 2, 4)
            )
            index += 1
        final = i == len(config["num_decoder_blocks"]) - 1
        if not final:
            index += 1
        modules.append((f"decoder.{index + 1}", f"decoder.layers.{index}", final))
        index += 1
    return Conversion(
        mapping={
            f"{old}.{p}": f"{new}.{p}"
            for old, new, bias in modules
            for p in (("weight", "bias") if bias else ("weight",))
        }
    )
