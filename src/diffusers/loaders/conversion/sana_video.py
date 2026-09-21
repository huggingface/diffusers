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
from .sana import sana_conversion


def sana_video_conversion(config):
    base = sana_conversion(config)
    mapping = {old: new.replace("patch_embed.proj.", "patch_embedding.") for old, new in base.mapping.items()}
    mapping.update(
        {
            f"blocks.{i}.mlp.t_conv.weight": f"transformer_blocks.{i}.ff.conv_temp.weight"
            for i in range(config["num_layers"])
        }
    )
    return Conversion(mapping=mapping, rules=base.rules)
