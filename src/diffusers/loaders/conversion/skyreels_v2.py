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
from .wan import wan_conversion


def skyreels_v2_conversion(config):
    base = wan_conversion(config)
    mapping = dict(base.mapping)
    if config["inject_sample_info"]:
        mapping["fps_embedding.weight"] = "fps_embedding.weight"
        mapping.update(
            {
                f"fps_projection.{i}.{p}": f"fps_projection.{name}.{p}"
                for i, name in ((0, "net.0.proj"), (2, "net.2"))
                for p in ("weight", "bias")
            }
        )
    return Conversion(mapping=mapping, rules=base.rules)
