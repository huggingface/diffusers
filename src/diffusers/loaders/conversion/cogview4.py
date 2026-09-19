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


from .cogview3plus import cogview3plus_conversion
from .core import Conversion


def cogview4_conversion(config):
    if config.get("original_format") == "megatron":
        from .cogview4_megatron import cogview4_megatron_conversion

        return cogview4_megatron_conversion(config)
    # The SAT parameter layout is shared with CogView3Plus; the configuration selects the dimensions and depth.
    base = cogview3plus_conversion(config)
    return Conversion(mapping=base.mapping, rules=base.rules)
