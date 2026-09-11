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


def anyflow_conversion(config):
    base = wan_conversion(
        {**config, "qk_norm": "rms_norm_across_heads", "added_kv_proj_dim": None, "pos_embed_seq_len": None}
    )
    keys = set(base.diffusers_keys)
    keys.update(f"condition_embedder.delta_embedder.linear_{i}.{p}" for i in (1, 2) for p in ("weight", "bias"))
    return Conversion(mapping={key: key for key in sorted(keys)})
