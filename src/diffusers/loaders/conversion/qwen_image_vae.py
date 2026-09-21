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
from .wan_vae import wan_vae_conversion


def qwen_image_vae_conversion(config):
    # Qwen-Image retains Wan 2.1's parameter layout; its execution and input channels differ.
    base = wan_vae_conversion({**config, "is_residual": False, "decoder_base_dim": None})
    return Conversion(mapping=base.mapping)
