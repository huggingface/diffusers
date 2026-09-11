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
from .spectrogram_notes import _spectrogram_encoder_rules
from .transforms import Permute


def spectrogram_continuous_conversion(config):
    mapping, rules = _spectrogram_encoder_rules(config)
    return Conversion(
        mapping=mapping, rules=rules + (Rule(("input_proj.kernel",), ("input_proj.weight",), Permute((1, 0))),)
    )
