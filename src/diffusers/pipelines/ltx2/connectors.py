# Copyright 2025 Lightricks and The HuggingFace Team. All rights reserved.
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

from ...models.condition_embedders.text_connector_ltx2 import (
    LTX2ConnectorTransformer1d,  # noqa: F401  re-exported for back-compat
    LTX2RotaryPosEmbed1d,  # noqa: F401
    LTX2TransformerBlock1d,  # noqa: F401
    per_layer_masked_mean_norm,  # noqa: F401
    per_token_rms_norm,  # noqa: F401
)
from ...models.condition_embedders.text_connector_ltx2 import (
    LTX2TextConnectors as _LTX2TextConnectors,
)
from ...utils import deprecate


class LTX2TextConnectors(_LTX2TextConnectors):
    def __init__(self, *args, **kwargs):
        deprecate(
            "LTX2TextConnectors",
            "1.0.0",
            "Importing `LTX2TextConnectors` from `diffusers.pipelines.ltx2.connectors` is deprecated. "
            "Import it from `diffusers.models.condition_embedders` instead "
            "(or `from diffusers import LTX2TextConnectors`).",
        )
        super().__init__(*args, **kwargs)
