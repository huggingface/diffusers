# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import os
from functools import lru_cache

from ...utils import is_sdnq_available, is_sdnq_version, logging
from ..base import DiffusersQuantizer


logger = logging.get_logger(__name__)

# Recommended sdnq version. Older versions load/save most SDNQ checkpoints fine; only some of the
# latest ones require this version, so a mismatch is a warning, not a hard error.
_RECOMMENDED_SDNQ_VERSION = "0.2.2"


def _check_sdnq_requirement():
    if not is_sdnq_available():
        raise ImportError(
            f"Loading or creating an SDNQ quantized model requires the sdnq library: "
            f"`pip install 'sdnq>={_RECOMMENDED_SDNQ_VERSION}'`"
        )

    if is_sdnq_version("<", _RECOMMENDED_SDNQ_VERSION) and not os.getenv("DIFFUSERS_NO_ADVISORY_WARNINGS"):
        logger.warning(
            f"Your installed sdnq is older than {_RECOMMENDED_SDNQ_VERSION}. Most SDNQ checkpoints will still "
            f"load and save fine, but some of the latest ones require sdnq>={_RECOMMENDED_SDNQ_VERSION}. "
            f"If you hit issues, upgrade with `pip install -U sdnq`."
        )


class SDNQQuantizer(DiffusersQuantizer):
    r"""
    Diffusers Quantizer for SDNQ (https://github.com/Disty0/sdnq).

    The `sdnq` library ships its own `DiffusersQuantizer` subclass; this class is a thin factory that defers to it. It
    only exists so that `quant_method="sdnq"` checkpoints load natively through `DiffusersAutoQuantizer` without
    requiring `import sdnq` beforehand. `sdnq` cannot be imported at module level because it imports
    `diffusers.quantizers.auto` at import time.
    """

    def __new__(cls, quantization_config, **kwargs):
        _check_sdnq_requirement()
        from sdnq import SDNQQuantizer as SDNQLibQuantizer

        return SDNQLibQuantizer(quantization_config, **kwargs)


@lru_cache
def _ensure_sdnq_registered():
    """
    Import sdnq once so it registers itself with transformers (import side effect). transformers loads text encoders
    through its own quantizer registry and only learns about SDNQ once sdnq has been imported; without this it
    silently skips the quantization config of a prequantized SDNQ component and mis-loads the weights.
    """
    import sdnq  # noqa: F401
