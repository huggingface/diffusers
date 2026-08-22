# coding=utf-8
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


import pytest

from diffusers import NVIDIAModelOptConfig
from diffusers.utils import is_nvidia_modelopt_available

from ...testing_utils import (
    is_modelopt,
    is_quantization,
    require_modelopt_version_greater_or_equal,
)


if is_nvidia_modelopt_available():
    import modelopt.torch.quantization as mtq


@is_quantization
@is_modelopt
@require_modelopt_version_greater_or_equal("0.33.1")
class TestNVIDIAModelOptConfigQuantCfg:
    """CPU-only regression tests for `NVIDIAModelOptConfig.get_config_from_quant_type`.

    These only build the config (no model, no accelerator). Diffusers' contract is simply to build a
    `modelopt_config` the installed ModelOpt accepts; the `quant_cfg` container shape (a mapping vs a
    list of entries, which changed in ModelOpt 0.44) is ModelOpt's concern, so it is not asserted
    here. This guards the regression where building the config raised `TypeError: 'list' object is
    not a mapping` on `NVIDIAModelOptConfig(quant_type=...)`.
    """

    def _weight_quantizer_num_bits(self, quant_cfg):
        # `quant_cfg` is a `{pattern: cfg}` mapping or a list of `{"quantizer_name": ...}` entries.
        if isinstance(quant_cfg, dict):
            return quant_cfg["*weight_quantizer"].get("num_bits")
        entry = next(e for e in quant_cfg if e["quantizer_name"] == "*weight_quantizer")
        return entry.get("cfg", {}).get("num_bits")

    @pytest.mark.parametrize(
        "init_kwargs, weight_num_bits",
        [
            ({"quant_type": "FP8"}, (4, 3)),
            ({"quant_type": "INT8"}, 8),
            ({"quant_type": "FP8_FP8"}, (4, 3)),
            ({"quant_type": "FP8", "weight_only": False}, (4, 3)),
            (
                {
                    "quant_type": "NVFP4",
                    "block_quantize": 128,
                    "channel_quantize": -1,
                    "scale_block_quantize": 8,
                    "scale_channel_quantize": -1,
                    "modules_to_not_convert": ["conv"],
                },
                (2, 1),
            ),
            (
                {
                    "quant_type": "INT4",
                    "block_quantize": 128,
                    "channel_quantize": -1,
                    "disable_conv_quantization": True,
                },
                4,
            ),
        ],
    )
    def test_quant_cfg_is_accepted_by_modelopt(self, init_kwargs, weight_num_bits):
        modelopt_config = NVIDIAModelOptConfig(**init_kwargs).modelopt_config
        # Diffusers builds the weight quantizer at the requested bit-width ...
        assert self._weight_quantizer_num_bits(modelopt_config["quant_cfg"]) == weight_num_bits
        # ... and the whole config must be consumable by the installed ModelOpt (what `mto.apply_mode`
        # relies on), whatever `quant_cfg` container shape that ModelOpt uses.
        mtq.config.QuantizeConfig(**modelopt_config)
