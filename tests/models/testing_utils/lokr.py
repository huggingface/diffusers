# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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
import torch
import torch.nn as nn

from diffusers.utils.import_utils import is_peft_available

from ...testing_utils import assert_tensors_close, is_lora, require_peft_backend, torch_device
from .common import BaseModelOutputMixin


if is_peft_available():
    from peft.tuners.lokr.layer import LoKrLayer, factorization

    from diffusers.loaders.peft import PeftAdapterMixin


def make_lokr_factors(out_features, in_features, factor=4, rank=None):
    """
    Random LoKr factors for an `out_features x in_features` layer, factorized like peft does with `decompose_factor`.
    With `rank`, the right factor is stored rank-decomposed (`lokr_w2_a @ lokr_w2_b`).

    Returns the factors keyed by their state dict suffix, and the delta weight they encode.
    """
    out_l, out_k = factorization(out_features, factor)
    in_m, in_n = factorization(in_features, factor)
    w1 = torch.randn(out_l, in_m)
    if rank is None:
        w2 = torch.randn(out_k, in_n)
        return {"lokr_w1": w1, "lokr_w2": w2}, torch.kron(w1, w2)
    w2_a, w2_b = torch.randn(out_k, rank), torch.randn(rank, in_n)
    return {"lokr_w1": w1, "lokr_w2_a": w2_a, "lokr_w2_b": w2_b}, torch.kron(w1, w2_a @ w2_b)


def check_lokr_deltas(model, expected_deltas, adapter_name="default", atol=1e-5):
    """Check that exactly the expected modules carry a LoKr adapter, each with the expected delta weight."""
    named_modules = dict(model.named_modules())
    adapted = {name for name, module in named_modules.items() if isinstance(module, LoKrLayer)}
    assert adapted == set(expected_deltas)
    for name, expected in expected_deltas.items():
        delta = named_modules[name].get_delta_weight(adapter_name).cpu()
        assert_tensors_close(delta, expected, atol=atol, rtol=0, msg=f"Wrong LoKr delta on {name}")


@is_lora
@require_peft_backend
class LoKrTesterMixin(BaseModelOutputMixin):
    """
    Mixin class for testing loading LoKr (LyCORIS Kronecker product) adapters with `load_lora_adapter`.

    Expected from config mixin:
        - model_class: The model class to test

    Expected methods from config mixin:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Pytest mark: lora
        Use `pytest -m "not lora"` to skip these tests
    """

    def setup_method(self):
        if not issubclass(self.model_class, PeftAdapterMixin):
            pytest.skip(f"PEFT is not supported for this model ({self.model_class.__name__}).")

    def _flatten_output(self, output):
        # Some models (e.g. Z-Image) return a list of per-sample tensors.
        if isinstance(output, (list, tuple)):
            return torch.cat([t.flatten() for t in output])
        return output

    def _model_output(self, model, inputs_dict):
        return self._flatten_output(model(**inputs_dict, return_dict=False)[0])

    def get_lokr_state_dict(self, model, rank=None):
        """
        A peft-format LoKr state dict on every attention `to_q` and `to_v`, with the expected delta of each. With
        `rank`, the `to_v` factors are rank-decomposed, so the config inference has to mix both kinds.
        """
        state_dict, expected_deltas = {}, {}
        for name, module in model.named_modules():
            projection = name.rsplit(".", 1)[-1]
            if isinstance(module, nn.Linear) and projection in ("to_q", "to_v"):
                factors, delta = make_lokr_factors(
                    module.out_features, module.in_features, rank=rank if projection == "to_v" else None
                )
                state_dict.update({f"{name}.{suffix}": weight for suffix, weight in factors.items()})
                expected_deltas[name] = delta
        return state_dict, expected_deltas

    @pytest.mark.parametrize("rank", [None, 1], ids=["full_factors", "rank_decomposed_factors"])
    @torch.no_grad()
    def test_lokr_adapter_loads_exact_kronecker_deltas(self, base_model_output, rank):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).eval().to(torch_device)
        state_dict, expected_deltas = self.get_lokr_state_dict(model, rank=rank)

        model.load_lora_adapter(state_dict, prefix=None, adapter_name="default")

        check_lokr_deltas(model, expected_deltas)
        output = self._model_output(model, self.get_dummy_inputs())
        base_output = self._flatten_output(base_model_output)
        assert not torch.allclose(output, base_output, atol=1e-4, rtol=1e-4), "Output should differ with LoKr"

    @torch.no_grad()
    def test_lokr_unload_restores_base_output(self, base_model_output):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).eval().to(torch_device)
        state_dict, _ = self.get_lokr_state_dict(model)

        model.load_lora_adapter(state_dict, prefix=None, adapter_name="default")
        model.unload_lora()

        assert not any(isinstance(module, LoKrLayer) for module in model.modules())
        output = self._model_output(model, self.get_dummy_inputs())
        assert_tensors_close(output, self._flatten_output(base_model_output), atol=1e-4, rtol=1e-4)

    def test_lokr_hotswap_raises(self):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).eval().to(torch_device)
        state_dict, _ = self.get_lokr_state_dict(model)
        model.load_lora_adapter(state_dict, prefix=None, adapter_name="default")

        with pytest.raises(ValueError, match="Hotswapping LoKr adapters is not supported"):
            model.load_lora_adapter(state_dict, prefix=None, adapter_name="default", hotswap=True)
