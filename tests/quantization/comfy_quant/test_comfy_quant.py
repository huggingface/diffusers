import pytest
import torch
import torch.nn as nn

from diffusers import ComfyQuantConfig
from diffusers.utils import is_comfy_kitchen_available

from ...testing_utils import require_torch


if is_comfy_kitchen_available():
    import comfy_kitchen.tensor as ck_tensor

    from diffusers.quantizers.comfy_quant.comfy_quantizer import ComfyQuantizer

device = "cuda" if torch.cuda.is_available() else "cpu"


@require_torch
@pytest.mark.skipif(not is_comfy_kitchen_available(), reason="comfy-kitchen is not available")
class TestComfyQuantizer:
    def test_create_quantized_param_fp8(self):
        config = ComfyQuantConfig(quant_format="fp8")
        quantizer = ComfyQuantizer(config)

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16)

        model = DummyModel()
        param_value = torch.randn(16, 16, dtype=torch.float32)

        quantizer.create_quantized_param(
            model=model,
            param_value=param_value,
            param_name="linear.weight",
            target_device=torch.device(device),
        )

        assert isinstance(model.linear.weight, ck_tensor.QuantizedTensor)
        assert model.linear.weight._layout_cls == "TensorCoreFP8Layout"

    def test_create_quantized_param_invalid_format(self):
        config = ComfyQuantConfig(quant_format="non_existent_format")
        quantizer = ComfyQuantizer(config)

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16)

        model = DummyModel()
        param_value = torch.randn(16, 16, dtype=torch.float32)

        with pytest.raises(ValueError, match="not found in `comfy_kitchen`"):
            quantizer.create_quantized_param(
                model=model,
                param_value=param_value,
                param_name="linear.weight",
                target_device=torch.device(device),
            )
