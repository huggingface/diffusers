from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union


from ...utils import get_module_from_name
from ..base import DiffusersQuantizer
from .utils import GGUFLinear

if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin

from ...utils import (
    is_accelerate_available,
    is_accelerate_version,
    is_torch_available,
    logging,
)

if is_accelerate_available():
    from accelerate import init_empty_weights

if is_torch_available():
    import torch
    import torch.nn as nn


logger = logging.get_logger(__name__)


class GGUFQuantizer(DiffusersQuantizer):
    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        self.quant_type = quantization_config.quant_type
        self.compute_dtype = quantization_config.compute_dtype

    def check_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        return

    def create_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: Optional[List[str]] = None,
    ):
        module, tensor_name = get_module_from_name(model, param_name)
        if tensor_name not in module._parameters:
            raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
        __import__("ipdb").set_trace()

        module._parameters[tensor_name] = param_value

        return

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        for name, module in model.named_children():
            if isinstance(module, nn.Linear) and name not in self.modules_to_not_convert:
                with init_empty_weights():
                    model._modules[name] = GGUFLinear(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                        compute_dtype=self.compute_dtype,
                        quant_type=self.quant_type,
                    )
