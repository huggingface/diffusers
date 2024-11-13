from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ...utils import get_module_from_name
from ..base import DiffusersQuantizer
from .utils import _replace_with_gguf_linear


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin

from ...utils import (
    is_accelerate_available,
    is_torch_available,
    logging,
)


if is_accelerate_available():
    pass

if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class GGUFQuantizer(DiffusersQuantizer):
    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        self.quant_type = quantization_config.quant_type
        self.compute_dtype = quantization_config.compute_dtype
        self.qtypes = quantization_config.qtypes
        self.pre_quantized = True

    def check_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        return True

    def check_quantized_param_shape(self, param_name, current_param_shape, loaded_param_shape):
        return True

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        return True

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

        if param_name == "transformer_blocks.0.attn.to_q.weight":
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
        model = _replace_with_gguf_linear(model, self.compute_dtype, self.quant_type)

    def _process_model_after_weight_loading(self, model: "ModelMixin", **kwargs):
        return model

    @property
    def is_serializable(self):
        return False

    @property
    def is_trainable(self) -> bool:
        # Because we're mandating `bitsandbytes` 0.43.3.
        return False
