from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ...utils import get_module_from_name
from ..base import DiffusersQuantizer
from .utils import GGUFParameter, _quant_shape_from_byte_shape, _replace_with_gguf_linear


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin

from ...utils import (
    is_gguf_available,
    is_torch_available,
    logging,
)


if is_torch_available():
    import torch

if is_gguf_available():
    import gguf

logger = logging.get_logger(__name__)


class GGUFQuantizer(DiffusersQuantizer):
    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        self.compute_dtype = quantization_config.compute_dtype
        self.pre_quantized = True

    def check_quantized_param_shape(self, param_name, current_param, loaded_param):
        loaded_param_shape = loaded_param.shape
        current_param_shape = current_param.shape
        quant_type = loaded_param.quant_type

        block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]

        inferred_shape = _quant_shape_from_byte_shape(loaded_param_shape, type_size, block_size)
        if inferred_shape != current_param_shape:
            raise ValueError(
                f"{param_name} has an expected quantized shape of: {inferred_shape}, but receieved shape: {loaded_param_shape}"
            )

        return True

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        if isinstance(param_value, GGUFParameter):
            return True

        return False

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

        module._parameters[tensor_name] = param_value

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        _replace_with_gguf_linear(model, self.compute_dtype)

    def _process_model_after_weight_loading(self, model: "ModelMixin", **kwargs):
        return model

    @property
    def is_serializable(self):
        return False

    @property
    def is_trainable(self) -> bool:
        # Because we're mandating `bitsandbytes` 0.43.3.
        return False
