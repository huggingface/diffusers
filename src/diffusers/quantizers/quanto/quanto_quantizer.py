from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

from ...utils import (
    get_module_from_name,
    is_accelerate_available,
    is_accelerate_version,
    is_optimum_quanto_available,
    is_optimum_quanto_version,
    logging,
)
from ..base import DiffusersQuantizer


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin


if is_accelerate_available():
    from accelerate.utils import CustomDtype, set_module_tensor_to_device

if is_optimum_quanto_available():
    from .utils import _replace_with_quanto_layers

logger = logging.get_logger(__name__)


class QuantoQuantizer(DiffusersQuantizer):
    r"""
    Diffusers Quantizer for Optimum Quanto
    """

    use_keep_in_fp32_modules = True
    requires_calibration = False
    required_packages = ["quanto", "accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        if not is_optimum_quanto_available():
            raise ImportError(
                "Loading an optimum-quanto quantized model requires optimum-quanto library (`pip install optimum-quanto`)"
            )
        if not is_optimum_quanto_version(">=", "0.2.6"):
            raise RuntimeError(
                "The minimum required version of `optimum-quanto` is 0.2.6. Please upgrade with `pip install -U optimum-quanto`."
            )
        if not is_accelerate_available():
            raise ImportError(
                "Loading an optimum-quanto quantized model requires accelerate library (`pip install accelerate`)"
            )

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ):
        # Quanto imports diffusers internally. This is here to prevent circular imports
        from optimum.quanto import QModuleMixin

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, QModuleMixin) and "weight" in tensor_name:
            return not module.frozen

        return False

    def create_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        *args,
        **kwargs,
    ):
        """
        Create the quantized parameter by calling .freeze() after setting it to the module.
        """
        dtype = kwargs.get("dtype", torch.float32)

        set_module_tensor_to_device(model, param_name, target_device, param_value, dtype)
        module, _ = get_module_from_name(model, param_name)
        module.freeze()
        module.weight.requires_grad = False

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        if is_accelerate_version(">=0.27.0"):
            mapping = {
                "int8": torch.int8,
                "float8": CustomDtype.FP8,
                "int4": CustomDtype.INT4,
                "int2": CustomDtype.INT2,
            }
            target_dtype = mapping[self.quantization_config.weights]
            return target_dtype
        else:
            raise ValueError(
                "You are using `device_map='auto'` on an optimum-quanto quantized model. To automatically compute"
                " the appropriate device map, you should upgrade your `accelerate` library,"
                "`pip install --upgrade accelerate` or install it from source."
            )

    def update_torch_dtype(self, torch_dtype: "torch.dtype" = None) -> "torch.dtype":
        if torch_dtype is None:
            logger.info("You did not specify `torch_dtype` in `from_pretrained`. Setting it to `torch.float32`.")
            torch_dtype = torch.float32
        return torch_dtype

    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        # Quanto imports diffusers internally. This is here to prevent circular imports
        from optimum.quanto import QModuleMixin

        not_missing_keys = []
        for name, module in model.named_modules():
            if isinstance(module, QModuleMixin):
                for missing in missing_keys:
                    if (
                        (name in missing or name in f"{prefix}.{missing}")
                        and not missing.endswith(".weight")
                        and not missing.endswith(".bias")
                    ):
                        not_missing_keys.append(missing)
        return [k for k in missing_keys if k not in not_missing_keys]

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        self.modules_to_not_convert = self.quantization_config.modules_to_not_convert

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        self.modules_to_not_convert.extend(keep_in_fp32_modules)

        model = _replace_with_quanto_layers(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model, **kwargs):
        return model

    def _dequantize(self, model):
        logger.warning("Dequantizing the full model is currently not supported with the Quanto backend")
        return

    @property
    def is_trainable(self, model: Optional["ModelMixin"] = None):
        return True

    def is_serializable(self, safe_serialization=None):
        return False
