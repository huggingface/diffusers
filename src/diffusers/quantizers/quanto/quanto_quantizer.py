from typing import TYPE_CHECKING, Any, Dict, List, Union

from diffusers.utils.import_utils import is_optimum_quanto_version

from ...utils import (
    get_module_from_name,
    is_accelerate_available,
    is_accelerate_version,
    is_optimum_quanto_available,
    is_torch_available,
    logging,
)
from ..base import DiffusersQuantizer


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin


if is_torch_available():
    import torch

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
            raise ImportError(
                "Loading an optimum-quanto quantized model requires `optimum-quanto>=0.2.6`. "
                "Please upgrade your installation with `pip install --upgrade optimum-quanto"
            )

        if not is_accelerate_available():
            raise ImportError(
                "Loading an optimum-quanto quantized model requires accelerate library (`pip install accelerate`)"
            )

        device_map = kwargs.get("device_map", None)
        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            raise ValueError(
                "`device_map` for multi-GPU inference or CPU/disk offload is currently not supported with Diffusers and the Quanto backend"
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
        from optimum.quanto import QModuleMixin, QTensor
        from optimum.quanto.tensor.packed import PackedTensor

        module, tensor_name = get_module_from_name(model, param_name)
        if self.pre_quantized and any(isinstance(module, t) for t in [QTensor, PackedTensor]):
            return True
        elif isinstance(module, QModuleMixin) and "weight" in tensor_name:
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
        module, tensor_name = get_module_from_name(model, param_name)
        if self.pre_quantized:
            setattr(module, tensor_name, param_value)
        else:
            set_module_tensor_to_device(model, param_name, target_device, param_value, dtype)
            module.freeze()
            module.weight.requires_grad = False

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        if is_accelerate_version(">=", "0.27.0"):
            mapping = {
                "int8": torch.int8,
                "float8": CustomDtype.FP8,
                "int4": CustomDtype.INT4,
                "int2": CustomDtype.INT2,
            }
            target_dtype = mapping[self.quantization_config.weights_dtype]

        return target_dtype

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
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model, **kwargs):
        return model

    @property
    def is_trainable(self):
        return True

    @property
    def is_serializable(self):
        return True

    @property
    def is_compileable(self) -> bool:
        return True
