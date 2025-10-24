from typing import TYPE_CHECKING, Any, Dict, List, Union

from ...utils import (
    get_module_from_name,
    is_accelerate_available,
    is_nvidia_modelopt_available,
    is_torch_available,
    logging,
)
from ..base import DiffusersQuantizer


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin


if is_torch_available():
    import torch
    import torch.nn as nn

if is_accelerate_available():
    from accelerate.utils import set_module_tensor_to_device


logger = logging.get_logger(__name__)


class NVIDIAModelOptQuantizer(DiffusersQuantizer):
    r"""
    Diffusers Quantizer for TensorRT Model Optimizer
    """

    use_keep_in_fp32_modules = True
    requires_calibration = False
    required_packages = ["nvidia_modelopt"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        if not is_nvidia_modelopt_available():
            raise ImportError(
                "Loading an nvidia-modelopt quantized model requires nvidia-modelopt library (`pip install nvidia-modelopt`)"
            )

        self.offload = False

        device_map = kwargs.get("device_map", None)
        if isinstance(device_map, dict):
            if "cpu" in device_map.values() or "disk" in device_map.values():
                if self.pre_quantized:
                    raise ValueError(
                        "You are attempting to perform cpu/disk offload with a pre-quantized modelopt model "
                        "This is not supported yet. Please remove the CPU or disk device from the `device_map` argument."
                    )
                else:
                    self.offload = True

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ):
        # ModelOpt imports diffusers internally. This is here to prevent circular imports
        from modelopt.torch.quantization.utils import is_quantized

        module, tensor_name = get_module_from_name(model, param_name)
        if self.pre_quantized:
            return True
        elif is_quantized(module) and "weight" in tensor_name:
            return True
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
        Create the quantized parameter by calling .calibrate() after setting it to the module.
        """
        # ModelOpt imports diffusers internally. This is here to prevent circular imports
        import modelopt.torch.quantization as mtq

        dtype = kwargs.get("dtype", torch.float32)
        module, tensor_name = get_module_from_name(model, param_name)
        if self.pre_quantized:
            module._parameters[tensor_name] = torch.nn.Parameter(param_value.to(device=target_device))
        else:
            set_module_tensor_to_device(model, param_name, target_device, param_value, dtype)
            mtq.calibrate(
                module, self.quantization_config.modelopt_config["algorithm"], self.quantization_config.forward_loop
            )
            mtq.compress(module)
            module.weight.requires_grad = False

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        if self.quantization_config.quant_type == "FP8":
            target_dtype = torch.float8_e4m3fn
        return target_dtype

    def update_torch_dtype(self, torch_dtype: "torch.dtype" = None) -> "torch.dtype":
        if torch_dtype is None:
            logger.info("You did not specify `torch_dtype` in `from_pretrained`. Setting it to `torch.float32`.")
            torch_dtype = torch.float32
        return torch_dtype

    def get_conv_param_names(self, model: "ModelMixin") -> List[str]:
        """
        Get parameter names for all convolutional layers in a HuggingFace ModelMixin. Includes Conv1d/2d/3d and
        ConvTranspose1d/2d/3d.
        """
        conv_types = (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        )

        conv_param_names = []
        for name, module in model.named_modules():
            if isinstance(module, conv_types):
                for param_name, _ in module.named_parameters(recurse=False):
                    conv_param_names.append(f"{name}.{param_name}")

        return conv_param_names

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        # ModelOpt imports diffusers internally. This is here to prevent circular imports
        import modelopt.torch.opt as mto

        if self.pre_quantized:
            return

        modules_to_not_convert = self.quantization_config.modules_to_not_convert

        if modules_to_not_convert is None:
            modules_to_not_convert = []
        if isinstance(modules_to_not_convert, str):
            modules_to_not_convert = [modules_to_not_convert]
        modules_to_not_convert.extend(keep_in_fp32_modules)
        if self.quantization_config.disable_conv_quantization:
            modules_to_not_convert.extend(self.get_conv_param_names(model))

        for module in modules_to_not_convert:
            self.quantization_config.modelopt_config["quant_cfg"]["*" + module + "*"] = {"enable": False}
        self.quantization_config.modules_to_not_convert = modules_to_not_convert
        mto.apply_mode(model, mode=[("quantize", self.quantization_config.modelopt_config)])
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model, **kwargs):
        # ModelOpt imports diffusers internally. This is here to prevent circular imports
        from modelopt.torch.opt import ModeloptStateManager

        if self.pre_quantized:
            return model

        for _, m in model.named_modules():
            if hasattr(m, ModeloptStateManager._state_key) and m is not model:
                ModeloptStateManager.remove_state(m)

        return model

    @property
    def is_trainable(self):
        return True

    @property
    def is_serializable(self):
        self.quantization_config.check_model_patching(operation="saving")
        return True
