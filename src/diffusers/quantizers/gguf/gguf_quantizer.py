from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ..base import DiffusersQuantizer


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin


from ...utils import (
    get_module_from_name,
    is_accelerate_available,
    is_accelerate_version,
    is_gguf_available,
    is_gguf_version,
    is_torch_available,
    logging,
)


if is_torch_available() and is_gguf_available():
    import torch

    from .utils import (
        GGML_QUANT_SIZES,
        GGUFParameter,
        _dequantize_gguf_and_restore_linear,
        _quant_shape_from_byte_shape,
        _replace_with_gguf_linear,
    )


logger = logging.get_logger(__name__)


class GGUFQuantizer(DiffusersQuantizer):
    use_keep_in_fp32_modules = True

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        self.compute_dtype = quantization_config.compute_dtype
        self.pre_quantized = quantization_config.pre_quantized
        self.modules_to_not_convert = quantization_config.modules_to_not_convert

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available() or is_accelerate_version("<", "0.26.0"):
            raise ImportError(
                "Loading GGUF Parameters requires `accelerate` installed in your environment: `pip install 'accelerate>=0.26.0'`"
            )
        if not is_gguf_available() or is_gguf_version("<", "0.10.0"):
            raise ImportError(
                "To load GGUF format files you must have `gguf` installed in your environment: `pip install gguf>=0.10.0`"
            )

    # Copied from diffusers.quantizers.bitsandbytes.bnb_quantizer.BnB4BitDiffusersQuantizer.adjust_max_memory
    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        # need more space for buffers that are created during quantization
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        if target_dtype != torch.uint8:
            logger.info(f"target_dtype {target_dtype} is replaced by `torch.uint8` for GGUF quantization")
        return torch.uint8

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = self.compute_dtype
        return torch_dtype

    def check_quantized_param_shape(self, param_name, current_param, loaded_param):
        loaded_param_shape = loaded_param.shape
        current_param_shape = current_param.shape
        quant_type = loaded_param.quant_type

        block_size, type_size = GGML_QUANT_SIZES[quant_type]

        inferred_shape = _quant_shape_from_byte_shape(loaded_param_shape, type_size, block_size)
        if inferred_shape != current_param_shape:
            raise ValueError(
                f"{param_name} has an expected quantized shape of: {inferred_shape}, but received shape: {loaded_param_shape}"
            )

        return True

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: Union["GGUFParameter", "torch.Tensor"],
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
        param_value: Union["GGUFParameter", "torch.Tensor"],
        param_name: str,
        target_device: "torch.device",
        state_dict: Optional[Dict[str, Any]] = None,
        unexpected_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        module, tensor_name = get_module_from_name(model, param_name)
        if tensor_name not in module._parameters and tensor_name not in module._buffers:
            raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")

        if tensor_name in module._parameters:
            module._parameters[tensor_name] = param_value.to(target_device)
        if tensor_name in module._buffers:
            module._buffers[tensor_name] = param_value.to(target_device)

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        state_dict = kwargs.get("state_dict", None)

        self.modules_to_not_convert.extend(keep_in_fp32_modules)
        self.modules_to_not_convert = [module for module in self.modules_to_not_convert if module is not None]

        _replace_with_gguf_linear(
            model, self.compute_dtype, state_dict, modules_to_not_convert=self.modules_to_not_convert
        )

    def _process_model_after_weight_loading(self, model: "ModelMixin", **kwargs):
        return model

    @property
    def is_serializable(self):
        return False

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True

    def _dequantize(self, model):
        is_model_on_cpu = model.device.type == "cpu"
        if is_model_on_cpu:
            logger.info(
                "Model was found to be on CPU (could happen as a result of `enable_model_cpu_offload()`). So, moving it to accelerator. After dequantization, will move the model back to CPU again to preserve the previous device."
            )
            device = (
                torch.accelerator.current_accelerator()
                if hasattr(torch, "accelerator")
                else torch.cuda.current_device()
            )
            model.to(device)

        model = _dequantize_gguf_and_restore_linear(model, self.modules_to_not_convert)
        if is_model_on_cpu:
            model.to("cpu")
        return model
