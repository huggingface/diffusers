from typing import TYPE_CHECKING, Any

from ...utils import get_module_from_name, is_comfy_quant_available, is_torch_available, logging
from ..base import DiffusersQuantizer


if is_comfy_quant_available():
    import comfy_kitchen.tensor as ck_tensor

if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin

if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class ComfyQuantizer(DiffusersQuantizer):
    """
    Quantizer for comfy-kitchen formats (FP8, INT8, etc.).
    """

    use_keep_in_fp32_modules = True
    requires_calibration = False
    required_packages = ["comfy_kitchen"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quant_format = getattr(quantization_config, "quant_format", "fp8")
        self.compute_dtype = quantization_config.compute_dtype
        self.modules_to_not_convert = quantization_config.modules_to_not_convert or []
        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        if is_comfy_quant_available():
            # Resolve the layout class once since quant_format is constant.
            layout_map = {
                "fp8": getattr(ck_tensor, "TensorCoreFP8Layout", None),
                "nvfp4": getattr(ck_tensor, "TensorCoreNVFP4Layout", None),
                "mxfp8": getattr(ck_tensor, "TensorCoreMXFP8Layout", None),
                "int8": getattr(ck_tensor, "TensorWiseINT8Layout", None),
                "int4_svd": getattr(ck_tensor, "TensorCoreSVDQuantW4A4Layout", None),
                "int4_awq": getattr(ck_tensor, "TensorCoreAWQW4A16Layout", None),
            }
            self.layout = layout_map.get(self.quant_format.lower())
            if self.layout is None:
                supported = list(layout_map.keys())
                raise ValueError(
                    f"The layout for '{self.quant_format}' was not found in `comfy_kitchen`. "
                    f"Supported formats are: {supported}."
                )
        else:
            self.layout = None

    def validate_environment(self, *args, **kwargs):
        if not is_comfy_quant_available():
            raise ImportError(
                "Loading Comfy Quant weights requires `comfy-kitchen`. "
                "Please install it with: `pip install comfy-kitchen`."
            )

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ) -> bool:
        # Based on comfy_kitchen, we will likely wrap tensors based on some config layout.
        # For now, we assume all linear weights that aren't excluded are quantized.
        # This will be refined based on comfy-kitchen's actual detection logic.
        if any(m in param_name.split(".") for m in self.modules_to_not_convert):
            return False
        return True

    def create_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: dict[str, Any] | None = None,
        unexpected_keys: list[str] | None = None,
        **kwargs,
    ):
        module, tensor_name = get_module_from_name(model, param_name)

        if self.pre_quantized: # we have pre-quantized weights from the checkpoint
            layout_cls = ck_tensor.get_layout_class(self.layout.__name__)

            orig_shape = tuple(getattr(module, tensor_name).shape)
            params_kwargs = {
                "orig_shape": orig_shape,
                "orig_dtype": self.compute_dtype,
                "scale" : torch.tensor(1.0, dtype=torch.float32, device=target_device),
            }

            if state_dict is not None:
                for k, v in list(state_dict.items()):
                    if k.startswith(param_name + "_"):
                        suffix = k[len(param_name) :]
                        if suffix.startswith("_"):
                            field_name = suffix[1:]
                            params_kwargs[field_name] = v.to(target_device)
                        if unexpected_keys is not None and k in unexpected_keys:
                            unexpected_keys.remove(k)


            params = layout_cls.Params(**params_kwargs)
            quantized_weight = ck_tensor.QuantizedTensor(param_value.to(target_device), self.layout.__name__, params)
        else:
            quantized_weight = ck_tensor.QuantizedTensor.from_float(param_value.to(target_device), self.layout.__name__)

        if tensor_name in module._parameters:
            module._parameters[tensor_name] = quantized_weight.to(target_device)
        if tensor_name in module._buffers:
            module._buffers[tensor_name] = quantized_weight.to(target_device)

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = self.compute_dtype
        return torch_dtype

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        device_map,
        keep_in_fp32_modules: list[str] = [],
        **kwargs,
    ):
        pass

    def _process_model_after_weight_loading(self, model, **kwargs):
        pass

    @property
    def is_serializable(self):
        return False

    @property
    def is_trainable(self):
        return False
