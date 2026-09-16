from typing import TYPE_CHECKING, Any

from ...utils import (
    get_module_from_name,
    is_torch_available,
    is_comfy_kitchen_available,
    logging
)
from ..base import DiffusersQuantizer

if is_comfy_kitchen_available():
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

    def validate_environment(self, *args, **kwargs):
        from ...utils.import_utils import is_comfy_kitchen_available

        if not is_comfy_kitchen_available():
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

        quantized_weight = ck_tensor.QuantizedTensor.from_float(
            param_value.to(target_device), self.layout.__name__
        )

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
