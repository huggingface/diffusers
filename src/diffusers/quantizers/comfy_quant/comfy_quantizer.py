from typing import TYPE_CHECKING, Any

from ...utils import (
    get_module_from_name,
    is_torch_available,
    logging,
)
from ..base import DiffusersQuantizer


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
        self.compute_dtype = quantization_config.compute_dtype
        self.modules_to_not_convert = quantization_config.modules_to_not_convert or []
        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

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

        # Defaulting to an example layout for now. Ideally this is pulled from config or metadata.
        # Since ComfyQuantConfig can store the exact layout/format, we'd use it here.
        # quantized_weight = ck_tensor.QuantizedTensor.from_float(param_value, ck_tensor.TensorCoreFP8Layout)

        # Since we don't have the exact layout detection in this PR snippet, we do a basic wrap.
        # This is a placeholder for the actual comfy-kitchen wrapping logic.
        quantized_weight = param_value

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
