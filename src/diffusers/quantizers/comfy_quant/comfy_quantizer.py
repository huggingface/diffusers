from typing import TYPE_CHECKING, Any
import json
from ...utils import (
    get_module_from_name,
    is_comfy_quant_available,
    is_torch_available,
    logging,
)
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

        self._checkpoint_keys = set()
        self._pending_quantized_state = {}

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

    @property
    def supports_parallel_loading(self) -> bool:
        # If pre-quantized, shards are buffered in maybe_update_state_dict. Parallel threads would corrupt the buffer.
        return not self.pre_quantized

    def maybe_update_loaded_keys(self, loaded_keys: list[str], checkpoint_files: list[str]) -> list[str]:
        # Track which keys exist in the global checkpoint to know if a tensor has an associated _scale component
        self._checkpoint_keys = set(loaded_keys)

        # We only want diffusers to try loading the base names, not the _scale / _orig_shape suffixes directly
        filtered_keys = set()
        for key in loaded_keys:
            if key.endswith("_scale") or key.endswith("_orig_shape"):
                continue
            filtered_keys.add(key)

        return list(filtered_keys)

    def maybe_update_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if not self.pre_quantized:
            return state_dict

        merged_state_dict = {**self._pending_quantized_state, **state_dict}
        self._pending_quantized_state = {}

        # Group components by base name
        param_groups = {}
        for k, v in merged_state_dict.items():
            if k.endswith("_scale"):
                base_name = k[: -len("_scale")]
                param_groups.setdefault(base_name, {})["scale"] = v
            elif k.endswith("_orig_shape"):
                base_name = k[: -len("_orig_shape")]
                param_groups.setdefault(base_name, {})["orig_shape"] = v
            else:
                param_groups.setdefault(k, {})["data"] = v

        reconstructed_state_dict = {}

        for base_name, components in param_groups.items():
            # Check if this parameter is expected to have a scale in the global checkpoint
            needs_scale = f"{base_name}_scale" in self._checkpoint_keys
            needs_orig_shape = f"{base_name}_orig_shape" in self._checkpoint_keys

            has_scale = "scale" in components
            has_orig_shape = "orig_shape" in components

            # If it doesn't need scale OR orig_shape, it's not a quantized tensor at all (e.g. Conv2d or bias).
            if not needs_scale and not needs_orig_shape:
                if "data" in components:
                    reconstructed_state_dict[base_name] = components["data"]
                continue

            # If it's a quantized weight, ensure we've loaded all its required components from the shards
            if "data" in components and (not needs_scale or has_scale) and (not needs_orig_shape or has_orig_shape):
                # All pieces are ready! Construct the QuantizedTensor
                data = components["data"]
                params_kwargs = {
                    "orig_dtype": self.compute_dtype,
                }
                if has_scale:
                    params_kwargs["scale"] = components["scale"]
                else:
                    params_kwargs["scale"] = torch.tensor(1.0, dtype=torch.float32, device=data.device)

                if has_orig_shape:
                    params_kwargs["orig_shape"] = torch.Size(components["orig_shape"].tolist())
                else:
                    params_kwargs["orig_shape"] = tuple(data.shape)

                params = self.layout.Params(**params_kwargs)
                quantized_weight = ck_tensor.QuantizedTensor(data, self.layout.__name__, params)
                reconstructed_state_dict[base_name] = quantized_weight
            else:
                # We are missing some components that exist in another shard. Buffer them.
                for suffix, v in components.items():
                    if suffix == "data":
                        self._pending_quantized_state[base_name] = v
                    else:
                        self._pending_quantized_state[f"{base_name}_{suffix}"] = v

        return reconstructed_state_dict

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ) -> bool:
        if any((key + "." in param_name) or (key == param_name) for key in self.modules_to_not_convert):
            return False

        module, tensor_name = get_module_from_name(model, param_name)
        return isinstance(module, torch.nn.Linear) and (tensor_name == "weight")

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

        if self.pre_quantized:
            # The tensor is already constructed by maybe_update_state_dict. Just assign it.
            quantized_weight = param_value.to(target_device)
        else:
            quantized_weight = ck_tensor.QuantizedTensor.from_float(
                param_value.to(target_device), self.layout.__name__
            )

        if tensor_name in module._parameters:
            module._parameters[tensor_name] = torch.nn.Parameter(quantized_weight)
        if tensor_name in module._buffers:
            module._buffers[tensor_name] = quantized_weight

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = self.compute_dtype
        return torch_dtype

    def adjust_max_memory(self, max_memory: dict[str, int | str]) -> dict[str, int | str]:
        # A 10% memory buffer for transient quantization overhead during loading
        max_memory = {key: val * 0.9 for key, val in max_memory.items()}
        return max_memory

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
        logger.warning(
            "comfy-kitchen quantized models cannot be saved via save_pretrained() because `safetensors` does not yet support custom tensor subclass flattening. Saving is disabled to prevent silent checkpoint corruption."
        )
        return False

    @property
    def is_trainable(self):
        return False

    def _dequantize(self, model):
        for name, module in model.named_modules():
            if (
                isinstance(module, torch.nn.Linear)
                and hasattr(module, "weight")
                and isinstance(module.weight, ck_tensor.QuantizedTensor)
            ):
                device = module.weight.device
                dequantized_weight = module.weight.dequantize().to(device)
                module.weight = torch.nn.Parameter(dequantized_weight)
        return model
