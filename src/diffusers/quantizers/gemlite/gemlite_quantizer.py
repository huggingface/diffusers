from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...utils import get_module_from_name, is_gemlite_available, is_gemlite_version, is_torch_available, logging
from ..base import DiffusersQuantizer


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin
    from ..quantization_config import GemLiteConfig

if is_torch_available():
    import torch
    import torch.nn as nn


logger = logging.get_logger(__name__)


GEMLITE_STATE_NAMES = ("W_q", "bias", "scales", "zeros", "metadata", "orig_shape", "meta_scale")

_GEMLITE_MIN_VERSION = "0.6.0"


def _is_in_skip_modules(name: str, modules_to_not_convert: list[str]) -> bool:
    return any((key + "." in name) or (key == name) for key in modules_to_not_convert)


class _GemLiteDiffusersProcessor:
    """Implement GemLite's `patch_model` protocol for Diffusers' pre-quantized loading path."""

    def __init__(self, quantization_config: "GemLiteConfig", modules_to_not_convert: list[str]):
        from gemlite.core import DType
        from gemlite.dtypes import DTYPE_TO_TORCH, PACKING_BITWIDTH_TO_TORCH_DTYPE

        gemlite_dtypes = {
            "fp16": DType.FP16,
            "float16": DType.FP16,
            "bf16": DType.BF16,
            "bfloat16": DType.BF16,
            "fp32": DType.FP32,
            "float32": DType.FP32,
        }
        self.bits = quantization_config.bits
        self.group_size = quantization_config.group_size
        self.input_dtype = gemlite_dtypes[quantization_config.input_dtype]
        self.output_dtype = gemlite_dtypes[quantization_config.output_dtype]
        self.scales_gemlite_dtype = gemlite_dtypes[quantization_config.scales_dtype]
        self.scales_dtype = DTYPE_TO_TORCH[self.scales_gemlite_dtype.value]
        self.zeros_dtype = DTYPE_TO_TORCH[gemlite_dtypes[quantization_config.zeros_dtype].value]
        self.packed_dtype = PACKING_BITWIDTH_TO_TORCH_DTYPE[quantization_config.packing_bitwidth]
        self.elements_per_sample = quantization_config.packing_bitwidth // quantization_config.bits
        self.modules_to_not_convert = modules_to_not_convert
        self.quantized_fqns = (
            set(quantization_config.quantized_fqns) if quantization_config.quantized_fqns is not None else None
        )

    def from_linear(self, linear: "nn.Linear") -> "nn.Module":
        from gemlite.core import GemLiteLinearTriton

        should_skip_linear = (
            self.quantized_fqns is not None and linear.name not in self.quantized_fqns
        ) or _is_in_skip_modules(linear.name, self.modules_to_not_convert)
        if should_skip_linear:
            return linear

        gemlite_linear = GemLiteLinearTriton(
            W_nbits=self.bits,
            group_size=self.group_size,
            in_features=linear.in_features,
            out_features=linear.out_features,
            input_dtype=self.input_dtype,
            output_dtype=self.output_dtype,
        ).to(linear.weight.device)
        gemlite_linear.elements_per_sample = self.elements_per_sample
        gemlite_linear.meta_dtype = self.scales_gemlite_dtype
        gemlite_linear.channel_scale_mode = 0
        gemlite_linear.W_group_mode = 0
        gemlite_linear.data_contiguous = True
        gemlite_linear.W_q = torch.empty(
            linear.in_features // self.elements_per_sample,
            linear.out_features,
            dtype=self.packed_dtype,
            device=linear.weight.device,
        )
        gemlite_linear.scales = torch.empty(
            linear.in_features // self.group_size,
            linear.out_features,
            dtype=self.scales_dtype,
            device=linear.weight.device,
        )
        gemlite_linear.zeros = torch.empty(
            linear.in_features // self.group_size,
            linear.out_features,
            dtype=self.zeros_dtype,
            device=linear.weight.device,
        )
        if linear.bias is not None:
            gemlite_linear.bias = nn.Parameter(torch.empty_like(linear.bias), requires_grad=False)
        gemlite_linear._gemlite_loaded_param_names = set()
        return gemlite_linear


class GemLiteQuantizer(DiffusersQuantizer):
    """
    Diffusers quantizer for GemLite.

    GemLite provides Triton kernels for fast low-bit matrix multiplication. This quantizer wires those kernels into
    Diffusers by replacing eligible `torch.nn.Linear` layers with `GemLiteLinearTriton` modules, allowing quantized
    weights to run directly through GemLite kernels instead of being materialized back to full precision.

    This quantizer only loads pre-quantized checkpoints. It replaces `torch.nn.Linear` modules with
    `GemLiteLinearTriton` modules before weight loading, then restores the serialized GemLite state (`W_q`, `scales`,
    `zeros`, `metadata`, `orig_shape`, `meta_scale`, ...) through the low-memory loader. The quantization config
    provides the serialized layout, so the replacement tensors have the checkpoint's exact shapes and dtypes before
    Accelerate estimates module sizes for `device_map="auto"`.

    Modules listed in `modules_to_not_convert` are skipped and left in their original dtype.
    """

    use_keep_in_fp32_modules = True
    requires_calibration = False
    required_packages = ["gemlite"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        if not self.pre_quantized:
            raise ValueError("GemLite quantization in Diffusers only supports loading pre-quantized checkpoints.")

        self.compute_dtype = quantization_config.compute_dtype
        self.modules_to_not_convert = quantization_config.modules_to_not_convert or []
        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

    def update_torch_dtype(self, torch_dtype: "torch.dtype | None") -> "torch.dtype":
        if torch_dtype is None:
            return self.compute_dtype
        if torch_dtype != self.compute_dtype:
            raise ValueError(
                "`torch_dtype` passed to `from_pretrained` must match `GemLiteConfig.compute_dtype`. "
                f"Got {torch_dtype} and {self.compute_dtype}, respectively."
            )
        return torch_dtype

    def get_special_dtypes_update(self, model, torch_dtype: "torch.dtype") -> dict[str, "torch.dtype"]:
        special_dtypes = super().get_special_dtypes_update(model, torch_dtype)

        from gemlite.core import GemLiteLinearTriton

        for module_name, module in model.named_modules():
            if not isinstance(module, GemLiteLinearTriton):
                continue
            for tensor_name, tensor in module.named_parameters(recurse=False):
                name = f"{module_name}.{tensor_name}" if module_name else tensor_name
                special_dtypes[name] = tensor.dtype
            for tensor_name, tensor in module.named_buffers(recurse=False):
                name = f"{module_name}.{tensor_name}" if module_name else tensor_name
                special_dtypes[name] = tensor.dtype

        return special_dtypes

    @property
    def supports_parallel_loading(self) -> bool:
        return False

    def validate_environment(self, *args, **kwargs):
        device_map = kwargs.get("device_map")
        if isinstance(device_map, dict) and "disk" in device_map.values():
            raise ValueError("GemLite quantization does not support disk offloading.")

        if not is_gemlite_available():
            raise ImportError(
                "Using GemLite quantization requires the gemlite library. Please install it with `pip install gemlite`."
            )
        if is_gemlite_version("<", _GEMLITE_MIN_VERSION):
            raise ImportError(
                f"Using GemLite quantization requires gemlite>={_GEMLITE_MIN_VERSION}. "
                "Please upgrade with `pip install -U gemlite`."
            )

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ) -> bool:
        module, tensor_name = get_module_from_name(model, param_name)
        from gemlite.core import GemLiteLinearTriton

        return tensor_name in GEMLITE_STATE_NAMES and isinstance(module, GemLiteLinearTriton)

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
        if tensor_name not in GEMLITE_STATE_NAMES:
            raise ValueError(f"`{param_name}` is not a GemLite serialized tensor.")

        value = state_dict[param_name] if state_dict is not None else param_value
        value = value.to(target_device)

        setattr(module, tensor_name, torch.nn.Parameter(value, requires_grad=False))
        module._gemlite_loaded_param_names.add(tensor_name)

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        device_map,
        keep_in_fp32_modules: list[str] = [],
        **kwargs,
    ):
        """
        Replace `nn.Linear` modules with GemLite modules before the checkpoint tensors are loaded. The serialized
        layout in the config creates correctly shaped GemLite tensors. Modules in `keep_in_fp32_modules` are excluded.
        """
        self.modules_to_not_convert.extend(keep_in_fp32_modules)
        self.modules_to_not_convert = [module for module in self.modules_to_not_convert if module is not None]

        from gemlite.helper import patch_model

        processor = _GemLiteDiffusersProcessor(self.quantization_config, self.modules_to_not_convert)
        patch_model(
            model,
            device=next(model.parameters()).device,
            processor=processor,
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "ModelMixin", **kwargs):
        from gemlite.core import GemLiteLinearTriton

        has_gemlite_linear = False
        for module in model.modules():
            if not isinstance(module, GemLiteLinearTriton):
                continue
            has_gemlite_linear = True
            state_dict = {
                name: getattr(module, name)
                for name in GEMLITE_STATE_NAMES
                if name in module._gemlite_loaded_param_names
            }
            module.load_state_dict(state_dict)

        if not has_gemlite_linear:
            logger.warning("No linear modules are using GemLite.")
        return model

    def get_state_dict_and_metadata(
        self, state_dict: dict[str, Any], safe_serialization: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # This hook runs from `ModelMixin.save_pretrained`, before the state dict is written. SafeTensors requires
        # contiguous tensors, so return a self-consistent GemLite state dict by materializing `W_q` and updating its
        # paired `data_contiguous` metadata together. The generic save path's later `.contiguous()` is then a no-op.
        state_dict = state_dict.copy()
        for name, w_q in state_dict.items():
            module_name, _, tensor_name = name.rpartition(".")
            if tensor_name != "W_q":
                continue

            metadata_name = f"{module_name}.metadata" if module_name else "metadata"
            orig_shape_name = f"{module_name}.orig_shape" if module_name else "orig_shape"
            if metadata_name not in state_dict or orig_shape_name not in state_dict:
                continue

            metadata = state_dict[metadata_name]
            data_contiguous = bool(metadata[-1].item())
            if not data_contiguous:
                state_dict[name] = w_q.contiguous()
                state_dict[metadata_name] = metadata.clone()
                state_dict[metadata_name][-1] = 1

        return state_dict, {}

    @property
    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True
