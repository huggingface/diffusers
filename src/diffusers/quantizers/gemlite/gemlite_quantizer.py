from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...utils import get_module_from_name, is_gemlite_available, is_gemlite_version, is_torch_available, logging
from ..base import DiffusersQuantizer


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin

if is_torch_available():
    import torch
    import torch.nn as nn


logger = logging.get_logger(__name__)


GEMLITE_STATE_NAMES = ("W_q", "bias", "scales", "zeros", "metadata", "orig_shape", "meta_scale")

# TODO(gemlite): bump this to the next GemLite release once it ships. The version we want to require for
# Diffusers' integration has not been released yet, so keep the runtime floor installable for now.
_GEMLITE_MIN_VERSION = "0.5.1"


def _is_in_skip_modules(name: str, modules_to_not_convert: list[str]) -> bool:
    return any((key + "." in name) or (key == name) for key in modules_to_not_convert)


def _replace_with_gemlite_linear(model: "ModelMixin", modules_to_not_convert: list[str]) -> int:
    from gemlite.core import GemLiteLinearTriton

    def replace(module: "nn.Module", prefix: str = "") -> int:
        replaced = 0
        for name, child in module.named_children():
            child_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not _is_in_skip_modules(child_name, modules_to_not_convert):
                gemlite_linear = GemLiteLinearTriton().to(child.weight.device)
                if child.bias is not None:
                    gemlite_linear.bias = nn.Parameter(torch.empty_like(child.bias), requires_grad=False)
                gemlite_linear._gemlite_loaded_param_names = set()
                setattr(module, name, gemlite_linear)
                replaced += 1
            else:
                replaced += replace(child, child_name)
        return replaced

    return replace(model)


def _quantize_linears_on_the_fly(
    model: "ModelMixin", modules_to_not_convert: list[str], compute_dtype, weight_quant_format: str = "int8"
) -> int:
    """Replace every eligible ``nn.Linear`` with a packed ``GemLiteLinearTriton``.

    Layers that the GemLite kernel cannot pack (``in_features`` not divisible by 32, the kernel's minimum
    size) are left untouched in their original dtype. This keeps diffusion transformers with small
    projection layers (e.g. time/guidance embeddings) loadable without manual skip-lists.
    """
    from gemlite.core import GemLiteLinearTriton
    from gemlite.helper import A16W8_FP8, A16W8_INT8

    quantized = 0

    def visit(module: "nn.Module", prefix: str = "") -> None:
        nonlocal quantized
        for name, child in module.named_children():
            child_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not _is_in_skip_modules(child_name, modules_to_not_convert):
                device = child.weight.device
                if device.type in ("cpu", "meta"):
                    raise ValueError(
                        "GemLite on-the-fly quantization requires the model to be on a CUDA or ROCm device at load "
                        f"time. Module `{child_name}` is on {device}; pass `device_map='auto'` or move the model to "
                        "GPU."
                    )
                if child.in_features % GemLiteLinearTriton.MIN_SIZE != 0:
                    # GemLite Triton kernels require in_features divisible by MIN_SIZE (32). Small
                    # projection layers (time/guidance embeddings in diffusion transformers) are left
                    # unquantized rather than crashing the whole load.
                    logger.debug(
                        "Skipping GemLite on-the-fly quantization for %s: in_features=%d not divisible by %d.",
                        child_name,
                        child.in_features,
                        GemLiteLinearTriton.MIN_SIZE,
                    )
                    visit(child, child_name)
                    continue
                if weight_quant_format == "fp8":
                    helper = A16W8_FP8(device=str(device), dtype=compute_dtype)
                else:
                    helper = A16W8_INT8(device=str(device), dtype=compute_dtype)
                setattr(module, name, helper.from_linear(child))
                quantized += 1
            else:
                visit(child, child_name)

    visit(model)
    return quantized


class GemLiteQuantizer(DiffusersQuantizer):
    """
    Diffusers quantizer for GemLite.

    GemLite provides Triton kernels for fast low-bit matrix multiplication. This quantizer wires those kernels into
    Diffusers by replacing eligible `torch.nn.Linear` layers with `GemLiteLinearTriton` modules, allowing quantized
    weights to run directly through GemLite kernels instead of being materialized back to full precision.

    Two modes are supported:

    - **Pre-quantized loading** (`pre_quantized=True`): replaces `torch.nn.Linear` modules with empty
      `GemLiteLinearTriton` modules before weight loading, then restores the serialized GemLite state
      (`W_q`, `scales`, `zeros`, `metadata`, `orig_shape`, `meta_scale`, ...) through the low-memory loader.
      This is the main path for checkpoints already saved in GemLite format.
    - **On-the-fly quantization** (`pre_quantized=False`): loads the model with its original fp16/bf16
      `nn.Linear` layers, then packs every eligible linear layer in-place into a `GemLiteLinearTriton` module
      at the end of weight loading using the selected A16W8 GemLite helper. `weight_quant_format="int8"` uses
      `A16W8_INT8`; `weight_quant_format="fp8"` uses `A16W8_FP8`.

    Modules listed in `modules_to_not_convert` are skipped and left in their original dtype.
    """

    requires_calibration = False
    required_packages = ["gemlite"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        self.compute_dtype = quantization_config.compute_dtype
        self.modules_to_not_convert = quantization_config.modules_to_not_convert or []
        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]
        self.weight_quant_format = quantization_config.weight_quant_format

    def validate_environment(self, *args, **kwargs):
        if not is_gemlite_available():
            raise ImportError(
                "Using GemLite quantization requires the gemlite library. Please install it with `pip install gemlite`."
            )
        if is_gemlite_version("<", _GEMLITE_MIN_VERSION):
            raise ImportError(
                f"Using GemLite quantization requires gemlite>={_GEMLITE_MIN_VERSION}. "
                "Please upgrade with `pip install -U gemlite`."
            )
        try:
            from gemlite.core import GemLiteLinearTriton  # noqa: F401
        except Exception as error:
            raise ImportError("GemLite is installed but its core linear module could not be imported.") from error

        if not torch.cuda.is_available():
            raise RuntimeError(
                "GemLite quantization requires a CUDA or ROCm device. "
                "Pass `device_map='auto'` or move the model to GPU before loading."
            )

        if not self.pre_quantized and self.weight_quant_format not in ("int8", "fp8"):
            raise ValueError(
                "GemLite on-the-fly quantization currently only supports `weight_quant_format='int8'` and "
                f"`weight_quant_format='fp8'`, "
                f"got {self.weight_quant_format!r}."
            )

    def update_torch_dtype(self, torch_dtype: "torch.dtype" = None) -> "torch.dtype":
        if torch_dtype is None:
            return self.compute_dtype
        if torch_dtype != self.compute_dtype:
            logger.info(
                "Overriding torch_dtype=%s with `torch_dtype=%s` to match GemLite compute dtype.",
                torch_dtype,
                self.compute_dtype,
            )
            return self.compute_dtype
        return torch_dtype

    def update_device_map(self, device_map):
        # On-the-fly quantization packs weights on a CUDA or ROCm device at the end of weight loading, so the model
        # must be dispatched onto GPU before `_process_model_after_weight_loading` runs.
        if not self.pre_quantized and device_map is None and torch.cuda.is_available():
            current_device = f"cuda:{torch.cuda.current_device()}"
            logger.info(
                "The device_map was not initialized. Setting device_map to {'' : %s} for GemLite on-the-fly "
                "quantization. Pass `device_map='auto'` to let accelerate dispatch across available GPUs.",
                current_device,
            )
            return {"": current_device}
        return device_map

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ) -> bool:
        if not self.pre_quantized:
            return False
        from gemlite.core import GemLiteLinearTriton

        module, tensor_name = get_module_from_name(model, param_name)
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

        value = state_dict[param_name] if state_dict is not None and param_name in state_dict else param_value
        value = value.to(target_device)

        # Mirror `GemLiteLinearTriton.pack()`: store as a non-trainable Parameter so the tensor lands in
        # `_parameters` (and therefore `state_dict()`), not as a plain attribute.
        setattr(module, tensor_name, torch.nn.Parameter(value, requires_grad=False))
        module._gemlite_loaded_param_names.add(tensor_name)

    def check_quantized_param_shape(self, *args, **kwargs):
        return True

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        device_map,
        keep_in_fp32_modules: list[str] = [],
        **kwargs,
    ):
        self.modules_to_not_convert.extend(keep_in_fp32_modules)
        self.modules_to_not_convert = [module for module in self.modules_to_not_convert if module is not None]

        if not self.pre_quantized:
            # Keep the original nn.Linear layers in place; weights are loaded into them, then packed into
            # GemLiteLinearTriton modules in `_process_model_after_weight_loading`.
            model.config.quantization_config = self.quantization_config
            return

        replaced = _replace_with_gemlite_linear(model, self.modules_to_not_convert)
        if replaced == 0:
            logger.warning("No linear modules were replaced with GemLite linear layers.")

        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "ModelMixin", **kwargs):
        from gemlite.core import GemLiteLinearTriton

        if not self.pre_quantized:
            quantized = _quantize_linears_on_the_fly(
                model, self.modules_to_not_convert, self.compute_dtype, self.weight_quant_format
            )
            if quantized == 0:
                logger.warning("No linear modules were quantized with GemLite on-the-fly.")
            return model

        for module in model.modules():
            if isinstance(module, GemLiteLinearTriton):
                # `create_quantized_param` stores serialized tensors as Parameters so they remain in
                # `state_dict()` during loading. GemLite's `load_state_dict` reassigns `metadata` and
                # `orig_shape` to Python containers, which nn.Module.__setattr__ rejects while the names
                # are still in `_parameters`. De-register first so gemlite can set them freely, then
                # build the state dict from the plain tensors.
                for meta_name in ("metadata", "orig_shape"):
                    param = module._parameters.pop(meta_name, None)
                    if param is not None:
                        object.__setattr__(module, meta_name, param.data)
                state_dict = {
                    name: getattr(module, name)
                    for name in GEMLITE_STATE_NAMES
                    if name in module._gemlite_loaded_param_names
                    if getattr(module, name, None) is not None
                }
                module.load_state_dict(state_dict)
                # GemLite converts metadata/orig_shape to plain Python containers during load_state_dict().
                # Register tensor copies again so save_pretrained() can emit a complete pre-quantized checkpoint.
                setattr(
                    module,
                    "metadata",
                    torch.nn.Parameter(
                        torch.tensor(module.get_meta_args(), device=module.W_q.device, dtype=torch.int32),
                        requires_grad=False,
                    ),
                )
                setattr(
                    module,
                    "orig_shape",
                    torch.nn.Parameter(
                        torch.tensor(
                            [module.out_features, module.in_features], device=module.W_q.device, dtype=torch.int32
                        ),
                        requires_grad=False,
                    ),
                )
        return model

    @property
    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True
