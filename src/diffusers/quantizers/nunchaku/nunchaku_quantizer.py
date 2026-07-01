from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..base import DiffusersQuantizer


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin


from ...utils import is_kernels_available, logging


logger = logging.get_logger(__name__)


class NunchakuLiteQuantizer(DiffusersQuantizer):
    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.compute_dtype = quantization_config.compute_dtype
        self.pre_quantized = quantization_config.pre_quantized
        self.runtime_manifest = None

    def validate_environment(self, *args, **kwargs):
        if not is_kernels_available():
            raise ImportError(
                "Loading Nunchaku Lite checkpoints requires the Hugging Face `kernels` package. "
                "Install it with `pip install kernels`."
            )

    def update_torch_dtype(self, torch_dtype):
        if torch_dtype is None:
            torch_dtype = self.compute_dtype
        return torch_dtype

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        state_dict: dict[str, Any] | None = None,
        metadata: dict[str, str] | None = None,
        **kwargs,
    ):
        if state_dict is None:
            raise ValueError("Nunchaku Lite quantization requires a checkpoint state dict before weight loading.")

        from .utils import parse_runtime_manifest, replace_with_nunchaku_linear

        quantization_config = self._parse_quantization_config(metadata)
        self.runtime_manifest = parse_runtime_manifest(quantization_config)
        replace_with_nunchaku_linear(model, self.runtime_manifest, self.compute_dtype)
        self._check_strict_state_dict_match(model, state_dict)
        logger.info(f"Applied Nunchaku Lite runtime manifest with {len(self.runtime_manifest.targets)} targets.")

    def _process_model_after_weight_loading(self, model: "ModelMixin", **kwargs):
        if self.runtime_manifest is not None:
            model._nunchaku_lite_runtime_manifest = self.runtime_manifest
        return model

    def _parse_quantization_config(self, metadata: dict[str, str] | None) -> dict[str, Any]:
        if not metadata or "quantization_config" not in metadata:
            raise ValueError("Nunchaku Lite checkpoints must include a JSON `quantization_config` metadata field.")
        try:
            quantization_config = json.loads(metadata["quantization_config"])
        except json.JSONDecodeError as exc:
            raise ValueError("Nunchaku Lite checkpoint metadata field `quantization_config` is not valid JSON.") from exc
        if not isinstance(quantization_config, dict):
            raise ValueError("Nunchaku Lite checkpoint metadata field `quantization_config` must decode to a JSON object.")
        return quantization_config

    def _check_strict_state_dict_match(self, model: "ModelMixin", state_dict: dict[str, Any]):
        expected_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        missing_keys = sorted(expected_keys - loaded_keys)
        unexpected_keys = sorted(loaded_keys - expected_keys)
        if missing_keys or unexpected_keys:
            message = "Nunchaku Lite checkpoint keys must exactly match the patched model state dict."
            if missing_keys:
                message += f" Missing keys: {missing_keys[:10]}"
                if len(missing_keys) > 10:
                    message += f" and {len(missing_keys) - 10} more"
                message += "."
            if unexpected_keys:
                message += f" Unexpected keys: {unexpected_keys[:10]}"
                if len(unexpected_keys) > 10:
                    message += f" and {len(unexpected_keys) - 10} more"
                message += "."
            raise ValueError(message)

    @property
    def is_serializable(self):
        return False

    @property
    def is_trainable(self) -> bool:
        return False
