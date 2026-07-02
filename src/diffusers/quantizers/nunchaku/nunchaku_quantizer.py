from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..base import DiffusersQuantizer
from .utils import (
    check_strict_state_dict_match,
    replace_with_nunchaku_linear,
)


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin


from ...utils import is_kernels_available, logging


logger = logging.get_logger(__name__)


class NunchakuLiteQuantizer(DiffusersQuantizer):
    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.compute_dtype = quantization_config.compute_dtype
        self.pre_quantized = quantization_config.pre_quantized

    def validate_environment(self, *args, **kwargs):
        if not is_kernels_available():
            raise ImportError(
                "Loading Nunchaku checkpoints requires the Hugging Face `kernels` package. "
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
        quantization_config = self.quantization_config.to_dict()
        num_replaced = replace_with_nunchaku_linear(model, quantization_config, self.compute_dtype)

        if state_dict is not None:
            check_strict_state_dict_match(model, state_dict)
        logger.info(f"Applied Nunchaku quantization config with {num_replaced} targets.")

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
