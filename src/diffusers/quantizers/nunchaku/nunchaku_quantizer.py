from __future__ import annotations

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

    def validate_environment(self, *args, **kwargs):
        if not is_kernels_available():
            raise ImportError(
                "Loading Nunchaku checkpoints requires the Hugging Face `kernels` package. "
                "Install it with `pip install kernels`."
            )
        import torch

        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            raise ValueError("Loading Nunchaku checkpoints requires a CUDA-capable NVIDIA GPU.")

        device_capability = torch.cuda.get_device_capability()

        if device_capability[0] == 9:
            raise ValueError("Loading Nunchaku checkpoints is not supported on Hopper NVIDIA GPUs.")

        has_nvfp4_config = (
            self.quantization_config.svdq_w4a4 is not None
            and self.quantization_config.svdq_w4a4["precision"] == "nvfp4"
        )
        has_int4_config = any(
            config is not None and config["precision"] == "int4"
            for config in (self.quantization_config.svdq_w4a4, self.quantization_config.awq_w4a16)
        )
        if has_nvfp4_config and device_capability < (10, 0):
            raise ValueError("Loading Nunchaku NVFP4 checkpoints requires a Blackwell or newer NVIDIA GPU.")
        if has_int4_config and device_capability < (7, 5):
            raise ValueError("Loading Nunchaku INT4 checkpoints on CUDA requires a Turing or newer NVIDIA GPU.")

    def update_torch_dtype(self, torch_dtype):
        if torch_dtype is None:
            torch_dtype = self.compute_dtype
        else:
            self.compute_dtype = torch_dtype
        return torch_dtype

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        state_dict: dict[str, Any] | None = None,
        **kwargs,
    ):
        from .utils import check_strict_state_dict_match, replace_with_nunchaku_linear

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
