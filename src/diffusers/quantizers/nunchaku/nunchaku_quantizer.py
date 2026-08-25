from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..base import DiffusersQuantizer


if TYPE_CHECKING:
    import torch

    from ...models.modeling_utils import ModelMixin


from ...utils import is_kernels_available, logging


logger = logging.get_logger(__name__)


class NunchakuLiteQuantizer(DiffusersQuantizer):
    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.compute_dtype = quantization_config.compute_dtype
        # Quantize on load when either the loader inferred an unquantized
        # checkpoint or the config explicitly requested `pre_quantized=False`.
        self.pre_quantized = self.pre_quantized and quantization_config.pre_quantized

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

        svdq_config = self.quantization_config.svdq_w4a4
        if not self.pre_quantized and svdq_config is not None and svdq_config.get("targets") is None:
            from .data_free import infer_data_free_targets

            svdq_config["targets"] = infer_data_free_targets(
                model,
                group_size=svdq_config["group_size"],
                exclude_targets=self.quantization_config.exclude_targets or (),
            )
            logger.info(f"Inferred {len(svdq_config['targets'])} data-free quantization targets.")

        quantization_config = self.quantization_config.to_dict()
        num_replaced = replace_with_nunchaku_linear(model, quantization_config, self.compute_dtype)

        if self.pre_quantized and state_dict is not None:
            check_strict_state_dict_match(model, state_dict)
        logger.info(f"Applied Nunchaku quantization config with {num_replaced} targets.")

    def update_missing_keys(self, model, missing_keys: list[str], prefix: str) -> list[str]:
        if self.pre_quantized:
            return missing_keys
        # In data-free mode the checkpoint holds `weight`/`bias` while the model
        # expects the packed parameters; those are produced at load time.
        from .data_free import DATA_FREE_PARAMETER_NAMES

        return [key for key in missing_keys if key.rpartition(".")[2] not in DATA_FREE_PARAMETER_NAMES]

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ) -> bool:
        if self.pre_quantized:
            return False
        from .utils import SVDQW4A4Linear

        module_name, _, tensor_name = param_name.rpartition(".")
        if tensor_name not in ("weight", "bias") or not module_name:
            return False
        try:
            module = model.get_submodule(module_name)
        except AttributeError:
            return False
        return isinstance(module, SVDQW4A4Linear)

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
        import torch

        from .data_free import pack_data_free_bias, quantize_linear_data_free

        module_name, _, tensor_name = param_name.rpartition(".")
        module = model.get_submodule(module_name)
        if unexpected_keys is not None and param_name in unexpected_keys:
            unexpected_keys.remove(param_name)
        if tensor_name == "bias":
            packed_bias = pack_data_free_bias(param_value.to(target_device), torch_dtype=self.compute_dtype)
            module._parameters["bias"] = torch.nn.Parameter(packed_bias, requires_grad=False)
            return
        quantized = quantize_linear_data_free(
            param_value.to(target_device),
            precision=module.precision,
            group_size=module.group_size,
            rank=module.rank,
            torch_dtype=self.compute_dtype,
        )
        for name, tensor in quantized.items():
            module._parameters[name] = torch.nn.Parameter(tensor.to(target_device), requires_grad=False)

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
