from typing import TYPE_CHECKING, Any, Dict, List, Union

from diffusers.utils.import_utils import is_nunchaku_version

from ...utils import (
    get_module_from_name,
    is_accelerate_available,
    is_nunchaku_available,
    is_torch_available,
    logging,
)
from ...utils.torch_utils import is_fp8_available
from ..base import DiffusersQuantizer


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin


if is_torch_available():
    import torch

if is_accelerate_available():
    pass

if is_nunchaku_available():
    from .utils import replace_with_nunchaku_linear

logger = logging.get_logger(__name__)


class NunchakuQuantizer(DiffusersQuantizer):
    r"""
    Diffusers Quantizer for Nunchaku (https://github.com/nunchaku-tech/nunchaku)
    """

    use_keep_in_fp32_modules = True
    requires_calibration = False
    required_packages = ["nunchaku", "accelerate"]

    dtype_map = {"int4": torch.int8}
    if is_fp8_available():
        dtype_map = {"nvfp4": torch.float8_e4m3fn}

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed for nunchaku quantization.")

        if not is_nunchaku_available():
            raise ImportError(
                "Loading an nunchaku quantized model requires nunchaku library (follow https://nunchaku.tech/docs/nunchaku/installation/installation.html)"
            )
        if not is_nunchaku_version(">=", "0.3.1"):
            raise ImportError(
                "Loading an nunchaku quantized model requires `nunchaku>=1.0.0`. "
                "Please upgrade your installation by following https://nunchaku.tech/docs/nunchaku/installation/installation.html."
            )

        if not is_accelerate_available():
            raise ImportError(
                "Loading an nunchaku quantized model requires accelerate library (`pip install accelerate`)"
            )

        # TODO: check
        # device_map = kwargs.get("device_map", None)
        # if isinstance(device_map, dict) and len(device_map.keys()) > 1:
        #     raise ValueError(
        #         "`device_map` for multi-GPU inference or CPU/disk offload is currently not supported with Diffusers and the nunchaku backend"
        #     )

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ):
        from nunchaku.models.linear import SVDQW4A4Linear

        module, tensor_name = get_module_from_name(model, param_name)
        if self.pre_quantized and isinstance(module, SVDQW4A4Linear):
            return True

        return False

    def create_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        *args,
        **kwargs,
    ):
        """
        Create a quantized parameter.
        """
        from nunchaku.models.linear import SVDQW4A4Linear

        module, tensor_name = get_module_from_name(model, param_name)
        if tensor_name not in module._parameters and tensor_name not in module._buffers:
            raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")

        if self.pre_quantized:
            if tensor_name in module._parameters:
                module._parameters[tensor_name] = torch.nn.Parameter(param_value.to(device=target_device))
            if tensor_name in module._buffers:
                module._buffers[tensor_name] = torch.nn.Parameter(param_value.to(target_device))

        elif isinstance(module, torch.nn.Linear):
            if tensor_name in module._parameters:
                module._parameters[tensor_name] = torch.nn.Parameter(param_value).to(device=target_device)
            if tensor_name in module._buffers:
                module._buffers[tensor_name] = torch.nn.Parameter(param_value).to(target_device)

            new_module = SVDQW4A4Linear.from_linear(module)
            setattr(model, param_name, new_module)

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        precision = self.quantization_config.precision
        expected_target_dtypes = [torch.int8]
        if is_fp8_available():
            expected_target_dtypes.append(torch.float8_e4m3fn)
        if target_dtype not in expected_target_dtypes:
            new_target_dtype = self.dtype_map[precision]

            logger.info(f"target_dtype {target_dtype} is replaced by {new_target_dtype} for `nunchaku` quantization")
            return new_target_dtype
        else:
            raise ValueError(f"Wrong `target_dtype` ({target_dtype}) provided.")

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            # We force the `dtype` to be bfloat16, this is a requirement from `nunchaku`
            logger.info(
                "Overriding torch_dtype=%s with `torch_dtype=torch.bfloat16` due to "
                "requirements of `nunchaku` to enable model loading in 4-bit. "
                "Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass"
                " torch_dtype=torch.bfloat16 to remove this warning.",
                torch_dtype,
            )
            torch_dtype = torch.bfloat16
        return torch_dtype

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        # TODO: deal with `device_map`
        self.modules_to_not_convert = self.quantization_config.modules_to_not_convert

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        self.modules_to_not_convert.extend(keep_in_fp32_modules)

        model = replace_with_nunchaku_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model, **kwargs):
        return model

    # @property
    # def is_serializable(self):
    #     return True
