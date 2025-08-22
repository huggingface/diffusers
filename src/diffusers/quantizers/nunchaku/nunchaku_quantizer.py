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

        module, _ = get_module_from_name(model, param_name)
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
        state_dict = args[0]
        if tensor_name not in module._parameters and tensor_name not in module._buffers:
            raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")

        if isinstance(module, SVDQW4A4Linear):
            if param_value.ndim == 1:
                module._parameters[tensor_name] = torch.nn.Parameter(param_value, requires_grad=False).to(
                    target_device
                )
            elif tensor_name == "qweight":
                module._parameters[tensor_name] = torch.nn.Parameter(param_value, requires_grad=False).to(
                    target_device
                )
                # if the tensor has qweight, but does not have low-rank branch, we need to add some artificial tensors
                for t in ["lora_up", "lora_down"]:
                    # need to check at the state dict level for this
                    new_tensor_name = param_name.replace(".qweight", f".{t}")
                    if new_tensor_name not in state_dict:
                        oc, ic = param_value.shape
                        ic = ic * 2  # v is packed into INT8, so we need to double the size
                        module._parameters[t] = torch.zeros(
                            (0, ic) if t == "lora_down" else (oc, 0), device=param_value.device, dtype=torch.bfloat16
                        )
            else:
                module._parameters[tensor_name] = torch.nn.Parameter(param_value, requires_grad=False).to(
                    target_device
                )

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
        self.modules_to_not_convert = self.quantization_config.modules_to_not_convert
        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]
        self.modules_to_not_convert.extend(keep_in_fp32_modules)
        # Purge `None`.
        # Unlike `transformers`, we don't know if we should always keep certain modules in FP32
        # in case of diffusion transformer models. For language models and others alike, `lm_head`
        # and tied modules are usually kept in FP32.
        self.modules_to_not_convert = [module for module in self.modules_to_not_convert if module is not None]

        # Extend `self.modules_to_not_convert` to keys that are supposed to be offloaded to `cpu` or `disk`
        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]
            self.modules_to_not_convert.extend(keys_on_cpu)

        model = replace_with_nunchaku_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model, **kwargs):
        return model

    @property
    def is_serializable(self):
        return False

    @property
    def is_trainable(self):
        return False
