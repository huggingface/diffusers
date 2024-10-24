from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ...utils import get_module_from_name
from ..base import DiffusersQuantizer


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin

from ...utils import (
    is_accelerate_available,
    is_accelerate_version,
    is_bitsandbytes_available,
    is_bitsandbytes_version,
    is_torch_available,
    logging,
)


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class GGUFQuantizer(DiffusersQuantizer)
    use_keep_in_fp32_modules = True

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def check_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
        ) -> bool:

        return

    def create_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: Optional[List[str]] = None,
    ):
        import bitsandbytes as bnb

        module, tensor_name = get_module_from_name(model, param_name)

        if tensor_name not in module._parameters:
            raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")

        old_value = getattr(module, tensor_name)

        if tensor_name == "bias":
            if param_value is None:
                new_value = old_value.to(target_device)
            else:
                new_value = param_value.to(target_device)

            new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad)
            module._parameters[tensor_name] = new_value
            return

        if not isinstance(module._parameters[tensor_name], bnb.nn.Params4bit):
            raise ValueError("this function only loads `Linear4bit components`")
        if (
            old_value.device == torch.device("meta")
            and target_device not in ["meta", torch.device("meta")]
            and param_value is None
        ):
            raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {target_device}.")

        # construct `new_value` for the module._parameters[tensor_name]:
        if self.pre_quantized:
            # 4bit loading. Collecting components for restoring quantized weight
            # This can be expanded to make a universal call for any quantized weight loading

            if not self.is_serializable:
                raise ValueError(
                    "Detected int4 weights but the version of bitsandbytes is not compatible with int4 serialization. "
                    "Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`."
                )

            if (param_name + ".quant_state.bitsandbytes__fp4" not in state_dict) and (
                param_name + ".quant_state.bitsandbytes__nf4" not in state_dict
            ):
                raise ValueError(
                    f"Supplied state dict for {param_name} does not contain `bitsandbytes__*` and possibly other `quantized_stats` components."
                )

            quantized_stats = {}
            for k, v in state_dict.items():
                # `startswith` to counter for edge cases where `param_name`
                # substring can be present in multiple places in the `state_dict`
                if param_name + "." in k and k.startswith(param_name):
                    quantized_stats[k] = v
                    if unexpected_keys is not None and k in unexpected_keys:
                        unexpected_keys.remove(k)

            new_value = bnb.nn.Params4bit.from_prequantized(
                data=param_value,
                quantized_stats=quantized_stats,
                requires_grad=False,
                device=target_device,
            )
        else:
            new_value = param_value.to("cpu")
            kwargs = old_value.__dict__
            new_value = bnb.nn.Params4bit(new_value, requires_grad=False, **kwargs).to(target_device)

        module._parameters[tensor_name] = new_value
