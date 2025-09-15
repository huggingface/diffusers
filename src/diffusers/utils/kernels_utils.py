from typing import Union

from ..utils import get_logger
from .import_utils import is_kernels_available


logger = get_logger(__name__)


_DEFAULT_HUB_ID_FA3 = "kernels-community/flash-attn3"


def _get_fa3_from_hub():
    if not is_kernels_available():
        return None
    else:
        from kernels import get_kernel

        try:
            # TODO: temporary revision for now. Remove when merged upstream into `main`.
            flash_attn_3_hub = get_kernel(_DEFAULT_HUB_ID_FA3, revision="fake-ops-return-probs")
            return flash_attn_3_hub
        except Exception as e:
            logger.error(f"An error occurred while fetching kernel '{_DEFAULT_HUB_ID_FA3}' from the Hub: {e}")
            raise


if is_kernels_available():
    from kernels import (
        Device,
        LayerRepository,
        register_kernel_mapping,
        replace_kernel_forward_from_hub,
        use_kernel_forward_from_hub,
    )

    _KERNEL_MAPPING: dict[str, dict[Union[Device, str], LayerRepository]] = {
        "RMSNorm": {
            "cuda": LayerRepository(repo_id="kernels-community/liger_kernels", layer_name="LigerRMSNorm"),
        },
        "MLP": {"cuda": LayerRepository(repo_id="medmekk/triton-llama-mlp", layer_name="TritonLlamaMLP")},
    }

    register_kernel_mapping(_KERNEL_MAPPING)

else:
    # Stub to make decorators int transformers work when `kernels`
    # is not installed.
    def use_kernel_forward_from_hub(*args, **kwargs):
        def decorator(cls):
            return cls

        return decorator

    class LayerRepository:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("LayerRepository requires `kernels` to be installed. Run `pip install kernels`.")

    def replace_kernel_forward_from_hub(*args, **kwargs):
        raise RuntimeError(
            "replace_kernel_forward_from_hub requires `kernels` to be installed. Run `pip install kernels`."
        )

    def register_kernel_mapping(*args, **kwargs):
        raise RuntimeError("register_kernel_mapping requires `kernels` to be installed. Run `pip install kernels`.")
