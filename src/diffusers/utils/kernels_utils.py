from ..utils import get_logger
from .import_utils import is_kernels_available


logger = get_logger(__name__)


_DEFAULT_HUB_ID_FA3 = "kernels-community/flash-attn3"
_DEFAULT_HUB_ID_SAGE = "kernels-community/sage_attention"
_KERNEL_REVISION = {
    # TODO: temporary revision for now. Remove when merged upstream into `main`.
    _DEFAULT_HUB_ID_FA3: "fake-ops-return-probs",
}


def _get_kernel_from_hub(kernel_id):
    if not is_kernels_available():
        return None
    else:
        from kernels import get_kernel

        try:
            if kernel_id not in _KERNEL_REVISION:
                raise NotImplementedError(f"{kernel_id} is not implemented in Diffusers.")
            kernel_hub = get_kernel(kernel_id, revision=_KERNEL_REVISION.get(kernel_id))
            return kernel_hub
        except Exception as e:
            logger.error(f"An error occurred while fetching kernel '{kernel_id}' from the Hub: {e}")
            raise
