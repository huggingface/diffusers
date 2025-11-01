from ..utils import get_logger
from .import_utils import is_kernels_available


if is_kernels_available():
    from kernels import get_kernel

logger = get_logger(__name__)

_DEFAULT_HUB_IDS = {
    "fa3": ("kernels-community/flash-attn3", {"revision": "fake-ops-return-probs"}),
    "fa": ("kernels-community/flash-attn", {}),
}


def _get_from_hub(key: str):
    if not is_kernels_available():
        return None

    hub_id, kwargs = _DEFAULT_HUB_IDS[key]
    try:
        return get_kernel(hub_id, **kwargs)
    except Exception as e:
        logger.error(f"An error occurred while fetching kernel '{hub_id}' from the Hub: {e}")
        raise


def _get_fa3_from_hub():
    return _get_from_hub("fa3")


def _get_fa_from_hub():
    return _get_from_hub("fa")
