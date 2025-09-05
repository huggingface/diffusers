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
