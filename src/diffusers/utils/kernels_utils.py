from .import_utils import is_kernels_available


_DEFAULT_HUB_ID_FA3 = "kernels-community/vllm-flash-attn3"


def _get_fa3_from_hub():
    if not is_kernels_available():
        return None
    else:
        from kernels import get_kernel

        try:
            vllm_flash_attn3 = get_kernel(_DEFAULT_HUB_ID_FA3)
            return vllm_flash_attn3
        except Exception as e:
            raise e
