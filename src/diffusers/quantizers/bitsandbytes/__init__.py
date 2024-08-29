from ...utils import is_accelerate_available, is_bitsandbytes_available, is_torch_available


if is_torch_available() and is_bitsandbytes_available() and is_accelerate_available():
    from .bnb_quantizer import BnB4BitDiffusersQuantizer, BnB8BitDiffusersQuantizer
    from .utils import (
        dequantize_and_replace,
        dequantize_bnb_weight,
        replace_with_bnb_linear,
        set_module_quantized_tensor_to_device,
    )
