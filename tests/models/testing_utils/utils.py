import torch

from diffusers.models.attention_dispatch import AttentionBackendName


_BF16_REQUIRED_BACKENDS = {
    AttentionBackendName._NATIVE_CUDNN,
    AttentionBackendName.FLASH_HUB,
    AttentionBackendName._FLASH_3_HUB,
}


def _maybe_cast_to_bf16(backend, model, inputs_dict):
    """Cast model and floating-point inputs to bfloat16 when the backend requires it."""
    if not backend or backend not in _BF16_REQUIRED_BACKENDS:
        return model, inputs_dict
    model = model.to(dtype=torch.bfloat16)
    inputs_dict = {
        k: v.to(dtype=torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
        for k, v in inputs_dict.items()
    }
    return model, inputs_dict
