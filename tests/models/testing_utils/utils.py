import torch

from diffusers.models.attention_dispatch import AttentionBackendName


_BF16_REQUIRED_BACKENDS = {
    AttentionBackendName._NATIVE_CUDNN,
    AttentionBackendName.FLASH_HUB,
    AttentionBackendName.FLASH_VARLEN_HUB,
    AttentionBackendName._FLASH_3_HUB,
    AttentionBackendName._FLASH_3_VARLEN_HUB,
}


def _maybe_cast_to_bf16(backend, model, inputs_dict):
    """Cast model and floating-point inputs to bfloat16 when the backend requires it."""
    if not backend or backend not in _BF16_REQUIRED_BACKENDS:
        return model, inputs_dict
    if getattr(model, "_keep_in_fp32_modules", None):
        raise NotImplementedError("Do not know how to define casting for models with `_keep_in_fp32_modules`.")
    model = model.to(dtype=torch.bfloat16)
    inputs_dict = {
        k: v.to(dtype=torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
        for k, v in inputs_dict.items()
    }
    return model, inputs_dict


def run_nondeterministic(fn):
    """
    Run `fn` with `enable_full_determinism`'s deterministic-algorithm requirement lifted, restoring the previous
    setting afterwards.

    Several models reach a backward kernel that has no deterministic CUDA implementation (reflection/replication
    padding, average pooling), which makes every test doing a backward pass raise under
    `torch.use_deterministic_algorithms(True)`. Wrap those tests instead of relaxing determinism for the whole module,
    and name the offending op at the call site.
    """
    was_enabled = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        fn()
    finally:
        torch.use_deterministic_algorithms(was_enabled)
