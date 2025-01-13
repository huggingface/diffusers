from ..utils import is_torch_available


if is_torch_available():
    from .hooks import HookRegistry
    from .pyramid_attention_broadcast import PyramidAttentionBroadcastConfig, apply_pyramid_attention_broadcast
