from ..utils import is_torch_available


if is_torch_available():
    from .pyramid_attention_broadcast import PyramidAttentionBroadcastConfig, apply_pyramid_attention_broadcast
