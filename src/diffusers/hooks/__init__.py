from ..utils import is_torch_available


if is_torch_available():
    from .group_offloading import apply_group_offloading
    from .hooks import HookRegistry
