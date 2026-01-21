from ...utils import is_torch_available


if is_torch_available():
    from .transformer_2d import Transformer2DModel
