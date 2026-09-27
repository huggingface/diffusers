from ...utils import is_torch_available


if is_torch_available():
    from .condition_embedder_anima import AnimaTextConditioner
    from .condition_embedder_minimax_music3 import MiniMaxMusic3ConditionEncoder
