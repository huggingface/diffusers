from ...utils import is_torch_available


if is_torch_available():
    from .duration_head_ltx2 import LTX2DurationHead
    from .prompt_enhancer_head_ideogram4 import Ideogram4PromptEnhancerHead
