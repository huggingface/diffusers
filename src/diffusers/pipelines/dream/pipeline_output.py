from dataclasses import dataclass
from typing import List

from ...utils import BaseOutput


@dataclass
class DreamTextPipelineOutput(BaseOutput):
    """
    Output class for the Dream-7B diffusion LLM.

    Args:
        text
    """

    # For example, should we also accept token ids? Or only output token ids?
    texts: List[str]
