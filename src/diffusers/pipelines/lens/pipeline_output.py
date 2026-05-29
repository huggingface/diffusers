from dataclasses import dataclass

from ...utils import BaseOutput


@dataclass
class LensPipelineOutput(BaseOutput):
    images: object
