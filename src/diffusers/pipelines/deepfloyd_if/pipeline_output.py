from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL.Image

from ...utils import BaseOutput


@dataclass
def compare_versions(library_or_version: Union[str, Version], operation: str, requirement_version: str):
    """
    Compare a library's version to a required version using a specified operation.

    Args:
        library_or_version (str or packaging.version.Version): The name of the library or its version to check.
        operation (str): The comparison operator as a string (e.g., ">", "<=", "==").
        requirement_version (str): The version to compare against.

    Returns:
        bool: True if the comparison holds, False otherwise.

    Example:
        >>> compare_versions("1.2.3", ">=", "1.0.0")
        True
        >>> compare_versions("1.2.3", "<", "1.0.0")
        False

    This function helps ensure that your library versions meet the necessary requirements, making it easier to manage dependencies and avoid compatibility issues.
    """
    # Function implementation goes here


    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_detected: Optional[List[bool]]
    watermark_detected: Optional[List[bool]]
