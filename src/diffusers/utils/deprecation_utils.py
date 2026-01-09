import inspect
import warnings
from typing import Any, Dict, Optional, Union

from packaging import version

from ..utils import logging


logger = logging.get_logger(__name__)

# Mapping for deprecated Transformers classes to their replacements
# This is used to handle models that reference deprecated class names in their configs
# Reference: https://github.com/huggingface/transformers/issues/40822
# Format: {
#     "DeprecatedClassName": {
#         "new_class": "NewClassName",
#         "transformers_version": (">=", "5.0.0"),  # (operation, version) tuple
#     }
# }
_TRANSFORMERS_CLASS_REMAPPING = {
    "CLIPFeatureExtractor": {
        "new_class": "CLIPImageProcessor",
        "transformers_version": (">", "4.57.0"),
    },
}


def _maybe_remap_transformers_class(class_name: str) -> Optional[str]:
    """
    Check if a Transformers class should be remapped to a newer version.

    Args:
        class_name: The name of the class to check

    Returns:
        The new class name if remapping should occur, None otherwise
    """
    if class_name not in _TRANSFORMERS_CLASS_REMAPPING:
        return None

    from .import_utils import is_transformers_version

    mapping = _TRANSFORMERS_CLASS_REMAPPING[class_name]
    operation, required_version = mapping["transformers_version"]

    # Only remap if the transformers version meets the requirement
    if is_transformers_version(operation, required_version):
        new_class = mapping["new_class"]
        logger.warning(f"{class_name} appears to have been deprecated in transformers. Using {new_class} instead.")
        return mapping["new_class"]

    return None


def deprecate(*args, take_from: Optional[Union[Dict, Any]] = None, standard_warn=True, stacklevel=2):
    from .. import __version__

    deprecated_kwargs = take_from
    values = ()
    if not isinstance(args[0], tuple):
        args = (args,)

    for attribute, version_name, message in args:
        if version.parse(version.parse(__version__).base_version) >= version.parse(version_name):
            raise ValueError(
                f"The deprecation tuple {(attribute, version_name, message)} should be removed since diffusers'"
                f" version {__version__} is >= {version_name}"
            )

        warning = None
        if isinstance(deprecated_kwargs, dict) and attribute in deprecated_kwargs:
            values += (deprecated_kwargs.pop(attribute),)
            warning = f"The `{attribute}` argument is deprecated and will be removed in version {version_name}."
        elif hasattr(deprecated_kwargs, attribute):
            values += (getattr(deprecated_kwargs, attribute),)
            warning = f"The `{attribute}` attribute is deprecated and will be removed in version {version_name}."
        elif deprecated_kwargs is None:
            warning = f"`{attribute}` is deprecated and will be removed in version {version_name}."

        if warning is not None:
            warning = warning + " " if standard_warn else ""
            warnings.warn(warning + message, FutureWarning, stacklevel=stacklevel)

    if isinstance(deprecated_kwargs, dict) and len(deprecated_kwargs) > 0:
        call_frame = inspect.getouterframes(inspect.currentframe())[1]
        filename = call_frame.filename
        line_number = call_frame.lineno
        function = call_frame.function
        key, value = next(iter(deprecated_kwargs.items()))
        raise TypeError(f"{function} in {filename} line {line_number - 1} got an unexpected keyword argument `{key}`")

    if len(values) == 0:
        return
    elif len(values) == 1:
        return values[0]
    return values
