import json
import os
from typing import Optional

from ..utils import _add_variant
from .import_utils import is_flashpack_available
from .logging import get_logger


logger = get_logger(__name__)


def save_flashpack(
    model,
    save_directory: str,
    variant: Optional[str] = None,
    is_main_process: bool = True,
):
    """
    Save model weights in FlashPack format along with a metadata config.

    Args:
        model: Diffusers model instance
        save_directory (`str`): Directory to save weights
        variant (`str`, *optional*): Model variant
    """
    if not is_flashpack_available():
        raise ImportError(
            "The `use_flashpack=True` argument requires the `flashpack` package. "
            "Install it with `pip install flashpack`."
        )

    from flashpack import pack_to_file

    os.makedirs(save_directory, exist_ok=True)

    weights_name = _add_variant("model.flashpack", variant)
    weights_path = os.path.join(save_directory, weights_name)
    config_path = os.path.join(save_directory, "flashpack_config.json")

    try:
        target_dtype = getattr(model, "dtype", None)
        logger.warning(f"Dtype used for FlashPack save: {target_dtype}")

        # 1. Save binary weights
        pack_to_file(model, weights_path, target_dtype=target_dtype)

        # 2. Save config metadata (best-effort)
        if hasattr(model, "config"):
            try:
                if hasattr(model.config, "to_dict"):
                    config_data = model.config.to_dict()
                else:
                    config_data = dict(model.config)

                with open(config_path, "w") as f:
                    json.dump(config_data, f, indent=4)

            except Exception as config_err:
                logger.warning(f"FlashPack weights saved, but config serialization failed: {config_err}")

    except Exception as e:
        logger.error(f"Failed to save weights in FlashPack format: {e}")
        raise


def load_flashpack(model, flashpack_file: str):
    """
    Assign FlashPack weights from a file into an initialized PyTorch model.
    """
    if not is_flashpack_available():
        raise ImportError("FlashPack weights require the `flashpack` package. Install with `pip install flashpack`.")

    from flashpack import assign_from_file

    logger.warning(f"Loading FlashPack weights from {flashpack_file}")

    try:
        assign_from_file(model, flashpack_file)
    except Exception as e:
        raise RuntimeError(f"Failed to load FlashPack weights from {flashpack_file}") from e
