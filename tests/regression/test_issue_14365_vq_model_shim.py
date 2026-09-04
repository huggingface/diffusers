"""Regression tests for issue #14365 — removal of the vq_model.py shim file.

The shim at ``src/diffusers/models/vq_model.py`` wraps the real implementations in
``src/diffusers/models/autoencoders/vq_model.py`` with deprecation warnings.  This
shim must be deleted so that the old import path raises a clear ``ImportError`` and
the deprecated Wuerstchen PaellaVQModel is updated to import directly from the
canonical autoencoders location.
"""

import os

import pytest


# ---------------------------------------------------------------------------
# Test A — public API must work
# ---------------------------------------------------------------------------


def test_vq_model_importable_from_public_api():
    """``from diffusers import VQModel`` must succeed and return the real class."""
    from diffusers import VQModel

    assert isinstance(VQModel, type), "VQModel must be a class"
    assert VQModel.__module__ == "diffusers.models.autoencoders.vq_model", (
        f"Expected __module__ 'diffusers.models.autoencoders.vq_model', got {VQModel.__module__!r}"
    )


# ---------------------------------------------------------------------------
# Test B — VQEncoderOutput importable from autoencoders path
# ---------------------------------------------------------------------------


def test_vq_encoder_output_available():
    """``from diffusers.models.autoencoders.vq_model import VQEncoderOutput`` must succeed."""
    from diffusers.models.autoencoders.vq_model import VQEncoderOutput

    assert isinstance(VQEncoderOutput, type), "VQEncoderOutput must be a class"


# ---------------------------------------------------------------------------
# Test C — deprecated shim file must NOT exist
# ---------------------------------------------------------------------------


def test_deprecated_shim_file_removed():
    """The shim file ``src/diffusers/models/vq_model.py`` must not exist on disk."""
    import diffusers.models

    base = os.path.dirname(diffusers.models.__file__)
    shim_path = os.path.join(base, "vq_model.py")
    assert not os.path.exists(shim_path), (
        f"Shim file still exists at {shim_path!r}. It must be removed so that the old import path raises ImportError."
    )


# ---------------------------------------------------------------------------
# Test D — old import path must raise ImportError
# ---------------------------------------------------------------------------


def test_deprecated_vq_model_import_fails():
    """``from diffusers.models.vq_model import VQModel`` must raise ImportError."""
    with pytest.raises((ImportError, ModuleNotFoundError)):
        # Use __import__ so we control the name passed to the import machinery
        __import__("diffusers.models.vq_model", fromlist=["VQModel"])


# ---------------------------------------------------------------------------
# Test E — PaellaVQModel must use VQEncoderOutput from autoencoders
# ---------------------------------------------------------------------------


def test_paella_vq_model_vq_encoder_output_works():
    """PaellaVQModel must import and use VQEncoderOutput from the canonical path.

    This requires that PaellaVQModel imports VQEncoderOutput from the canonical
    ``diffusers.models.autoencoders.vq_model`` path, not from the deprecated shim.

    BEFORE the fix: importing PaellaVQModel raises ``ValueError`` because the
    shim's ``deprecate()`` deadline has passed.

    AFTER the fix: PaellaVQModel imports successfully and its VQEncoderOutput
    reference resolves to the canonical autoencoders path.
    """
    from diffusers.pipelines.deprecated.wuerstchen.modeling_paella_vq_model import PaellaVQModel

    assert PaellaVQModel is not None, "PaellaVQModel must be importable"

    import diffusers.pipelines.deprecated.wuerstchen.modeling_paella_vq_model as _paella_mod
    from diffusers.models.autoencoders.vq_model import VQEncoderOutput as CanonicalOutput

    assert _paella_mod.VQEncoderOutput is CanonicalOutput, (
        f"PaellaVQModel must use VQEncoderOutput from autoencoders.vq_model, got {_paella_mod.VQEncoderOutput}"
    )
