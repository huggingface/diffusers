# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loading + import-path tests for classes relocated under ``models/condition_embedders``,
``models/autoencoders``, ``models/unets``, and ``models/others``.

The point of this file is narrow: confirm that (a) the new canonical import paths work,
(b) the deprecated import paths still resolve and emit a ``FutureWarning`` on instantiation,
and (c) ``save_pretrained`` / ``from_pretrained`` round-trips through a temp directory using the
new path — i.e. ``_class_name`` resolution in saved configs is unaffected by the move. No real
checkpoints are downloaded; instantiation kwargs are tiny by design so this file stays fast and
the entire suite can be deleted once the deprecation shims are removed.
"""

import tempfile
import warnings

import pytest


CANONICAL_CASES = [
    # (top-level import name, factory returning a small instance)
    ("CLIPImageProjection", lambda C: C(hidden_size=32)),
    (
        "AudioLDM2ProjectionModel",
        lambda C: C(text_encoder_dim=8, text_encoder_1_dim=8, langauge_model_dim=16),
    ),
    ("StableAudioProjectionModel", lambda C: C(text_encoder_dim=8, conditioning_dim=8, min_value=0, max_value=10)),
    ("ReduxImageEncoder", lambda C: C(redux_dim=32, txt_in_features=16)),
    (
        "LTXLatentUpsamplerModel",
        lambda C: C(in_channels=4, mid_channels=64, num_blocks_per_stage=1, dims=2, spatial_upsample=True),
    ),
]


SHIM_CASES = [
    # (old module dotted path, class name, factory)
    (
        "diffusers.pipelines.stable_diffusion.clip_image_project_model",
        "CLIPImageProjection",
        lambda C: C(hidden_size=32),
    ),
    (
        "diffusers.pipelines.audioldm2.modeling_audioldm2",
        "AudioLDM2ProjectionModel",
        lambda C: C(text_encoder_dim=8, text_encoder_1_dim=8, langauge_model_dim=16),
    ),
    (
        "diffusers.pipelines.stable_audio.modeling_stable_audio",
        "StableAudioProjectionModel",
        lambda C: C(text_encoder_dim=8, conditioning_dim=8, min_value=0, max_value=10),
    ),
    (
        "diffusers.pipelines.flux.modeling_flux",
        "ReduxImageEncoder",
        lambda C: C(redux_dim=32, txt_in_features=16),
    ),
    (
        "diffusers.pipelines.ltx.modeling_latent_upsampler",
        "LTXLatentUpsamplerModel",
        lambda C: C(in_channels=4, mid_channels=64, num_blocks_per_stage=1, dims=2, spatial_upsample=True),
    ),
    (
        "diffusers.pipelines.deepfloyd_if.watermark",
        "IFWatermarker",
        lambda C: C(),
    ),
    (
        "diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer",
        "StableUnCLIPImageNormalizer",
        lambda C: C(embedding_dim=16),
    ),
]


@pytest.mark.parametrize("name, factory", CANONICAL_CASES, ids=[c[0] for c in CANONICAL_CASES])
def test_canonical_import_and_save_load_roundtrip(name, factory):
    """The new canonical path imports, instantiates, and survives a save/load round-trip."""
    import diffusers

    cls = getattr(diffusers, name)
    model = factory(cls)

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        loaded = cls.from_pretrained(tmp)

    assert isinstance(loaded, cls)
    # Config resolution check: the saved `_class_name` must resolve back to a real class.
    assert loaded.config._class_name == name


@pytest.mark.parametrize(
    "module, name, factory",
    SHIM_CASES,
    ids=[f"{m.rsplit('.', 1)[-1]}:{n}" for m, n, _ in SHIM_CASES],
)
def test_deprecated_path_warns_on_instantiation(module, name, factory):
    """Importing from the old pipeline path still works but instantiation emits FutureWarning."""
    import importlib

    mod = importlib.import_module(module)
    cls = getattr(mod, name)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        factory(cls)

    assert any(issubclass(w.category, FutureWarning) and "deprecated" in str(w.message).lower() for w in caught), (
        f"expected a FutureWarning containing 'deprecated' for {module}.{name}, got: {[str(w.message) for w in caught]}"
    )


@pytest.mark.parametrize(
    "module, name, factory",
    SHIM_CASES,
    ids=[f"{m.rsplit('.', 1)[-1]}:{n}" for m, n, _ in SHIM_CASES],
)
def test_deprecated_path_isinstance_compatible(module, name, factory):
    """An instance built via the shim subclass is still an instance of the canonical class."""
    import importlib

    shim_cls = getattr(importlib.import_module(module), name)

    # The canonical class lives in either top-level diffusers (most moves) or
    # diffusers.models.others (pipeline-local utilities like IFWatermarker).
    try:
        import diffusers

        canonical_cls = getattr(diffusers, name)
    except AttributeError:
        from diffusers.models import others

        canonical_cls = getattr(others, name)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        instance = factory(shim_cls)

    assert isinstance(instance, canonical_cls)
