# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import importlib
import warnings

import pytest
import torch

import diffusers
from diffusers import pipelines
from diffusers.pipelines.pipeline_loading_utils import (
    ALL_IMPORTABLE_CLASSES,
    get_class_obj_and_candidates,
    simple_get_class_obj,
)


# (pipeline folder, deprecated module path, class name, init kwargs)
RELOCATED_CLASSES = [
    ("ace_step", "diffusers.pipelines.ace_step.modeling_ace_step", "AceStepAudioTokenDetokenizer", {}),
    ("ace_step", "diffusers.pipelines.ace_step.modeling_ace_step", "AceStepAudioTokenizer", {}),
    ("ace_step", "diffusers.pipelines.ace_step.modeling_ace_step", "AceStepConditionEncoder", {}),
    (
        "audioldm2",
        "diffusers.pipelines.audioldm2.modeling_audioldm2",
        "AudioLDM2ProjectionModel",
        {"text_encoder_dim": 8, "text_encoder_1_dim": 8, "langauge_model_dim": 16},
    ),
    (
        "audioldm2",
        "diffusers.pipelines.audioldm2.modeling_audioldm2",
        "AudioLDM2UNet2DConditionModel",
        {
            "block_out_channels": (8, 16),
            "layers_per_block": 1,
            "norm_num_groups": 8,
            "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D"),
            "up_block_types": ("CrossAttnUpBlock2D", "UpBlock2D"),
            "cross_attention_dim": ([None, 8], [None, 8]),
            "attention_head_dim": 2,
        },
    ),
    ("flux", "diffusers.pipelines.flux.modeling_flux", "ReduxImageEncoder", {}),
    ("ltx", "diffusers.pipelines.ltx.modeling_latent_upsampler", "LTXLatentUpsamplerModel", {}),
    ("ltx2", "diffusers.pipelines.ltx2.latent_upsampler", "LTX2LatentUpsamplerModel", {}),
    ("ltx2", "diffusers.pipelines.ltx2.connectors", "LTX2TextConnectors", {}),
    ("ltx2", "diffusers.pipelines.ltx2.vocoder", "LTX2Vocoder", {}),
    ("ltx2", "diffusers.pipelines.ltx2.vocoder", "LTX2VocoderWithBWE", {}),
    ("shap_e", "diffusers.pipelines.shap_e.renderer", "ShapERenderer", {}),
    (
        "stable_audio",
        "diffusers.pipelines.stable_audio.modeling_stable_audio",
        "StableAudioProjectionModel",
        {"text_encoder_dim": 8, "conditioning_dim": 8, "min_value": 0, "max_value": 10},
    ),
    (
        "stable_audio_3",
        "diffusers.pipelines.stable_audio_3.modeling_stable_audio_3",
        "StableAudio3DurationEmbedder",
        {},
    ),
    ("stable_diffusion", "diffusers.pipelines.stable_diffusion.clip_image_project_model", "CLIPImageProjection", {}),
    (
        "stable_diffusion",
        "diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer",
        "StableUnCLIPImageNormalizer",
        {},
    ),
]
IDS = [name for _, _, name, _ in RELOCATED_CLASSES]

# Relocated symbols that are not top-level `diffusers` exports and so are not in `_RELOCATED_PIPELINE_CLASSES`:
# (deprecated module path, new module path, symbol name, call kwargs)
_SHAP_E_RENDERER = "diffusers.models.autoencoders.renderer_shap_e"
_SHAP_E_CAMERA = "diffusers.models.autoencoders.camera_shap_e"
RELOCATED_HELPERS = [
    (
        "diffusers.pipelines.ace_step.modeling_ace_step",
        "diffusers.models.condition_embedders.condition_embedder_ace_step",
        "AceStepLyricEncoder",
        {},
    ),
    (
        "diffusers.pipelines.ace_step.modeling_ace_step",
        "diffusers.models.condition_embedders.condition_embedder_ace_step",
        "AceStepTimbreEncoder",
        {},
    ),
    ("diffusers.pipelines.shap_e.renderer", _SHAP_E_RENDERER, "MLPNeRSTFModel", {}),
    ("diffusers.pipelines.shap_e.renderer", _SHAP_E_RENDERER, "ShapEParamsProjModel", {}),
    (
        "diffusers.pipelines.shap_e.renderer",
        _SHAP_E_RENDERER,
        "VoidNeRFModel",
        {"background": [0.0, 0.0, 0.0], "channel_scale": 1.0},
    ),
    (
        "diffusers.pipelines.shap_e.renderer",
        _SHAP_E_RENDERER,
        "BoundingBoxVolume",
        {"bbox_min": [-1.0, -1.0, -1.0], "bbox_max": [1.0, 1.0, 1.0]},
    ),
    ("diffusers.pipelines.shap_e.renderer", _SHAP_E_RENDERER, "StratifiedRaySampler", {}),
    (
        "diffusers.pipelines.shap_e.renderer",
        _SHAP_E_RENDERER,
        "ImportanceRaySampler",
        {"volume_range": None, "ts": torch.zeros(1, 1, 2, 1), "weights": torch.ones(1, 1, 2, 1)},
    ),
    (
        "diffusers.pipelines.shap_e.renderer",
        _SHAP_E_RENDERER,
        "MLPNeRFModelOutput",
        {
            "density": torch.zeros(1),
            "signed_distance": torch.zeros(1),
            "channels": torch.zeros(1),
            "ts": torch.zeros(1),
        },
    ),
    (
        "diffusers.pipelines.shap_e.camera",
        _SHAP_E_CAMERA,
        "DifferentiableProjectiveCamera",
        {
            "origin": torch.zeros(1, 3),
            "x": torch.zeros(1, 3),
            "y": torch.zeros(1, 3),
            "z": torch.zeros(1, 3),
            "width": 4,
            "height": 4,
            "x_fov": 0.7,
            "y_fov": 0.7,
            "shape": (1,),
        },
    ),
    ("diffusers.pipelines.shap_e.camera", _SHAP_E_CAMERA, "create_pan_cameras", {"size": 4}),
]
HELPER_IDS = [name for _, _, name, _ in RELOCATED_HELPERS]


def _future_warnings(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]


@pytest.mark.parametrize("folder, module, name, kwargs", RELOCATED_CLASSES, ids=IDS)
def test_deprecated_path_warns_and_subclasses_new_class(folder, module, name, kwargs):
    new_cls = getattr(diffusers, name)
    old_cls = getattr(importlib.import_module(module), name)
    assert new_cls.__module__.startswith("diffusers.models.")
    assert issubclass(old_cls, new_cls)

    with torch.device("meta"):
        assert not _future_warnings(lambda: new_cls(**kwargs))
        messages = _future_warnings(lambda: old_cls(**kwargs))
    assert any("deprecated" in message for message in messages), messages


@pytest.mark.parametrize("folder, module, name, kwargs", RELOCATED_CLASSES, ids=IDS)
def test_model_index_entry_resolves_to_new_class(folder, module, name, kwargs):
    new_cls = getattr(diffusers, name)
    class_obj, _ = get_class_obj_and_candidates(folder, name, ALL_IMPORTABLE_CLASSES, pipelines, True)
    assert class_obj is new_cls
    assert simple_get_class_obj(folder, name) is new_cls


@pytest.mark.parametrize("module, new_module, name, kwargs", RELOCATED_HELPERS, ids=HELPER_IDS)
def test_deprecated_helper_path_warns(module, new_module, name, kwargs):
    new_obj = getattr(importlib.import_module(new_module), name)
    old_obj = getattr(importlib.import_module(module), name)
    if isinstance(old_obj, type):
        assert issubclass(old_obj, new_obj)

    with torch.device("meta"):
        assert not _future_warnings(lambda: new_obj(**kwargs))
        messages = _future_warnings(lambda: old_obj(**kwargs))
    assert any(name in message and "deprecated" in message for message in messages), messages
