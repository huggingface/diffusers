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

import importlib
import warnings

import pytest
import torch


# (deprecated module path, class name, kwargs). Most ctors have full defaults so ``kwargs`` is
# empty; only ``AudioLDM2ProjectionModel`` and ``StableAudioProjectionModel`` have required
# positional args, so we pass the smallest values that satisfy the signature.
DEPRECATED_PATHS = [
    ("diffusers.pipelines.stable_diffusion.clip_image_project_model", "CLIPImageProjection", {}),
    (
        "diffusers.pipelines.audioldm2.modeling_audioldm2",
        "AudioLDM2ProjectionModel",
        {"text_encoder_dim": 8, "text_encoder_1_dim": 8, "langauge_model_dim": 16},
    ),
    ("diffusers.pipelines.audioldm2.modeling_audioldm2", "AudioLDM2UNet2DConditionModel", {}),
    (
        "diffusers.pipelines.stable_audio.modeling_stable_audio",
        "StableAudioProjectionModel",
        {"text_encoder_dim": 8, "conditioning_dim": 8, "min_value": 0, "max_value": 10},
    ),
    ("diffusers.pipelines.flux.modeling_flux", "ReduxImageEncoder", {}),
    ("diffusers.pipelines.ltx.modeling_latent_upsampler", "LTXLatentUpsamplerModel", {}),
    ("diffusers.pipelines.ltx2.latent_upsampler", "LTX2LatentUpsamplerModel", {}),
    ("diffusers.pipelines.ltx2.vocoder", "LTX2Vocoder", {}),
    ("diffusers.pipelines.ltx2.vocoder", "LTX2VocoderWithBWE", {}),
    ("diffusers.pipelines.ltx2.connectors", "LTX2TextConnectors", {}),
    ("diffusers.pipelines.ace_step.modeling_ace_step", "AceStepAudioTokenizer", {}),
    ("diffusers.pipelines.ace_step.modeling_ace_step", "AceStepAudioTokenDetokenizer", {}),
    ("diffusers.pipelines.ace_step.modeling_ace_step", "AceStepConditionEncoder", {}),
    ("diffusers.pipelines.ace_step.modeling_ace_step", "AceStepLyricEncoder", {}),
    ("diffusers.pipelines.ace_step.modeling_ace_step", "AceStepTimbreEncoder", {}),
    ("diffusers.pipelines.shap_e.renderer", "ShapERenderer", {}),
    ("diffusers.pipelines.deepfloyd_if.watermark", "IFWatermarker", {}),
    ("diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer", "StableUnCLIPImageNormalizer", {}),
]


@pytest.mark.parametrize(
    "module, name, kwargs",
    DEPRECATED_PATHS,
    ids=[name for _, name, _ in DEPRECATED_PATHS],
)
def test_deprecated_path_warns_on_use(module, name, kwargs):
    """Constructing the relocated class through its deprecated pipeline path emits FutureWarning.

    Instantiation runs under ``torch.device("meta")`` so the test stays fast and CPU-only — the
    parameters are meta tensors and no real memory is allocated. We only verify the deprecation
    signal here; functional behaviour of each class is covered by its own dedicated tests at the
    canonical model path.
    """
    mod = importlib.import_module(module)
    cls = getattr(mod, name)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.device("meta"):
            cls(**kwargs)

    assert any(issubclass(w.category, FutureWarning) and "deprecated" in str(w.message).lower() for w in caught), (
        f"expected a FutureWarning containing 'deprecated' for {module}.{name}, got: {[str(w.message) for w in caught]}"
    )
