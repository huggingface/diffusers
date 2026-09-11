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

import pytest
import torch

from diffusers.loaders.conversion import get_conversion
from diffusers.loaders.conversion.transforms import MergeEqual, Reverse


def test_single_file_config_defaults_have_bidirectional_definitions():
    from diffusers.loaders.conversion.registry import CONVERSION_BUILDERS
    from diffusers.loaders.single_file_model import SINGLE_FILE_CONFIGS

    assert SINGLE_FILE_CONFIGS.keys() <= CONVERSION_BUILDERS.keys()


def test_new_single_file_model_requires_config_and_resolves_subclasses():
    from diffusers import AutoencoderTiny, DiTTransformer2DModel
    from diffusers.loaders.single_file_model import _get_single_file_loadable_mapping_class

    class DerivedDiT(DiTTransformer2DModel):
        pass

    assert _get_single_file_loadable_mapping_class(DerivedDiT) == "DiTTransformer2DModel"
    with pytest.raises(ValueError, match="requires an explicit Diffusers"):
        AutoencoderTiny.from_single_file({}, local_files_only=True)


CASES = [
    ("tests.models.autoencoders.test_models_asymmetric_autoencoder_kl", "AsymmetricAutoencoderKLTesterConfig"),
    ("tests.models.autoencoders.test_models_autoencoder_cosmos", "AutoencoderKLCosmosTesterConfig"),
    ("tests.models.autoencoders.test_models_autoencoder_dc", "AutoencoderDCTesterConfig"),
    ("tests.models.autoencoders.test_models_autoencoder_hunyuan_video", "AutoencoderKLHunyuanVideoTesterConfig"),
    ("tests.models.autoencoders.test_models_autoencoder_kl", "AutoencoderKLTesterConfig"),
    ("tests.models.autoencoders.test_models_autoencoder_kl_ltx2_audio", "AutoencoderKLLTX2AudioTesterConfig"),
    ("tests.models.autoencoders.test_models_autoencoder_kl_minimax_h3", "AutoencoderKLMiniMaxH3TesterConfig"),
    (
        "tests.models.autoencoders.test_models_autoencoder_kl_minimax_h3_audio",
        "AutoencoderKLMiniMaxH3AudioTesterConfig",
    ),
    (
        "tests.models.autoencoders.test_models_autoencoder_kl_temporal_decoder",
        "AutoencoderKLTemporalDecoderTesterConfig",
    ),
    ("tests.models.autoencoders.test_models_autoencoder_ltx2_video", "AutoencoderKLLTX2VideoTesterConfig"),
    ("tests.models.autoencoders.test_models_autoencoder_ltx_video", "AutoencoderKLLTXVideo090TesterConfig"),
    ("tests.models.autoencoders.test_models_autoencoder_ltx_video", "AutoencoderKLLTXVideo091TesterConfig"),
    ("tests.models.autoencoders.test_models_autoencoder_mochi", "AutoencoderKLMochiTesterConfig"),
    ("tests.models.autoencoders.test_models_autoencoder_oobleck", "AutoencoderOobleckTesterConfig"),
    ("tests.models.autoencoders.test_models_autoencoder_same", "AutoencoderSAMETesterConfig"),
    ("tests.models.autoencoders.test_models_autoencoder_tiny", "AutoencoderTinyTesterConfig"),
    ("tests.models.autoencoders.test_models_autoencoder_wan", "AutoencoderKLWanTesterConfig"),
    ("tests.models.autoencoders.test_models_vq", "VQModelTesterConfig"),
    ("tests.models.controlnets.test_models_controlnet_cosmos", "CosmosControlNetModelTesterConfig"),
    ("tests.models.transformers.test_models_dit_transformer2d", "DiTTransformer2DTesterConfig"),
    ("tests.models.transformers.test_models_pixart_transformer2d", "PixArtTransformer2DTesterConfig"),
    ("tests.models.transformers.test_models_prior", "PriorTransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_ace_step", "AceStepTransformer1DModelTesterConfig"),
    ("tests.models.transformers.test_models_transformer_anyflow", "AnyFlowTransformer3DTesterConfig"),
    ("tests.models.transformers.test_models_transformer_anyflow_far", "AnyFlowFARTransformer3DTesterConfig"),
    ("tests.models.transformers.test_models_transformer_aura_flow", "AuraFlowTransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_chroma", "ChromaTransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_cogview3plus", "CogView3PlusTransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_cogview4", "CogView4TransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_cosmos", "CosmosTransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_cosmos", "CosmosTransformerVideoToWorldTesterConfig"),
    ("tests.models.transformers.test_models_transformer_ernie_image", "ErnieImageTransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_flux", "FluxTransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_flux2", "Flux2TransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_flux2", "Flux2TransformerKVCacheTesterConfig"),
    ("tests.models.transformers.test_models_transformer_hidream", "HiDreamTransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_hunyuan_1_5", "HunyuanVideo15TransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_hunyuan_dit", "HunyuanDiTTesterConfig"),
    ("tests.models.transformers.test_models_transformer_hunyuan_video", "HunyuanVideoTransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_hunyuan_video", "HunyuanVideoI2VTransformerTesterConfig"),
    (
        "tests.models.transformers.test_models_transformer_hunyuan_video",
        "HunyuanVideoTokenReplaceTransformerTesterConfig",
    ),
    ("tests.models.transformers.test_models_transformer_longcat_audio_dit", "LongCatAudioDiTTransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_ltx", "LTXTransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_ltx2", "LTX2TransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_lumina", "LuminaNextDiTTesterConfig"),
    ("tests.models.transformers.test_models_transformer_lumina2", "Lumina2TransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_minimax_h3", "MiniMaxH3TransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_minimax_music3", "MiniMaxMusic3Transformer1DTesterConfig"),
    ("tests.models.transformers.test_models_transformer_mochi", "MochiTransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_omnigen", "OmniGenTransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_ovis_image", "OvisImageTransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_prx", "PRXTransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_sana", "SanaTransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_sana_video", "SanaVideoTransformer3DTesterConfig"),
    ("tests.models.transformers.test_models_transformer_sd3", "SD3TransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_sd3", "SD35TransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_stable_audio3", "StableAudio3DiTTesterConfig"),
    ("tests.models.transformers.test_models_transformer_wan", "WanTransformer3DTesterConfig"),
    ("tests.models.transformers.test_models_transformer_wan_animate", "WanAnimateTransformer3DTesterConfig"),
    ("tests.models.transformers.test_models_transformer_wan_animate_2", "WanAnimate2TransformerTesterConfig"),
    ("tests.models.transformers.test_models_transformer_wan_vace", "WanVACETransformer3DTesterConfig"),
    ("tests.models.transformers.test_models_transformer_z_image", "ZImageTransformerTesterConfig"),
    ("tests.models.unets.test_models_unet_1d", "UNet1DModelTesterConfig"),
    ("tests.models.unets.test_models_unet_1d", "UNetRLModelTesterConfig"),
    ("tests.models.unets.test_models_unet_2d", "Unet2DModelTesterConfig"),
    ("tests.models.unets.test_models_unet_2d", "UNetLDMModelTesterConfig"),
    ("tests.models.unets.test_models_unet_2d", "NCSNppModelTesterConfig"),
    ("tests.models.unets.test_models_unet_2d_condition", "UNet2DConditionTesterConfig"),
    ("tests.models.unets.test_models_unet_3d_condition", "UNet3DConditionModelTesterConfig"),
    ("tests.models.unets.test_models_unet_spatiotemporal", "UNetSpatioTemporalConditionModelTesterConfig"),
]


@pytest.mark.parametrize("module_name,tester_name", CASES)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["float32", "float16", "bfloat16"]
)
def test_model_conversion_covers_parameters_and_round_trips(module_name, tester_name, dtype, tmp_path):
    module = importlib.import_module(module_name)
    tester = getattr(module, tester_name)()
    model = tester.model_class(**tester.get_init_dict()).to(dtype=dtype)
    state = {
        key: torch.randn_like(value) if value.is_floating_point() else value.clone()
        for key, value in model.state_dict().items()
    }
    conversion = get_conversion(type(model).__name__, dict(model.config))
    assert conversion.diffusers_keys == set(state)
    # DiT and Lumina imports replicate shared original parameters into separate Diffusers modules.
    for rule in conversion.rules:
        if isinstance(rule.transform, Reverse) and isinstance(rule.transform.transform, MergeEqual):
            for key in rule.diffusers[1:]:
                state[key] = state[rule.diffusers[0]].clone()
    original = conversion.to_original(state)
    restored = conversion.to_diffusers(original)
    assert set(restored) == set(state)
    for key in state:
        torch.testing.assert_close(restored[key], state[key], rtol=0, atol=0)
    original_restored = conversion.to_original(restored)
    for key in original:
        torch.testing.assert_close(original_restored[key], original[key], rtol=0, atol=0)
    if dtype == torch.float32:
        if hasattr(type(model), "from_single_file"):
            model.save_config(tmp_path)
            loaded = type(model).from_single_file(original, config=str(tmp_path), local_files_only=True)
            for key in state:
                torch.testing.assert_close(loaded.state_dict()[key], state[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "module_name,tester_name,prefix",
    [
        (
            "tests.models.unets.test_models_unet_2d_condition",
            "UNet2DConditionTesterConfig",
            "model.diffusion_model.",
        ),
        (
            "tests.models.autoencoders.test_models_autoencoder_kl",
            "AutoencoderKLTesterConfig",
            "first_stage_model.",
        ),
    ],
)
def test_ldm_component_selection_and_ema(module_name, tester_name, prefix):
    from diffusers.loaders import single_file_utils

    tester = getattr(importlib.import_module(module_name), tester_name)()
    model = tester.model_class(**tester.get_init_dict())
    state = model.state_dict()
    config = dict(model.config, _class_name=type(model).__name__)
    conversion = get_conversion(type(model).__name__, config)
    original = conversion.to_original(state)
    checkpoint = {prefix + key: value for key, value in original.items()}
    legacy = single_file_utils.convert_model_checkpoint(checkpoint.copy(), config)
    assert set(legacy) == set(state)
    for key in state:
        torch.testing.assert_close(legacy[key], state[key], rtol=0, atol=0)
    if prefix == "model.diffusion_model.":
        checkpoint.update(
            {"model_ema." + "".join(key.split(".")[1:]): value + 1 for key, value in checkpoint.copy().items()}
        )
        expected = conversion.to_diffusers({key: value + 1 for key, value in original.items()})
        selected = single_file_utils.convert_model_checkpoint(checkpoint, config, extract_ema=True)
        for key in state:
            torch.testing.assert_close(selected[key], expected[key], rtol=0, atol=0)
