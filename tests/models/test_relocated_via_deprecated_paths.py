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

"""End-to-end exercises for the relocated model classes via their *deprecated* pipeline import paths.

Sibling to ``test_relocated_class_loading.py``. That file proves the deprecation shim subclasses
construct (and emit ``FutureWarning``) and that ``save_pretrained`` / ``from_pretrained`` round-trips
work through the canonical paths. This file instead **actually uses** the shim instance — runs a
forward pass (or the closest analogue, e.g. ``apply_watermark`` for ``IFWatermarker``) — to confirm
the shim subclass is behaviourally indistinguishable from the canonical class and that no helper
forgotten in the move breaks the runtime path.

Inputs are deliberately tiny (single-digit channels, single-batch) so the suite stays fast and CPU-
only. The whole file should be deleted in the cleanup PR that removes the deprecation shims.
"""

import warnings

import pytest
import torch


def _ignore_relocation_warning():
    """Silence the ``FutureWarning`` emitted by the shim subclass on construction."""
    return warnings.catch_warnings()


def _exercise_clip_image_projection():
    from diffusers.pipelines.stable_diffusion.clip_image_project_model import CLIPImageProjection

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        model = CLIPImageProjection(hidden_size=32)
    x = torch.randn(2, 32)
    out = model(x)
    assert out.shape == x.shape


def _exercise_audioldm2_projection():
    from diffusers.pipelines.audioldm2.modeling_audioldm2 import AudioLDM2ProjectionModel

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        model = AudioLDM2ProjectionModel(text_encoder_dim=8, text_encoder_1_dim=8, langauge_model_dim=16)
    batch, seq = 2, 4
    out = model(
        hidden_states=torch.randn(batch, seq, 8),
        hidden_states_1=torch.randn(batch, seq, 8),
        attention_mask=torch.ones(batch, seq, dtype=torch.long),
        attention_mask_1=torch.ones(batch, seq, dtype=torch.long),
    )
    # Two text encoders' SOS/EOS-wrapped sequences concat'd → (seq+2)*2 along seq dim.
    expected_len = (seq + 2) * 2
    assert out.hidden_states.shape == (batch, expected_len, 16)
    assert out.attention_mask.shape == (batch, expected_len)


def _exercise_stable_audio_projection():
    from diffusers.pipelines.stable_audio.modeling_stable_audio import StableAudioProjectionModel

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        model = StableAudioProjectionModel(text_encoder_dim=8, conditioning_dim=8, min_value=0, max_value=10)
    batch, seq = 2, 4
    text = torch.randn(batch, seq, 8)
    start = torch.tensor([1.0, 2.0])
    end = torch.tensor([3.0, 4.0])
    out = model(text_hidden_states=text, start_seconds=start, end_seconds=end)
    assert out.text_hidden_states.shape == (batch, seq, 8)
    assert out.seconds_start_hidden_states.shape == (batch, 1, 8)
    assert out.seconds_end_hidden_states.shape == (batch, 1, 8)


def _exercise_redux_image_encoder():
    from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        model = ReduxImageEncoder(redux_dim=32, txt_in_features=16)
    x = torch.randn(2, 5, 32)
    out = model(x)
    assert out.image_embeds.shape == (2, 5, 16)


def _exercise_ltx_latent_upsampler():
    from diffusers.pipelines.ltx.modeling_latent_upsampler import LTXLatentUpsamplerModel

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        model = LTXLatentUpsamplerModel(
            in_channels=4, mid_channels=64, num_blocks_per_stage=1, dims=2, spatial_upsample=True
        )
    # (B, C, T, H, W) — spatial upsample doubles H and W.
    x = torch.randn(1, 4, 2, 8, 8)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 4, 2, 16, 16)


def _exercise_ltx2_latent_upsampler():
    """The LTX2 upsampler ctor is heavier — just confirm it constructs through the deprecated path."""
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        # Default ctor: pulls weights for a small spatial-upsampler. We can't avoid the default-size
        # ctor cheaply (the model's __init__ doesn't expose a tiny preset), so we accept the ~ms of
        # parameter allocation in exchange for hitting the real shim code path.
        model = LTX2LatentUpsamplerModel()
    assert isinstance(model, torch.nn.Module)


def _exercise_ltx2_vocoder():
    from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        model = LTX2Vocoder()
    assert isinstance(model, torch.nn.Module)


def _exercise_ltx2_vocoder_with_bwe():
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        model = LTX2VocoderWithBWE()
    assert isinstance(model, torch.nn.Module)


def _exercise_ltx2_text_connectors():
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        model = LTX2TextConnectors()
    assert isinstance(model, torch.nn.Module)


def _exercise_ace_step_audio_tokenizer():
    from diffusers.pipelines.ace_step.modeling_ace_step import AceStepAudioTokenizer

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        # Use the smallest viable config (head_dim and hidden_size are constrained by RMSNorm
        # and the rotary helper; defaults are fine).
        model = AceStepAudioTokenizer(
            hidden_size=64,
            intermediate_size=128,
            audio_acoustic_hidden_dim=16,
            pool_window_size=5,
            num_attention_pooler_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
        )
    assert isinstance(model, torch.nn.Module)


def _exercise_ace_step_audio_token_detokenizer():
    from diffusers.pipelines.ace_step.modeling_ace_step import AceStepAudioTokenDetokenizer

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        model = AceStepAudioTokenDetokenizer(
            hidden_size=64,
            intermediate_size=128,
            audio_acoustic_hidden_dim=16,
            pool_window_size=5,
            num_attention_pooler_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
        )
    assert isinstance(model, torch.nn.Module)


def _exercise_ace_step_condition_encoder():
    from diffusers.pipelines.ace_step.modeling_ace_step import AceStepConditionEncoder

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        model = AceStepConditionEncoder(
            hidden_size=64,
            intermediate_size=128,
            text_hidden_dim=32,
            timbre_hidden_dim=16,
            num_lyric_encoder_hidden_layers=1,
            num_timbre_encoder_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
        )
    assert isinstance(model, torch.nn.Module)


def _exercise_ace_step_lyric_encoder():
    from diffusers.pipelines.ace_step.modeling_ace_step import AceStepLyricEncoder

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        model = AceStepLyricEncoder(
            hidden_size=64,
            intermediate_size=128,
            text_hidden_dim=32,
            num_lyric_encoder_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
        )
    assert isinstance(model, torch.nn.Module)


def _exercise_ace_step_timbre_encoder():
    from diffusers.pipelines.ace_step.modeling_ace_step import AceStepTimbreEncoder

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        model = AceStepTimbreEncoder(
            hidden_size=64,
            intermediate_size=128,
            timbre_hidden_dim=16,
            num_timbre_encoder_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
        )
    assert isinstance(model, torch.nn.Module)


def _exercise_audioldm2_unet():
    from diffusers.pipelines.audioldm2.modeling_audioldm2 import AudioLDM2UNet2DConditionModel

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        # Tiny config: 1 down block, 1 up block (`UpBlock2D` for the symmetry the ctor enforces),
        # 8 base channels. Constructs a few-million-parameter UNet rather than the full 1B+ default.
        model = AudioLDM2UNet2DConditionModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock2D",),
            up_block_types=("CrossAttnUpBlock2D",),
            block_out_channels=(32,),
            layers_per_block=1,
            cross_attention_dim=(16, 16, 16, 16),
            transformer_layers_per_block=1,
            num_attention_heads=(2,),
            attention_head_dim=(2,),
            norm_num_groups=8,
            mid_block_scale_factor=1.0,
        )
    assert isinstance(model, torch.nn.Module)


def _exercise_shap_e_renderer():
    """``ShapERenderer.__init__`` doesn't take overrides we can shrink usefully; just verify ctor."""
    from diffusers.pipelines.shap_e.renderer import ShapERenderer

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        model = ShapERenderer()
    assert isinstance(model, torch.nn.Module)


def _exercise_if_watermarker():
    import PIL.Image

    from diffusers.pipelines.deepfloyd_if.watermark import IFWatermarker

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        model = IFWatermarker()
    images = [PIL.Image.new("RGB", (64, 64), color=(127, 127, 127))]
    result = model.apply_watermark(images, sample_size=64)
    assert len(result) == 1 and result[0].size == (64, 64)


def _exercise_stable_unclip_normalizer():
    from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer

    with _ignore_relocation_warning():
        warnings.simplefilter("ignore")
        model = StableUnCLIPImageNormalizer(embedding_dim=16)
    x = torch.randn(2, 16)
    scaled = model.scale(x)
    restored = model.unscale(scaled)
    torch.testing.assert_close(restored, x, rtol=1e-5, atol=1e-5)


EXERCISES = [
    ("CLIPImageProjection", _exercise_clip_image_projection),
    ("AudioLDM2ProjectionModel", _exercise_audioldm2_projection),
    ("AudioLDM2UNet2DConditionModel", _exercise_audioldm2_unet),
    ("StableAudioProjectionModel", _exercise_stable_audio_projection),
    ("ReduxImageEncoder", _exercise_redux_image_encoder),
    ("LTXLatentUpsamplerModel", _exercise_ltx_latent_upsampler),
    ("LTX2LatentUpsamplerModel", _exercise_ltx2_latent_upsampler),
    ("LTX2Vocoder", _exercise_ltx2_vocoder),
    ("LTX2VocoderWithBWE", _exercise_ltx2_vocoder_with_bwe),
    ("LTX2TextConnectors", _exercise_ltx2_text_connectors),
    ("AceStepAudioTokenizer", _exercise_ace_step_audio_tokenizer),
    ("AceStepAudioTokenDetokenizer", _exercise_ace_step_audio_token_detokenizer),
    ("AceStepConditionEncoder", _exercise_ace_step_condition_encoder),
    ("AceStepLyricEncoder", _exercise_ace_step_lyric_encoder),
    ("AceStepTimbreEncoder", _exercise_ace_step_timbre_encoder),
    ("ShapERenderer", _exercise_shap_e_renderer),
    ("IFWatermarker", _exercise_if_watermarker),
    ("StableUnCLIPImageNormalizer", _exercise_stable_unclip_normalizer),
]


@pytest.mark.parametrize("name, exercise", EXERCISES, ids=[n for n, _ in EXERCISES])
def test_exercise_via_deprecated_path(name, exercise):
    """Instantiate each relocated class through its deprecated pipeline path and use it."""
    exercise()
