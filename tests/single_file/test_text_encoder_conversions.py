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

from unittest.mock import patch

import pytest
import torch
import transformers

from diffusers.loaders.conversion import get_conversion


@pytest.mark.parametrize("projection", [False, True])
def test_pipeline_openclip_loading_uses_shared_conversion(projection):
    from diffusers.loaders.single_file_utils import create_diffusers_clip_model_from_ldm

    cls = transformers.CLIPTextModelWithProjection if projection else transformers.CLIPTextModel
    config = transformers.CLIPTextConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=23 if not projection else 2,
        vocab_size=16,
        max_position_embeddings=8,
        projection_dim=8,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = cls(config)
    conversion = get_conversion(cls.__name__, dict(config.to_dict(), original_format="openclip"))
    original = conversion.to_original(model.state_dict())
    prefix = "cond_stage_model.model."
    checkpoint = {prefix + key: value for key, value in original.items()}
    checkpoint[prefix + "logit_scale"] = torch.ones(1)
    if not projection:
        checkpoint[prefix + "text_projection"] = torch.randn(8, 8)
        checkpoint[prefix + "transformer.resblocks.23.attn.in_proj_weight"] = torch.randn(24, 8)
    keys = set(checkpoint)
    with patch.object(cls.config_class, "from_pretrained", return_value=config):
        loaded = create_diffusers_clip_model_from_ldm(cls, checkpoint, config="unused", local_files_only=True)
    assert set(checkpoint) == keys
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], tensor, rtol=0, atol=0)


CLIP_CONFIG = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "vocab_size": 32,
    "max_position_embeddings": 16,
    "image_size": 16,
    "patch_size": 8,
    "projection_dim": 8,
}
CLAP_TEXT_CONFIG = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "vocab_size": 32,
    "max_position_embeddings": 16,
    "projection_dim": 8,
}
CLAP_AUDIO_CONFIG = {
    "hidden_size": 16,
    "patch_embeds_hidden_size": 8,
    "depths": [1, 2],
    "num_attention_heads": [2, 2],
    "spec_size": 16,
    "num_mel_bins": 4,
    "patch_size": 2,
    "patch_stride": [2, 2],
    "window_size": 2,
    "projection_dim": 8,
}
CASES = [
    (model, "CLIPVisionConfig" if "Vision" in model else "CLIPTextConfig", CLIP_CONFIG, fmt)
    for model in ("CLIPTextModel", "CLIPTextModelWithProjection", "CLIPVisionModel", "CLIPVisionModelWithProjection")
    for fmt in ("clip", "openclip")
] + [
    ("ContextCLIPTextModel", "CLIPTextConfig", CLIP_CONFIG, "clip"),
    ("ContextCLIPTextModel", "CLIPTextConfig", CLIP_CONFIG, "openclip"),
    (
        "UMT5EncoderModel",
        "UMT5Config",
        {"d_model": 16, "d_ff": 32, "d_kv": 8, "num_heads": 2, "num_layers": 2, "vocab_size": 32},
        None,
    ),
    (
        "SpeechT5HifiGan",
        "SpeechT5HifiGanConfig",
        {
            "model_in_dim": 8,
            "upsample_initial_channel": 16,
            "upsample_rates": [2, 2],
            "upsample_kernel_sizes": [4, 4],
            "resblock_kernel_sizes": [3],
            "resblock_dilation_sizes": [[1, 3, 5]],
        },
        None,
    ),
    ("ClapTextModelWithProjection", "ClapTextConfig", CLAP_TEXT_CONFIG, None),
    ("ClapAudioModel", "ClapAudioConfig", CLAP_AUDIO_CONFIG, None),
    ("ClapAudioModelWithProjection", "ClapAudioConfig", CLAP_AUDIO_CONFIG, None),
    (
        "ClapModel",
        "ClapConfig",
        {"text_config": CLAP_TEXT_CONFIG, "audio_config": CLAP_AUDIO_CONFIG, "projection_dim": 8},
        None,
    ),
    (
        "ClapModel",
        "ClapConfig",
        {
            "text_config": CLAP_TEXT_CONFIG,
            "audio_config": {**CLAP_AUDIO_CONFIG, "enable_fusion": True},
            "projection_dim": 8,
        },
        None,
    ),
    (
        "T5EncoderModel",
        "T5Config",
        {"vocab_size": 32, "d_model": 16, "d_ff": 32, "d_kv": 8, "num_layers": 2, "num_heads": 2},
        None,
    ),
    (
        "Qwen3Model",
        "Qwen3Config",
        {
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
        },
        None,
    ),
    (
        "Qwen3ForCausalLM",
        "Qwen3Config",
        {
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
        },
        None,
    ),
    (
        "Blip2QFormerModel",
        "Blip2Config",
        {
            "qformer_config": {
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "encoder_hidden_size": 16,
            },
            "vision_config": {
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "image_size": 16,
                "patch_size": 8,
            },
        },
        None,
    ),
]


@pytest.mark.parametrize(
    "name,config_name,values,original_format", CASES, ids=[f"{name}-{fmt}" for name, _, _, fmt in CASES]
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["float32", "float16", "bfloat16"]
)
def test_external_encoder_conversion(name, config_name, values, original_format, dtype):
    config = getattr(transformers, config_name)(**values)
    if name == "Blip2QFormerModel":
        from diffusers.pipelines.deprecated.blip_diffusion.modeling_blip2 import Blip2QFormerModel

        # Tokenization is unused by state-dict conversion; avoid downloading a vocabulary to construct the model.
        with patch("diffusers.pipelines.deprecated.blip_diffusion.modeling_blip2.BertTokenizer.from_pretrained"):
            model = Blip2QFormerModel(config)
    elif name == "ContextCLIPTextModel":
        from diffusers.pipelines.deprecated.blip_diffusion.modeling_ctx_clip import ContextCLIPTextModel

        model = ContextCLIPTextModel(config)
    else:
        model = getattr(transformers, name)(config)
    model.to(dtype=dtype)
    state = model.state_dict()
    config = model.config.to_dict()
    if original_format is not None:
        config["original_format"] = original_format
    conversion = get_conversion(name, config)
    assert conversion.diffusers_keys == set(state)
    original = conversion.to_original(state)
    restored = conversion.to_diffusers(original)
    for key in state:
        torch.testing.assert_close(restored[key], state[key], rtol=0, atol=0)
