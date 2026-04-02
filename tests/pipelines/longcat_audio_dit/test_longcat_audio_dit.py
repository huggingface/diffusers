import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import UMT5Config, UMT5EncoderModel

from diffusers import LongCatAudioDiTPipeline, LongCatAudioDiTTransformer, LongCatAudioDiTVae
from tests.testing_utils import require_torch_accelerator, slow, torch_device


class DummyTokenizer:
    model_max_length = 16

    def __call__(self, texts, padding="longest", truncation=True, max_length=None, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        batch = len(texts)
        return type(
            "TokenBatch",
            (),
            {
                "input_ids": torch.ones(batch, 4, dtype=torch.long),
                "attention_mask": torch.ones(batch, 4, dtype=torch.long),
            },
        )


def _build_components():
    text_encoder = UMT5EncoderModel(UMT5Config(d_model=32, num_layers=1, num_heads=4, d_ff=64, vocab_size=128))
    transformer = LongCatAudioDiTTransformer(
        dit_dim=64,
        dit_depth=2,
        dit_heads=4,
        dit_text_dim=32,
        latent_dim=8,
        text_conv=False,
    )
    vae = LongCatAudioDiTVae(
        in_channels=1,
        channels=16,
        c_mults=[1, 2],
        strides=[2],
        latent_dim=8,
        encoder_latent_dim=16,
        downsampling_ratio=2,
        sample_rate=24000,
    )
    return text_encoder, transformer, vae


def test_longcat_audio_dit_vae_import():
    assert LongCatAudioDiTVae is not None


def test_longcat_audio_pipeline_constructs():
    text_encoder, transformer, vae = _build_components()
    pipe = LongCatAudioDiTPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=DummyTokenizer(), transformer=transformer
    )
    assert pipe is not None


def test_longcat_audio_pipeline_forward_pt_output():
    text_encoder, transformer, vae = _build_components()
    pipe = LongCatAudioDiTPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=DummyTokenizer(), transformer=transformer
    )

    result = pipe(
        prompt="soft ocean ambience", audio_end_in_s=0.1, num_inference_steps=2, guidance_scale=1.0, output_type="pt"
    )

    assert result.audios.ndim == 3
    assert result.audios.shape[0] == 1
    assert result.audios.shape[1] == 1
    assert result.audios.shape[-1] > 0


def test_longcat_audio_pipeline_from_pretrained_local_dir(tmp_path, monkeypatch):
    text_encoder, transformer, vae = _build_components()
    model_dir = tmp_path / "longcat-audio-dit"
    model_dir.mkdir()

    config = {
        "dit_dim": 64,
        "dit_depth": 2,
        "dit_heads": 4,
        "dit_text_dim": 32,
        "latent_dim": 8,
        "dit_dropout": 0.0,
        "dit_bias": True,
        "dit_cross_attn": True,
        "dit_adaln_type": "global",
        "dit_adaln_use_text_cond": True,
        "dit_long_skip": True,
        "dit_text_conv": False,
        "dit_qk_norm": True,
        "dit_cross_attn_norm": False,
        "dit_eps": 1e-6,
        "dit_use_latent_condition": True,
        "sampling_rate": 24000,
        "latent_hop": 2,
        "max_wav_duration": 30.0,
        "text_norm_feat": True,
        "text_add_embed": True,
        "text_encoder_model": "dummy-umt5",
        "text_encoder_config": text_encoder.config.to_dict(),
        "vae_config": {**dict(vae.config), "model_type": "longcat_audio_dit_vae"},
    }
    with (model_dir / "config.json").open("w") as handle:
        json.dump(config, handle)

    state_dict = {}
    state_dict.update({f"text_encoder.{k}": v for k, v in text_encoder.state_dict().items() if k != "shared.weight"})
    state_dict.update({f"transformer.{k}": v for k, v in transformer.state_dict().items()})
    state_dict.update({f"vae.{k}": v for k, v in vae.state_dict().items()})
    save_file(state_dict, model_dir / "model.safetensors")

    monkeypatch.setattr(
        "diffusers.pipelines.longcat_audio_dit.pipeline_longcat_audio_dit.T5Tokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    pipe = LongCatAudioDiTPipeline.from_pretrained(model_dir, local_files_only=True)
    result = pipe(
        prompt="soft ocean ambience", audio_end_in_s=0.1, num_inference_steps=2, guidance_scale=1.0, output_type="pt"
    )

    assert isinstance(pipe, LongCatAudioDiTPipeline)
    assert pipe.sample_rate == 24000
    assert pipe.latent_hop == 2
    assert result.audios.ndim == 3
    assert result.audios.shape[-1] > 0


def test_longcat_audio_top_level_imports():
    assert LongCatAudioDiTPipeline is not None
    assert LongCatAudioDiTTransformer is not None
    assert LongCatAudioDiTVae is not None


@slow
@require_torch_accelerator
def test_longcat_audio_pipeline_from_pretrained_real_local_weights():
    model_path = Path(os.getenv("LONGCAT_AUDIO_DIT_MODEL_PATH", "/data/models/meituan-longcat/LongCat-AudioDiT-1B"))
    tokenizer_path_env = os.getenv("LONGCAT_AUDIO_DIT_TOKENIZER_PATH")
    if tokenizer_path_env is None:
        pytest.skip("LONGCAT_AUDIO_DIT_TOKENIZER_PATH is not set")
    tokenizer_path = Path(tokenizer_path_env)

    if not model_path.exists():
        pytest.skip(f"LongCat-AudioDiT model path not found: {model_path}")
    if not tokenizer_path.exists():
        pytest.skip(f"LongCat-AudioDiT tokenizer path not found: {tokenizer_path}")

    pipe = LongCatAudioDiTPipeline.from_pretrained(
        model_path,
        tokenizer=tokenizer_path,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipe = pipe.to(torch_device)

    result = pipe(
        prompt="A calm ocean wave ambience with soft wind in the background.",
        audio_end_in_s=2.0,
        num_inference_steps=2,
        guidance_scale=4.0,
        output_type="pt",
    )

    assert result.audios.ndim == 3
    assert result.audios.shape[0] == 1
    assert result.audios.shape[1] == 1
    assert result.audios.shape[-1] > 0
