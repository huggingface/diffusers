import importlib
import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import CLIPConfig, CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPVisionConfig

from diffusers import AutoencoderKL, AutoencoderTiny, UNet2DConditionModel
from diffusers.loaders.conversion import get_conversion
from diffusers.loaders.conversion.configs import get_config_preset, list_config_presets
from diffusers.loaders.conversion.io import Checkpoint, convert_checkpoint
from diffusers.loaders.conversion.pipeline import export_pipeline_checkpoint
from diffusers.loaders.conversion.source import MergedCheckpoint, load_source_manifest
from diffusers.pipelines.deepfloyd_if.safety_checker import IFSafetyChecker
from diffusers.pipelines.deprecated.paint_by_example.image_encoder import PaintByExampleImageEncoder
from diffusers.pipelines.deprecated.vq_diffusion.pipeline_vq_diffusion import LearnedClassifierFreeSamplingEmbeddings
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import LDMBertConfig, LDMBertModel


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["bert", "paint", "if", "learned", "empty"])
def test_legacy_pipeline_component_conversion(kind, dtype):
    if kind == "bert":
        config = LDMBertConfig(
            vocab_size=32,
            max_position_embeddings=16,
            d_model=16,
            encoder_layers=2,
            encoder_ffn_dim=32,
            encoder_attention_heads=2,
            head_dim=8,
        )
        model = LDMBertModel(config)
    elif kind == "paint":
        config = CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            projection_dim=8,
            image_size=16,
            patch_size=8,
        )
        model = PaintByExampleImageEncoder(config)
    elif kind == "if":
        config = CLIPConfig(
            vision_config={
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "projection_dim": 8,
                "image_size": 16,
                "patch_size": 8,
            }
        )
        model = IFSafetyChecker(config)
    else:
        config = {"learnable": kind == "learned", "hidden_size": 8, "length": 4}
        model = LearnedClassifierFreeSamplingEmbeddings(**config)
    model = model.to(dtype)
    config = config.to_dict() if hasattr(config, "to_dict") else config
    conversion = get_conversion(type(model).__name__, config)
    state = model.state_dict()
    assert conversion.diffusers_keys == state.keys()
    original = conversion.to_original(state)
    restored = conversion.to_diffusers(original)
    for key, value in state.items():
        torch.testing.assert_close(restored[key], value, rtol=0, atol=0)


def test_tiny_vae_source_manifest_and_duplicates(tmp_path):
    model = AutoencoderTiny(
        encoder_block_out_channels=(8, 8),
        decoder_block_out_channels=(8, 8),
        num_encoder_blocks=(1, 1),
        num_decoder_blocks=(1, 1),
    )
    config = dict(model.config)
    original = get_conversion("AutoencoderTiny", config).to_original(model.state_dict())
    sources = []
    for part in ("encoder", "decoder"):
        state = {key.removeprefix(part + "."): value for key, value in original.items() if key.startswith(part + ".")}
        save_file(state, tmp_path / f"{part}.safetensors")
        sources.append({"path": f"{part}.safetensors", "output_prefix": part + "."})
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"sources": sources}))
    merged = load_source_manifest(path)
    output = convert_checkpoint(merged, tmp_path / "output", config=config, model_class="AutoencoderTiny")
    for key, value in Checkpoint(output).items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="Duplicate source tensor"):
        MergedCheckpoint([("", merged), ("", merged)])


@pytest.mark.parametrize("pipeline_format", ["sd", "sdxl"])
@pytest.mark.parametrize("output_format", ["safetensors", "pytorch"])
def test_pipeline_checkpoint_export(tmp_path, pipeline_format, output_format):
    unet = UNet2DConditionModel(
        block_out_channels=(32,),
        down_block_types=("CrossAttnDownBlock2D",),
        up_block_types=("CrossAttnUpBlock2D",),
        layers_per_block=1,
        cross_attention_dim=32,
    )
    vae = AutoencoderKL(
        block_out_channels=(32,),
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        layers_per_block=1,
    )
    text_config = CLIPTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        projection_dim=8,
        max_position_embeddings=8,
    )
    text = CLIPTextModel(text_config)
    components = {"unet": unet, "vae": vae, "text_encoder": text}
    specs = {
        "unet": ("model.diffusion_model.", None),
        "vae": ("first_stage_model.", None),
        "text_encoder": (
            "cond_stage_model.transformer." if pipeline_format == "sd" else "conditioner.embedders.0.transformer.",
            "clip",
        ),
    }
    if pipeline_format == "sdxl":
        components["text_encoder_2"] = CLIPTextModelWithProjection(text_config)
        specs["text_encoder_2"] = ("conditioner.embedders.1.model.", "openclip")
    for name, model in components.items():
        model.save_pretrained(tmp_path / name)
    output = export_pipeline_checkpoint(
        tmp_path,
        tmp_path / ("original.safetensors" if output_format == "safetensors" else "original.pt"),
        pipeline_format=pipeline_format,
        output_format=output_format,
    )
    for name, model in components.items():
        prefix, fmt = specs[name]
        config = model.config.to_dict() if hasattr(model.config, "to_dict") else dict(model.config)
        if fmt:
            config["original_format"] = fmt
        state = Checkpoint(output, prefix=prefix, wrapper=("state_dict",) if output_format == "pytorch" else ())
        restored = get_conversion(type(model).__name__, config).to_diffusers(state)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(restored[key], value, rtol=0, atol=0)


def test_configuration_helpers_are_importable_and_reusable():
    names = list_config_presets()
    modules = {name.split(".", 1)[0] for name in names if "." in name and not name.startswith("asymmetric-")}
    for name in modules:
        importlib.import_module("diffusers.loaders.conversion.configs." + name)
    tiny = get_config_preset("tiny-vae")
    assert tiny == {"_class_name": "AutoencoderTiny"}
    first = get_config_preset("asymmetric-vae-1.5")
    first["up_block_out_channels"].append(1)
    assert len(get_config_preset("asymmetric-vae-1.5")["up_block_out_channels"]) == 4
    config = get_config_preset("prx.build_config", arguments={"variant": "flux"})
    assert config["in_channels"] == 16
    config = get_config_preset(
        "unidiffuser.create_unidiffuser_unet_config", arguments={"config_type": "test", "version": 1}
    )
    assert config["use_data_type_embedding"] is True
    config = get_config_preset("wan.get_transformer_config", arguments={"model_type": "Wan-T2V-1.3B"})
    assert config["num_attention_heads"] == 12
    config = get_config_preset("prx.create_scheduler_config", arguments={"shift": 3.0})
    assert config["shift"] == 3.0


@pytest.mark.parametrize("source_format", ["torchscript", "python-model"])
def test_module_source_manifest(tmp_path, source_format):
    model = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "model.pt"
    wrapper = []
    if source_format == "torchscript":
        torch.jit.script(model).save(str(checkpoint))
    else:
        torch.save({"ema": model}, checkpoint)
        wrapper = ["ema"]
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {"path": "model.pt", "format": source_format, "wrapper": wrapper, "output_prefix": "decoder."}
                ]
            }
        )
    )
    state = load_source_manifest(manifest)
    assert set(state) == {"decoder.weight", "decoder.bias"}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(state["decoder." + name], value, rtol=0, atol=0)
