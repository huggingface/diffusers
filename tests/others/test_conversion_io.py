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

import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from diffusers import FluxTransformer2DModel
from diffusers.loaders.conversion import get_conversion
from diffusers.loaders.conversion.checkpoint import convert_component_checkpoint
from diffusers.loaders.conversion.io import Checkpoint, convert_checkpoint


def checkpoint_digest(state):
    """Hash names, shapes, dtypes and raw tensor bytes, independently of checkpoint serialization."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(json.dumps([key, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def test_svd_vae_selects_qualified_namespaces_from_bundle(tmp_path):
    from tests.models.autoencoders.test_models_autoencoder_kl_temporal_decoder import (
        AutoencoderKLTemporalDecoderTesterConfig,
    )

    tester = AutoencoderKLTemporalDecoderTesterConfig()
    model = tester.model_class(**tester.get_init_dict())
    config = dict(model.config)
    original = get_conversion(type(model).__name__, config).to_original(model.state_dict())
    bundle = {
        **original,
        "model.diffusion_model.input_blocks.0.0.weight": torch.zeros(1),
        "conditioner.embedders.0.model.visual.conv1.weight": torch.zeros(1),
    }
    loaded = type(model).from_single_file(bundle, config=config, local_files_only=True)
    assert checkpoint_digest(loaded.state_dict()) == checkpoint_digest(model.state_dict())
    source = tmp_path / "svd.safetensors"
    save_file(bundle, source)
    output = convert_checkpoint(source, tmp_path / "vae", config=config, model_class=type(model).__name__)
    assert checkpoint_digest(Checkpoint(output)) == checkpoint_digest(model.state_dict())
    bundle["first_stage_model.decoder.typo.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected keys.*typo"):
        type(model).from_single_file(bundle, config=config, local_files_only=True)


@pytest.mark.parametrize("component", ["unet_text", "unet_image"])
def test_versatile_components_keep_qualified_namespaces(component):
    from diffusers import UNet2DConditionModel
    from diffusers.pipelines.deprecated.versatile_diffusion.modeling_text_unet import UNetFlatConditionModel
    from tests.single_file.test_auxiliary_conversions import CASES

    cls = UNetFlatConditionModel if component == "unet_text" else UNet2DConditionModel
    if component == "unet_image":
        from tests.models.unets.test_models_unet_2d_condition import UNet2DConditionTesterConfig

        config = UNet2DConditionTesterConfig().get_init_dict()
        config["original_format"] = "versatile_image"
    else:
        config = dict(next(config for name, config in CASES if name == cls.__name__))
    model = cls(**{key: value for key, value in config.items() if key != "original_format"})
    resolved_config = {**dict(model.config), **config}
    original = get_conversion(cls.__name__, resolved_config).to_original(model.state_dict())
    other = "unet_image" if component == "unet_text" else "unet_text"
    bundle = {**original, f"model.diffusion_model.{other}.unrelated.weight": torch.ones(1)}
    loaded = cls.from_single_file(bundle, config=resolved_config)
    assert checkpoint_digest(loaded.state_dict()) == checkpoint_digest(model.state_dict())
    bundle[f"model.diffusion_model.{component}.typo.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="unexpected keys.*typo"):
        cls.from_single_file(bundle, config=resolved_config)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hunyuan_image_vae_original_temporal_axis(dtype):
    from diffusers import AutoencoderKLHunyuanImage

    model = AutoencoderKLHunyuanImage(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(32, 64),
        layers_per_block=1,
        spatial_compression_ratio=2,
        sample_size=16,
    ).to(dtype)
    config = dict(model.config)
    spatial = get_conversion(type(model).__name__, {**config, "original_format": "hunyuan_image_vae_2d"})
    # Construct the original 5D source independently of the temporal-axis transform's inverse.
    original = {
        key: value.unsqueeze(2) if key.endswith(".weight") and value.ndim == 4 else value
        for key, value in spatial.to_original(model.state_dict()).items()
    }
    conversion = get_conversion(type(model).__name__, config)
    restored = conversion.to_diffusers(original)
    assert checkpoint_digest(restored) == checkpoint_digest(model.state_dict())
    assert checkpoint_digest(conversion.to_original(restored)) == checkpoint_digest(original)
    assert conversion.to_original(restored)["encoder.conv_in.weight"].shape == (32, 3, 1, 3, 3)
    loaded = type(model).from_single_file(
        {"vae." + k: v for k, v in original.items()}, config=config, torch_dtype=dtype
    )
    assert checkpoint_digest(loaded.state_dict()) == checkpoint_digest(model.state_dict())
    invalid = {**original, "encoder.conv_in.weight": original["encoder.conv_in.weight"].expand(-1, -1, 2, -1, -1)}
    with pytest.raises(ValueError, match="singleton axis"):
        conversion.to_diffusers(invalid)


@pytest.mark.parametrize("family", ["cosmos", "ltx2"])
def test_inferred_original_format_survives_save_and_export(tmp_path, family):
    if family == "cosmos":
        from tests.models.transformers.test_models_transformer_cosmos import CosmosTransformerTesterConfig

        tester = CosmosTransformerTesterConfig()
        model = tester.model_class(**tester.get_init_dict())
        original_format = "cosmos1"
    else:
        from diffusers import LTX2VideoDiffusionDecoderModel

        model = LTX2VideoDiffusionDecoderModel(
            latent_channels=8,
            decoder_head_dim=8,
            decoder_stage_channels=[32, 16, 8, 8, 8],
            decoder_stage_depths=[1, 1, 1, 1, 1],
            decoder_upsample_channel_reductions=[2, 2, 1, 1],
            decoder_t_emb_dim=16,
        )
        original_format = "ltx2_diffusion_decoder_gated"
    config = dict(model.config)
    conversion = get_conversion(type(model).__name__, {**config, "original_format": original_format})
    original = conversion.to_original(model.state_dict())
    source = tmp_path / "original.safetensors"
    save_file(original, source)
    output = convert_checkpoint(source, tmp_path / "model", config=config, model_class=type(model).__name__)
    saved_config = json.loads((output / "config.json").read_text())
    assert saved_config["original_format"] == original_format
    exported = convert_checkpoint(output, tmp_path / "exported", config=saved_config, reverse=True)
    assert checkpoint_digest(Checkpoint(exported)) == checkpoint_digest(original)
    loaded = type(model).from_single_file(original, config=config)
    loaded.save_config(tmp_path / "loader")
    assert json.loads((tmp_path / "loader/config.json").read_text())["original_format"] == original_format
    assert "original_format" not in config


def test_pytorch_auxiliary_metadata_matches_mapping_import(tmp_path):
    from tests.models.transformers.test_models_transformer_cosmos import CosmosTransformerTesterConfig

    tester = CosmosTransformerTesterConfig()
    model = tester.model_class(**tester.get_init_dict())
    config = dict(model.config)
    original = get_conversion(type(model).__name__, config).to_original(model.state_dict())
    source = {**original, "_extra_state": {"train_info": 1}}
    path = tmp_path / "original.pt"
    torch.save({"state_dict": source}, path)
    output = convert_checkpoint(
        path, tmp_path / "model", config=config, model_class=type(model).__name__, input_wrapper=["state_dict"]
    )
    assert checkpoint_digest(Checkpoint(output)) == checkpoint_digest(
        convert_component_checkpoint(source, config, type(model).__name__)
    )
    source["unknown_metadata"] = {"train_info": 1}
    torch.save({"state_dict": source}, path)
    with pytest.raises(ValueError, match="unexpected keys.*unknown_metadata"):
        convert_checkpoint(
            path, tmp_path / "invalid", config=config, model_class=type(model).__name__, input_wrapper=["state_dict"]
        )
    del source["unknown_metadata"]
    source["x_embedder.proj.1.weight"] = {"not": "a tensor"}
    torch.save({"state_dict": source}, path)
    with pytest.raises(ValueError, match="must be a tensor"):
        convert_checkpoint(
            path, tmp_path / "invalid", config=config, model_class=type(model).__name__, input_wrapper=["state_dict"]
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sharded_file_round_trip(tmp_path, dtype):
    model = FluxTransformer2DModel(
        num_layers=1,
        num_single_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        joint_attention_dim=8,
        pooled_projection_dim=8,
        axes_dims_rope=(2, 2, 4),
    ).to(dtype=dtype)
    source = tmp_path / "source"
    model.save_pretrained(source, max_shard_size="10KB")
    original = convert_checkpoint(
        source,
        tmp_path / "original",
        config=dict(model.config),
        reverse=True,
        output_prefix="model.diffusion_model.",
        max_shard_size=3000,
    )
    restored = convert_checkpoint(
        original,
        tmp_path / "restored",
        config=dict(model.config),
        input_prefix="model.diffusion_model.",
        max_shard_size=2000,
    )
    loaded = FluxTransformer2DModel.from_pretrained(restored, torch_dtype=dtype, local_files_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    assert json.loads((original / "conversion.json").read_text())["lossless"]
    assert (original / "conversion_config.json").is_file()


def test_explicit_pytorch_wrapper_and_component(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"training_step": 10, "state_dict": {"component.x": torch.arange(3), "other.y": torch.ones(1)}}, path)
    with pytest.raises(ValueError, match="tensor state dict"):
        Checkpoint(path)
    state = Checkpoint(path, wrapper=("state_dict",), prefix="component.")
    assert set(state) == {"x"}
    torch.testing.assert_close(state["x"], torch.arange(3))


def test_tied_text_embeddings_survive_file_conversion(tmp_path):
    from transformers import T5Config, T5EncoderModel

    model = T5EncoderModel(T5Config(vocab_size=16, d_model=8, d_ff=16, d_kv=4, num_heads=2, num_layers=1))
    source = tmp_path / "source"
    model.save_pretrained(source)
    config = model.config.to_dict()
    original = convert_checkpoint(
        source, tmp_path / "original", config=config, model_class="T5EncoderModel", reverse=True
    )
    restored = convert_checkpoint(original, tmp_path / "restored", config=config, model_class="T5EncoderModel")
    loaded = T5EncoderModel.from_pretrained(restored, local_files_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["outside", "wrong_shard", "missing_key", "extra_key", "duplicate"])
def test_invalid_shard_index_is_rejected(tmp_path, failure):
    save_file({"a": torch.ones(2)}, tmp_path / "one.safetensors")
    save_file({"b": torch.zeros(2)}, tmp_path / "two.safetensors")
    weight_map = {"a": "one.safetensors", "b": "two.safetensors"}
    if failure == "outside":
        weight_map["b"] = "../outside.safetensors"
    elif failure == "wrong_shard":
        weight_map = {"a": "two.safetensors", "b": "one.safetensors"}
    elif failure == "missing_key":
        weight_map["c"] = "one.safetensors"
    elif failure == "extra_key":
        save_file({"a": torch.ones(2), "c": torch.ones(2)}, tmp_path / "one.safetensors")
    else:
        save_file({"a": torch.ones(2), "b": torch.ones(2)}, tmp_path / "two.safetensors")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(ValueError, match="escapes|does not match|Duplicate"):
        Checkpoint(index)


def test_failed_conversion_does_not_publish_partial_output(tmp_path):
    source = tmp_path / "input.safetensors"
    save_file({"projection.weight": torch.ones(2)}, source)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="missing keys"):
        convert_checkpoint(source, output, model_class="CCProjection", config={}, reverse=True)
    assert not output.exists()
    assert not list(tmp_path.glob(".conversion-*"))
    output.mkdir()
    with pytest.raises(FileExistsError):
        convert_checkpoint(source, output, model_class="CCProjection", config={}, reverse=True)


def test_pytorch_output_wrappers_round_trip(tmp_path):
    source = tmp_path / "weights.safetensors"
    state = {"projection.weight": torch.randn(8, 16), "projection.bias": torch.randn(8)}
    save_file(state, source)
    output = convert_checkpoint(
        source,
        tmp_path / "original.pt",
        config={},
        model_class="CCProjection",
        reverse=True,
        output_format="pytorch",
        output_wrapper=("model", "state_dict"),
        output_prefix="component.",
    )
    restored = convert_checkpoint(
        output,
        tmp_path / "restored",
        config={},
        model_class="CCProjection",
        input_wrapper=("model", "state_dict"),
        input_prefix="component.",
    )
    for key, value in Checkpoint(restored).items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_single_file_component_selection_is_strict_and_nonmutating():
    model = FluxTransformer2DModel(
        num_layers=1,
        num_single_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        joint_attention_dim=8,
        pooled_projection_dim=8,
        axes_dims_rope=(2, 2, 4),
    )
    conversion = get_conversion(type(model).__name__, dict(model.config))
    original = conversion.to_original(model.state_dict())
    torch.testing.assert_close(
        original["double_blocks.0.img_attn.qkv.weight"],
        torch.cat([model.state_dict()[f"transformer_blocks.0.attn.to_{part}.weight"] for part in ("q", "k", "v")]),
        rtol=0,
        atol=0,
    )
    checkpoint = {"model.diffusion_model." + key: value for key, value in original.items()}
    checkpoint["first_stage_model.some_weight"] = torch.ones(1)
    keys = set(checkpoint)
    loaded = convert_component_checkpoint(checkpoint, dict(model.config), type(model).__name__)
    assert set(checkpoint) == keys
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded[key], value, rtol=0, atol=0)
    checkpoint["model.diffusion_model.typo"] = torch.ones(1)
    with pytest.raises(ValueError, match="unexpected keys.*typo"):
        convert_component_checkpoint(checkpoint, dict(model.config), type(model).__name__)


def test_asymmetric_loader_excludes_training_aliases():
    from diffusers import AsymmetricAutoencoderKL

    model = AsymmetricAutoencoderKL(
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        down_block_out_channels=(32,),
        up_block_out_channels=(32,),
        layers_per_down_block=1,
        layers_per_up_block=1,
    )
    config = dict(model.config)
    original = get_conversion("AsymmetricAutoencoderKL", config).to_original(model.state_dict())
    original["decoder.up_layers.0.weight"] = torch.ones(1)
    original["loss.discriminator.weight"] = torch.ones(1)
    for restored in (
        AsymmetricAutoencoderKL.from_single_file(original, config=config, local_files_only=True).state_dict(),
        convert_component_checkpoint(original, config, "AsymmetricAutoencoderKL"),
    ):
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(restored[key], tensor, rtol=0, atol=0)


def test_single_file_accepts_config_mapping_without_mutating_it():
    model = FluxTransformer2DModel(
        num_layers=1,
        num_single_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        joint_attention_dim=8,
        pooled_projection_dim=8,
        axes_dims_rope=(2, 2, 4),
    )
    config = dict(model.config)
    before = dict(config)
    original = get_conversion(type(model).__name__, config).to_original(model.state_dict())
    loaded = type(model).from_single_file(original, config=config, local_files_only=True)
    assert config == before
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)


def test_ltx2_decoder_partial_gates_and_auxiliary_state(tmp_path):
    from diffusers import LTX2VideoDiffusionDecoderModel

    model = LTX2VideoDiffusionDecoderModel(
        latent_channels=8,
        decoder_head_dim=8,
        decoder_stage_channels=[32, 16, 8, 8, 8],
        decoder_stage_depths=[1, 1, 1, 1, 1],
        decoder_upsample_channel_reductions=[2, 2, 1, 1],
        decoder_t_emb_dim=16,
    )
    config = dict(model.config)
    conversion = get_conversion(type(model).__name__, {**config, "original_format": "ltx2_diffusion_decoder_gated"})
    source = conversion.to_original(model.state_dict())
    gates = sorted(key for key in source if key.endswith((".gate_msa", ".gate_mlp", ".gate_ctx")))
    for key in gates[1:]:
        del source[key]
    source[gates[0]] = torch.full_like(source[gates[0]], 0.5)
    source["encoder.unused.weight"] = torch.ones(1)
    source["decoder.coarse_head.weight"] = torch.ones(1)
    source["decoder.diff_blocks.0.coarse_proj.weight"] = torch.ones(1)
    expected = dict(model.state_dict())
    rule = next(rule for rule in conversion.rules if rule.original[0] == gates[0])
    for key in rule.diffusers:
        expected[key] = (expected[key].float() * 0.5).to(expected[key].dtype)
    original_keys = set(source)
    path = tmp_path / "original.safetensors"
    save_file(source, path)
    output = convert_checkpoint(path, tmp_path / "converted", config=config, model_class=type(model).__name__)
    for actual in (convert_component_checkpoint(source, config, type(model).__name__), Checkpoint(output)):
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
    assert set(source) == original_keys
