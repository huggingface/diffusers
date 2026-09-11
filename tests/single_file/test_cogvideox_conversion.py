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

import json

import pytest
import torch

from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers.loaders.conversion.cogvideox import (
    COGVIDEOX_TRANSFORMER_PREFIX,
    CogVideoXAdaLN,
    cogvideox_transformer_conversion,
    cogvideox_vae_conversion,
    convert_cogvideox_transformer_checkpoint_to_diffusers,
    convert_cogvideox_vae_checkpoint_to_diffusers,
    unwrap_cogvideox_checkpoint,
)
from diffusers.loaders.conversion.io import convert_checkpoint


def export_component(model_path, output_path, component):
    from pathlib import Path

    model_path = Path(model_path)
    if not (model_path / "config.json").is_file():
        model_path = model_path / component
    config = json.loads((model_path / "config.json").read_text())
    expected = "CogVideoXTransformer3DModel" if component == "transformer" else "AutoencoderKLCogVideoX"
    if config["_class_name"] != expected:
        raise ValueError(f"Expected {expected}")
    return convert_checkpoint(
        model_path,
        output_path,
        config=config,
        reverse=True,
        output_format="pytorch",
        output_prefix=COGVIDEOX_TRANSFORMER_PREFIX if component == "transformer" else "",
        output_wrapper=("module" if component == "transformer" else "state_dict",),
    )


def make_transformer(variant="1.0-t2v", **kwargs):
    torch.manual_seed(0)
    return CogVideoXTransformer3DModel(
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=8 if variant.endswith("i2v") else 4,
        out_channels=4,
        time_embed_dim=16,
        text_embed_dim=8,
        sample_height=4,
        sample_width=4,
        sample_frames=5,
        max_text_seq_length=4,
        patch_size_t=2 if variant.startswith("1.5") else None,
        patch_bias=variant.startswith("1.0"),
        use_rotary_positional_embeddings=variant != "1.0-t2v",
        use_learned_positional_embeddings=variant == "1.0-i2v",
        ofs_embed_dim=8 if variant == "1.5-i2v" else None,
        **kwargs,
    )


def make_vae(channels=(8, 16), quant_conv=False):
    torch.manual_seed(0)
    return AutoencoderKLCogVideoX(
        block_out_channels=channels,
        down_block_types=("CogVideoXDownBlock3D",) * len(channels),
        up_block_types=("CogVideoXUpBlock3D",) * len(channels),
        layers_per_block=1,
        norm_num_groups=4,
        latent_channels=4,
        sample_height=16,
        sample_width=16,
        use_quant_conv=quant_conv,
        use_post_quant_conv=quant_conv,
    )


class TestCogVideoXConversion:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("parameter", ["weight", "bias"])
    def test_adaln_known_layout(self, dtype, parameter):
        # SAT groups: image attention, image MLP, text attention, text MLP; three tensors per group.
        original = torch.arange(24, dtype=dtype)
        if parameter == "weight":
            original = original[:, None].expand(-1, 3).clone()
        norm1, norm2 = CogVideoXAdaLN(2).forward((original,))
        expected1 = torch.cat((original[0:6], original[12:18]))
        expected2 = torch.cat((original[6:12], original[18:24]))
        torch.testing.assert_close(norm1, expected1, rtol=0, atol=0)
        torch.testing.assert_close(norm2, expected2, rtol=0, atol=0)
        torch.testing.assert_close(CogVideoXAdaLN(2).inverse((norm1, norm2))[0], original, rtol=0, atol=0)

    @pytest.mark.parametrize("variant", ["1.0-t2v", "1.0-i2v", "1.5-t2v", "1.5-i2v"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_transformer_complete_roundtrip(self, variant, dtype):
        model = make_transformer(variant).to(dtype=dtype)
        conversion = cogvideox_transformer_conversion(model.config)
        diffusers = model.state_dict()
        original = conversion.to_original(diffusers)
        # Verify names and ordering independently of the inverse executor.
        torch.testing.assert_close(
            original["transformer.layers.1.attention.query_key_value.weight"],
            torch.cat([diffusers[f"transformer_blocks.1.attn1.to_{name}.weight"] for name in "qkv"]),
            rtol=0,
            atol=0,
        )
        assert (
            original["transformer.layers.0.post_attention_layernorm.weight"]
            is diffusers["transformer_blocks.0.norm2.norm.weight"]
        )
        assert (
            original["mixins.adaln_layer.query_layernorm_list.0.bias"]
            is diffusers["transformer_blocks.0.attn1.norm_q.bias"]
        )
        restored = conversion.to_diffusers(original)
        for key in diffusers:
            torch.testing.assert_close(restored[key], diffusers[key], rtol=0, atol=0)
        for key, tensor in conversion.to_original(restored).items():
            torch.testing.assert_close(tensor, original[key], rtol=0, atol=0)
        if variant == "1.0-i2v":
            assert "mixins.pos_embed.pos_embedding" in original
        if variant.startswith("1.5"):
            assert "mixins.patch_embed.proj.bias" in original

    def test_transformer_without_optional_biases_and_affine_norms(self):
        model = make_transformer(attention_bias=False, norm_elementwise_affine=False)
        conversion = cogvideox_transformer_conversion(model.config)
        original = conversion.to_original(model.state_dict())
        assert "transformer.layers.0.attention.query_key_value.bias" not in original
        assert "transformer.layers.0.input_layernorm.weight" not in original
        restored = conversion.to_diffusers(original)
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(restored[key], tensor, rtol=0, atol=0)

    @pytest.mark.parametrize("channels", [(8, 8), (8, 16, 16, 32)])
    @pytest.mark.parametrize("quant_conv", [False, True])
    def test_vae_complete_roundtrip_and_up_block_numbering(self, channels, quant_conv):
        model = make_vae(channels, quant_conv)
        conversion = cogvideox_vae_conversion(model.config)
        diffusers = model.state_dict()
        original = conversion.to_original(diffusers)
        for i in range(len(channels)):
            assert (
                original[f"decoder.up.{len(channels) - 1 - i}.block.0.conv1.conv.weight"]
                is diffusers[f"decoder.up_blocks.{i}.resnets.0.conv1.conv.weight"]
            )
        assert (
            original["encoder.mid.block_2.conv1.conv.weight"]
            is diffusers["encoder.mid_block.resnets.1.conv1.conv.weight"]
        )
        restored = conversion.to_diffusers(original)
        for key in diffusers:
            torch.testing.assert_close(restored[key], diffusers[key], rtol=0, atol=0)

    def test_nested_checkpoint_and_component_selection_are_non_mutating(self):
        model = make_transformer()
        original = cogvideox_transformer_conversion(model.config).to_original(model.state_dict())
        prefixed = {COGVIDEOX_TRANSFORMER_PREFIX + key: tensor for key, tensor in original.items()}
        prefixed[COGVIDEOX_TRANSFORMER_PREFIX + "mixins.pos_embed.freqs_sin"] = torch.zeros(1)
        prefixed["conditioner.sibling.weight"] = torch.ones(1)
        wrapped = {"model": {"module": {"state_dict": prefixed}}, "iteration": 10}
        original_keys = set(prefixed)
        converted = convert_cogvideox_transformer_checkpoint_to_diffusers(wrapped, model.config)
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(converted[key], tensor, rtol=0, atol=0)
        assert set(prefixed) == original_keys
        assert wrapped["iteration"] == 10
        assert prefixed["conditioner.sibling.weight"].item() == 1

    def test_unknown_auxiliary_key_is_not_silently_ignored(self):
        model = make_transformer()
        original = cogvideox_transformer_conversion(model.config).to_original(model.state_dict())
        original["mixins.pos_embed.freqs_sin_typo"] = torch.zeros(1)
        with pytest.raises(ValueError, match="unexpected keys.*freqs_sin_typo"):
            convert_cogvideox_transformer_checkpoint_to_diffusers(original, model.config)

    def test_mixed_prefixes_are_rejected(self):
        model = make_transformer()
        original = cogvideox_transformer_conversion(model.config).to_original(model.state_dict())
        original[COGVIDEOX_TRANSFORMER_PREFIX + "time_embed.0.weight"] = original["time_embed.0.weight"]
        with pytest.raises(ValueError, match="mixes prefixed and unprefixed"):
            convert_cogvideox_transformer_checkpoint_to_diffusers(original, model.config)

    def test_ambiguous_wrappers_are_rejected(self):
        with pytest.raises(ValueError, match="Ambiguous"):
            unwrap_cogvideox_checkpoint({"model": {}, "module": {}})

    def test_vae_loss_is_excluded_without_mutation(self):
        model = make_vae()
        original = cogvideox_vae_conversion(model.config).to_original(model.state_dict())
        original["loss.discriminator.weight"] = torch.ones(1)
        converted = convert_cogvideox_vae_checkpoint_to_diffusers({"state_dict": original}, model.config)
        assert "loss.discriminator.weight" in original
        assert set(converted) == set(model.state_dict())

    @pytest.mark.parametrize("model_class", [CogVideoXTransformer3DModel, AutoencoderKLCogVideoX])
    def test_single_file_requires_explicit_config(self, model_class):
        with pytest.raises(ValueError, match="requires an explicit Diffusers `config`"):
            model_class.from_single_file({})

    @pytest.mark.parametrize("component", ["transformer", "vae"])
    @pytest.mark.parametrize("safe_serialization", [False, True])
    @pytest.mark.parametrize("sharded", [False, True])
    def test_export_and_single_file_loading(self, tmp_path, component, safe_serialization, sharded):
        model = make_transformer("1.5-i2v") if component == "transformer" else make_vae()
        model = model.to(dtype=torch.bfloat16)
        component_path = tmp_path / "pipeline" / component
        model.save_pretrained(
            component_path,
            safe_serialization=safe_serialization,
            max_shard_size="20KB" if sharded else "5GB",
        )
        output_path = tmp_path / "original.pt"
        export_component(str(component_path.parent), str(output_path), component)
        checkpoint = torch.load(output_path, weights_only=True)
        original = unwrap_cogvideox_checkpoint(checkpoint)
        assert all(tensor.dtype == torch.bfloat16 for tensor in original.values())
        loaded = type(model).from_single_file(
            str(output_path), config=str(component_path), torch_dtype=torch.bfloat16, local_files_only=True
        )
        restored_path = tmp_path / "shared-converter"
        convert_checkpoint(
            output_path,
            restored_path,
            config=dict(model.config),
            input_wrapper=("module" if component == "transformer" else "state_dict",),
            input_prefix=COGVIDEOX_TRANSFORMER_PREFIX if component == "transformer" else "",
        )
        restored_model = type(model).from_pretrained(restored_path, torch_dtype=torch.bfloat16, local_files_only=True)
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], tensor, rtol=0, atol=0)
            torch.testing.assert_close(restored_model.state_dict()[key], tensor, rtol=0, atol=0)

        # Conversion can create disjoint tensor views sharing storage. They must survive safetensors serialization.
        roundtrip_path = tmp_path / "roundtrip"
        loaded.save_pretrained(roundtrip_path)
        reloaded = type(model).from_pretrained(roundtrip_path, torch_dtype=torch.bfloat16, local_files_only=True)
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(reloaded.state_dict()[key], tensor, rtol=0, atol=0)

    @pytest.mark.parametrize("low_cpu_mem_usage", [False, True])
    def test_transformer_forward_after_single_file_loading(self, tmp_path, low_cpu_mem_usage):
        model = make_transformer().eval()
        model.save_config(tmp_path)
        original = cogvideox_transformer_conversion(model.config).to_original(model.state_dict())
        loaded = (
            type(model)
            .from_single_file(
                original, config=str(tmp_path), low_cpu_mem_usage=low_cpu_mem_usage, local_files_only=True
            )
            .eval()
        )
        generator = torch.Generator().manual_seed(42)
        inputs = {
            "hidden_states": torch.randn(1, 2, 4, 4, 4, generator=generator),
            "encoder_hidden_states": torch.randn(1, 4, 8, generator=generator),
            "timestep": torch.tensor([1]),
        }
        with torch.no_grad():
            expected = model(**inputs).sample
            actual = loaded(**inputs).sample
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_older_transformer_config_and_fixed_position_embedding(self, tmp_path, dtype):
        model = make_transformer().to(dtype=dtype)
        model.save_pretrained(tmp_path)
        config_path = tmp_path / "config.json"
        config = json.loads(config_path.read_text())
        # These options were introduced after the original 1.0 checkpoints.
        for key in ("ofs_embed_dim", "patch_size_t", "patch_bias", "use_learned_positional_embeddings"):
            config.pop(key)
        config_path.write_text(json.dumps(config))
        output_path = tmp_path / "original.pt"
        export_component(str(tmp_path), str(output_path), "transformer")
        checkpoint = torch.load(output_path, weights_only=True)
        original_positions = checkpoint["module"][COGVIDEOX_TRANSFORMER_PREFIX + "mixins.pos_embed.pos_embedding"]
        torch.testing.assert_close(original_positions, model.patch_embed.pos_embedding, rtol=0, atol=0)
        loaded = type(model).from_single_file(
            str(output_path), config=str(tmp_path), local_files_only=True, torch_dtype=dtype
        )
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], tensor, rtol=0, atol=0)

    def test_quantized_export_is_rejected(self, tmp_path):
        model = make_transformer()
        model.save_config(tmp_path)
        config_path = tmp_path / "config.json"
        config = json.loads(config_path.read_text())
        config["quantization_config"] = {"quant_method": "bitsandbytes"}
        config_path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="requires unpacked, unquantized tensor weights"):
            export_component(str(tmp_path), str(tmp_path / "original.pt"), "transformer")
