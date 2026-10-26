# Copyright 2026 The VAST-AI Team. All rights reserved.
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

"""Convert a local original EdenLive release into standard Diffusers components."""

import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    CLIPVisionConfig,
    CLIPVisionModel,
    DINOv3ViTConfig,
    DINOv3ViTModel,
    UMT5EncoderModel,
)

from diffusers import (
    AutoencoderKLWan,
    EdenLiveLightTAEDecoder,
    EdenLiveTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
)
from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
from diffusers.models.model_loading_utils import load_state_dict


def read_config(path):
    return json.loads(path.read_text())


def tensor_record(value):
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(raw).hexdigest()}


def save_component(model, state, destination, manifest):
    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    model.save_pretrained(destination, safe_serialization=True, max_shard_size="5GB")
    dtype = next(iter(state.values())).dtype
    restored = type(model).from_pretrained(destination, torch_dtype=dtype, local_files_only=True)
    actual = restored.state_dict()
    if actual.keys() != state.keys():
        raise ValueError(f"Round-trip keys differ for {destination.name}")
    records = {}
    for name, original in state.items():
        value = actual[name]
        if original.dtype != value.dtype or not torch.equal(original, value):
            raise ValueError(f"Round-trip changed {destination.name}/{name}")
        records[name] = tensor_record(original)
    manifest[destination.name] = records
    del restored
    gc.collect()
    print(f"Verified {destination}: {len(records)} tensors", flush=True)


def convert_lighttae(state, model):
    layer_map = {1: "conv_in", 22: "conv_out"}
    for stage in range(3):
        for block in range(3):
            layer_map[3 + 6 * stage + block] = f"stages.{stage}.memory_blocks.{block}"
        layer_map[7 + 6 * stage] = f"stages.{stage}.temporal_upsample"
        layer_map[8 + 6 * stage] = f"stages.{stage}.conv_out"
    converted = {}
    for key, value in state.items():
        if key.startswith("encoder."):
            continue
        key = key.replace("decoder.blocks.", "decoder.", 1)
        _, layer, suffix = key.split(".", 2)
        converted[f"{layer_map[int(layer)]}.{suffix}"] = value
    for key, expected in model.state_dict().items():
        if key.endswith(".conv.weight") and key in converted and converted[key].shape[0] > expected.shape[0]:
            converted[key] = converted[key][-expected.shape[0] :]
    return converted


def _proxy_state(state):
    normalized = {}
    for key, value in state.items():
        for prefix in ("model.module.", "model.", "module."):
            if key.startswith(prefix):
                key = key.removeprefix(prefix)
                break
        key = key.removeprefix("encoder.")
        if key in normalized:
            raise ValueError(f"Duplicate proxy checkpoint key: {key}.")
        if not key.startswith(("backbone.", "patch_projection.")) and key not in (
            "global_projection.weight",
            "global_projection.bias",
        ):
            raise ValueError(f"Unrecognized proxy checkpoint key: {key}.")
        normalized[key] = value
    return normalized


def _remap_wan22_vae_key(key):
    """Normalize only the nested Wan 2.2 blocks missing from the native mapper."""
    parts = key.split(".")
    if len(parts) < 5 or parts[:2] not in (["encoder", "downsamples"], ["decoder", "upsamples"]):
        return key
    block = "down_blocks" if parts[0] == "encoder" else "up_blocks"
    prefix = f"{parts[0]}.{block}.{parts[2]}"
    if parts[3] == "avg_shortcut":
        return f"{prefix}.avg_shortcut.{'.'.join(parts[4:])}"
    if len(parts) < 6 or parts[3] != parts[1]:
        return key
    slot, rest = parts[4], parts[5:]
    if rest[0] == "residual" and len(rest) > 2:
        field = {"0": "norm1", "2": "conv1", "3": "norm2", "6": "conv2"}.get(rest[1])
        if field is not None:
            return f"{prefix}.resnets.{slot}.{field}.{'.'.join(rest[2:])}"
    elif rest[0] == "shortcut":
        return f"{prefix}.resnets.{slot}.conv_shortcut.{'.'.join(rest[1:])}"
    elif parts[0] == "encoder" and slot == "2":
        return f"{prefix}.downsampler.{'.'.join(rest)}"
    elif parts[0] == "decoder" and slot == "3":
        return f"{prefix}.upsampler.{'.'.join(rest)}"
    return key


def convert(source, output, size, variant):
    if output.exists() and any(output.iterdir()):
        raise ValueError("The output directory must be empty.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    proxy_state = _proxy_state(load_state_dict(str(source / "vision-encoder/projection.safetensors")))
    projection = {
        key.replace("patch_projection.", "proxy_projection.", 1): value
        for key, value in proxy_state.items()
        if key.startswith("patch_projection.")
    }
    del proxy_state
    folder = f"dit-{size}-bidir" if variant == "bidirectional" else f"dit-{size}"
    config = read_config(source / folder / "config.json")
    config.pop("cross_attn_norm", None)
    state = load_state_dict(str(source / folder / "diffusion_pytorch_model.safetensors"))
    config["proxy_input_channels"] = projection["proxy_projection.weight"].shape[1]
    config["proxy_channels"] = state["patch_embedding.weight"].shape[1] - config["in_channels"]
    config["condition_rmsnorm"] = "patch_embedding.condition_norm.weight" in state
    state.update(projection)
    with torch.device("meta"):
        model = EdenLiveTransformer3DModel.from_config(config)
    for attribute in ("rope", "cache_rope"):
        rope = getattr(model, attribute)
        setattr(model, attribute, type(rope)(rope.attention_head_dim, tuple(rope.patch_size), rope.max_seq_len))
    save_component(model, state, output / "transformer", manifest)
    del model, state, projection
    gc.collect()

    config = read_config(source / f"vae-{size}" / "config.json")
    version = "2" if size == "5b" else "1"
    state = load_state_dict(str(source / f"vae-{size}" / f"Wan2.{version}_VAE.pth"))
    state = convert_wan_vae_to_diffusers({_remap_wan22_vae_key(key): value for key, value in state.items()})
    config["in_channels"] = state["encoder.conv_in.weight"].shape[1]
    config["out_channels"] = state["decoder.conv_out.weight"].shape[0]
    with torch.device("meta"):
        model = AutoencoderKLWan.from_config(config)
    save_component(model, state, output / "vae", manifest)
    del model, state
    gc.collect()

    state = load_state_dict(str(source / f"decoder-{size}" / f"lighttaew2_{version}.pth"))
    with torch.device("meta"):
        model = EdenLiveLightTAEDecoder(
            latent_channels=48 if size == "5b" else 16, patch_size=2 if size == "5b" else 1
        )
    save_component(model, convert_lighttae(state, model), output / "decoder", manifest)
    del model, state
    gc.collect()

    state = _proxy_state(load_state_dict(str(source / "vision-encoder/projection.safetensors")))
    config = DINOv3ViTConfig.from_dict(read_config(source / "vision-encoder/backbone/config.json"))
    with torch.device("meta"):
        model = DINOv3ViTModel(config)
    model.rope_embeddings = type(model.rope_embeddings)(config)
    expected = set(model.state_dict())
    backbone = {}
    for key, value in state.items():
        if key.startswith("backbone."):
            key = key.removeprefix("backbone.")
            if key not in expected and f"model.{key}" in expected:
                key = f"model.{key}"
            backbone[key] = value
    save_component(model, backbone, output / "proxy_encoder", manifest)
    del model, state, backbone
    gc.collect()

    for name in ("text_encoder", "tokenizer"):
        shutil.copytree(source / "umt5" / name, output / name, dirs_exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(output / "tokenizer", local_files_only=True)
    original = load_state_dict(str(source / "umt5/text_encoder/model.safetensors"))
    model = UMT5EncoderModel.from_pretrained(
        output / "text_encoder", dtype=next(iter(original.values())).dtype, local_files_only=True
    )
    restored = model.state_dict()
    for key, value in original.items():
        if restored[key].dtype != value.dtype or not torch.equal(restored[key], value):
            raise ValueError(f"Text encoder round-trip changed {key}")
    manifest["text_encoder"] = {key: tensor_record(value) for key, value in original.items()}
    del original, restored, model
    gc.collect()

    if size == "14b":
        config = CLIPVisionConfig.from_dict(read_config(source / "clip/image_encoder/config.json"))
        state = load_state_dict(str(source / "clip/image_encoder/model.safetensors"))
        state.pop("visual_projection.weight")
        with torch.device("meta"):
            model = CLIPVisionModel(config)
        expected_keys = set(model.state_dict())
        state = {
            key if key in expected_keys else key.removeprefix("vision_model."): value for key, value in state.items()
        }
        save_component(model, state, output / "image_encoder", manifest)
        shutil.copytree(source / "clip/image_processor", output / "image_processor", dirs_exist_ok=True)
        del model, state
        gc.collect()

    if size == "5b":
        scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5.0)
    else:
        scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    if variant == "causal":
        scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0 if size == "5b" else 3.0, stochastic_sampling=True)
    scheduler.save_pretrained(output / "scheduler")
    index = {
        "_class_name": "EdenLiveCausalPipeline" if variant == "causal" else "EdenLivePipeline",
        "transformer": ["diffusers", "EdenLiveTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
        "decoder": [None, None],
        "text_encoder": ["transformers", "UMT5EncoderModel"],
        "tokenizer": ["transformers", type(tokenizer).__name__],
        "proxy_encoder": ["transformers", "DINOv3ViTModel"],
        "image_encoder": ["transformers", "CLIPVisionModel"] if size == "14b" else [None, None],
        "image_processor": ["transformers", "CLIPImageProcessor"] if size == "14b" else [None, None],
        "scheduler": ["diffusers", type(scheduler).__name__],
        "proxy_image_height": 480,
        "proxy_image_width": 832,
    }
    if variant == "causal":
        index.update(chunk_size=3, max_context_frames=18, sink_frames=6)
    (output / "model_index.json").write_text(json.dumps(index, indent=2) + "\n")
    for name in ("LICENSE", "NOTICE"):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    for component, destinations in (
        ("vision-encoder", ("proxy_encoder", "transformer")),
        ("clip", ("image_encoder", "image_processor") if size == "14b" else ()),
    ):
        for destination in destinations:
            for name in ("LICENSE.md", "NOTICE.md"):
                shutil.copy2(source / component / name, output / destination / name)
    (output / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local original EdenLive release directory.")
    parser.add_argument(
        "--output", type=Path, required=True, help="Destination for one standard Diffusers model root."
    )
    parser.add_argument("--size", choices=("5b", "14b"), required=True)
    parser.add_argument("--variant", choices=("bidirectional", "causal"), required=True)
    args = parser.parse_args()
    convert(args.source, args.output, args.size, args.variant)
