#!/usr/bin/env python3
"""Convert the Echo BF16 release checkpoint into a Diffusers Modular Pipeline repository."""

import argparse
import gc
import json
from pathlib import Path

import torch
from convert_ltx2_to_diffusers import (
    convert_ltx2_audio_vae,
    convert_ltx2_connectors,
    convert_ltx2_transformer,
    convert_ltx2_video_vae,
    convert_ltx2_vocoder,
    get_model_state_dict_from_combined_ckpt,
)
from safetensors.torch import load_file

from diffusers import __version__


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path

    manifest_path = path / "checkpoint.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    precision = str(manifest.get("precision", "")).lower()
    if precision not in {"bf16", "bfloat16"}:
        raise ValueError(
            "Echo conversion supports the BF16 release only. The FP8 and FP4 releases use custom packed "
            f"kernels and cannot be represented by this converter, got precision={precision!r}."
        )
    model_filename = manifest.get("files", {}).get("model")
    if not model_filename:
        raise ValueError(f"Checkpoint manifest has no `files.model`: {manifest_path}")
    return path / model_filename


def save_component(model, output_path: Path, name: str, max_shard_size: str) -> None:
    model.to(torch.bfloat16).save_pretrained(
        output_path / name,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )


def component_entry(library: str, class_name: str, repo: str, subfolder: str | None):
    return [
        None,
        None,
        {
            "type_hint": [library, class_name],
            "pretrained_model_name_or_path": repo,
            "subfolder": subfolder,
            "variant": None,
            "revision": None,
        },
    ]


def write_modular_index(output_path: Path, base_model: str, repo_id: str | None) -> None:
    echo_repo = repo_id or str(output_path)
    index = {
        "_class_name": "EchoModularPipeline",
        "_diffusers_version": __version__,
        "_blocks_class_name": "EchoBlocks",
        "text_encoder": component_entry("transformers", "Gemma3ForConditionalGeneration", base_model, None),
        "tokenizer": component_entry("transformers", "GemmaTokenizerFast", base_model, None),
        "connectors": component_entry("ltx2", "LTX2TextConnectors", echo_repo, "connectors"),
        "vae": component_entry("diffusers", "AutoencoderKLLTX2Video", echo_repo, "vae"),
        "audio_vae": component_entry("diffusers", "AutoencoderKLLTX2Audio", echo_repo, "audio_vae"),
        "transformer": component_entry("diffusers", "LTX2VideoTransformer3DModel", echo_repo, "transformer"),
        "vocoder": component_entry("ltx2", "LTX2VocoderWithBWE", echo_repo, "vocoder"),
    }
    (output_path / "modular_model_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def convert(
    checkpoint: Path,
    output_path: Path,
    base_model: str,
    repo_id: str | None,
    max_shard_size: str,
) -> None:
    source = resolve_checkpoint(checkpoint)
    output_path = output_path.expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading Echo checkpoint from {source}", flush=True)
    combined = load_file(str(source), device="cpu")
    component_states = {
        "dit": get_model_state_dict_from_combined_ckpt(combined, "model.diffusion_model"),
        "vae": get_model_state_dict_from_combined_ckpt(combined, "vae"),
        "audio_vae": get_model_state_dict_from_combined_ckpt(combined, "audio_vae"),
        "vocoder": get_model_state_dict_from_combined_ckpt(combined, "vocoder"),
    }
    del combined
    missing = [name for name, state in component_states.items() if not state]
    if missing:
        raise ValueError(f"Echo checkpoint is missing components: {missing}")

    tensor_counts = {}
    dit_state = component_states.pop("dit")
    transformer = convert_ltx2_transformer(dict(dit_state), version="2.3")
    tensor_counts["transformer"] = len(transformer.state_dict())
    save_component(transformer, output_path, "transformer", max_shard_size)
    del transformer
    gc.collect()

    connectors = convert_ltx2_connectors(dict(dit_state), version="2.3")
    tensor_counts["connectors"] = len(connectors.state_dict())
    save_component(connectors, output_path, "connectors", max_shard_size)
    del connectors, dit_state
    gc.collect()

    vae = convert_ltx2_video_vae(component_states.pop("vae"), version="2.3", timestep_conditioning=False)
    tensor_counts["vae"] = len(vae.state_dict())
    save_component(vae, output_path, "vae", max_shard_size)
    del vae
    gc.collect()

    audio_vae = convert_ltx2_audio_vae(component_states.pop("audio_vae"), version="2.3")
    tensor_counts["audio_vae"] = len(audio_vae.state_dict())
    save_component(audio_vae, output_path, "audio_vae", max_shard_size)
    del audio_vae
    gc.collect()

    vocoder = convert_ltx2_vocoder(component_states.pop("vocoder"), version="2.3")
    tensor_counts["vocoder"] = len(vocoder.state_dict())
    save_component(vocoder, output_path, "vocoder", max_shard_size)
    del vocoder
    gc.collect()

    write_modular_index(output_path, base_model=base_model, repo_id=repo_id)
    report = {
        "schema": "echo.diffusers.conversion.v1",
        "source": str(source),
        "precision": "bfloat16",
        "tensor_counts": tensor_counts,
        "base_model": base_model,
        "model_repo": repo_id,
        "vocoder_output_sample_rate": 48000,
    }
    (output_path / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Echo conversion complete: {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Echo BF16 checkpoint or release folder.")
    parser.add_argument("--output-path", type=Path, required=True, help="Destination Diffusers model directory.")
    parser.add_argument(
        "--base-model",
        default="google/gemma-3-12b-it",
        help="Gemma model repo/path containing the text encoder and tokenizer at its root.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Future Hub repo id for the converted Echo components. Defaults to the local output path.",
    )
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(
        checkpoint=args.checkpoint,
        output_path=args.output_path,
        base_model=args.base_model,
        repo_id=args.repo_id,
        max_shard_size=args.max_shard_size,
    )
