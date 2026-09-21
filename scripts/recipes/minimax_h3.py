"""Convert an original MiniMax-H3 checkpoint into the diffusers layout.

The transformer checkpoint is streamed shard by shard, so peak memory stays close to a single shard (~4.9 GiB) and
never approaches the 62 GiB of the full 33B DiT.

Tensor layouts are converted by diffusers.loaders.conversion; this recipe supplies configuration and pipeline
metadata. The shared `minimax_h3_shards` layout handles per-head interleaved QKV and the FFN gate/value ordering.
Source preparation excludes the non-persistent `rope.inv_freq` buffer, which the model regenerates from config.

The FL2VA and Ref2VA variants differ only in the transformer weights, so the variant is selected by pointing
`--checkpoint_path` at the corresponding folder. Both land in one repository, which carries a single
`modular_model_index.json`: MiniMax-H3 is integrated as Modular Diffusers blocks only, so no `model_index.json` is
written.

Usage:

```bash
# Validate the key mapping without any weights present.
python scripts/recipes/minimax_h3.py \
    --checkpoint_path /path/to/MiniMax-H3/FL2VA --output_path /tmp/h3-diffusers --dry_run

# Convert, and point the component loading specs at the Hub id the result is published under.
python scripts/recipes/minimax_h3.py \
    --checkpoint_path /path/to/MiniMax-H3/FL2VA --output_path /tmp/h3-diffusers \
    --modular_repo_id MiniMaxAI/MiniMax-H3
```
"""

import argparse
import glob
import json
import os
import struct
from typing import Any

import torch

from diffusers import MiniMaxH3Transformer3DModel
from diffusers.loaders.conversion import get_conversion
from diffusers.loaders.conversion.configs.minimax_h3 import (
    MINIMAX_H3_TEST_TRANSFORMER_CONFIG,
    MINIMAX_H3_TEST_VIDEO_VAE_CONFIG,
    MINIMAX_H3_TRANSFORMER_CONFIG,
    MINIMAX_H3_VIDEO_VAE_CONFIG,
    get_audio_vae_config,
)
from diffusers.loaders.conversion.io import Checkpoint, convert_checkpoint
from diffusers.loaders.conversion.source import MergedCheckpoint


# `MiniMaxH3Transformer3DModel` argument names. The original config uses the sglang-native names listed in the
# comments; everything else in the original config (`adaln_out_features`, `final_adaln_out_features`) is derived.

# A tiny configuration with the checkpoint-tied dimensions left intact, for building fixtures.

# MiniMax-H3 ships a mixed-precision checkpoint. These *original* keys are float32; everything else is bfloat16 —
# including the AdaLN projections.
MINIMAX_H3_FP32_SOURCE_PREFIXES = (
    "video_patch_proj.",
    "audio_patch_proj.",
    "time_embedder.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
)

# `rope.inv_freq` is `1 / rope_theta ** (arange(0, 2 * rope_freq_dim, 2) / (2 * rope_freq_dim))`, which
# `MiniMaxH3RotaryPosEmbed` recomputes into a non-persistent buffer. The recomputed tensor is bitwise equal to the
# shipped one in both released variants, so the key is not carried into the diffusers checkpoint.
MINIMAX_H3_TRANSFORMER_DROPPED_KEYS = ("rope.inv_freq",)


def get_transformer_key_plan(config):
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel(**config)
    shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    conversion = get_conversion("MiniMaxH3Transformer3DModel", config)
    plan = {old: [(new, shapes[new])] for old, new in conversion.mapping.items()}
    for rule in conversion.rules:
        for old in rule.original:
            plan[old] = [(new, shapes[new]) for new in rule.diffusers]
    return plan


#
# ---------------------------------------------------------------------------------------------------------------
# Video VAE
# ---------------------------------------------------------------------------------------------------------------
#

# `AutoencoderKLMiniMaxH3` argument names. Field-for-field equal to `video_vae/source/config.json`, with the original
# names in the comments. The keys that only ever take one value in the release (`use_3d_conv`, `use_vit_decoder`,
# `causal_encoder`, `causal_decoder`, `use_t_isolated_gn`, `space_up` / `time_up`, `zq_ch_*`, `num_res_blocks_decoder`,
# `shift_factor` / `scaling_factor`) are baked into the port instead of being config knobs.

# A tiny configuration with the checkpoint-tied dimensions (`latent_channels`, the temporal geometry and the rotary
# ratio) left intact, for building fixtures and for the CPU parity check.

# `decoder.mask_token` is an all-zero buffer belonging to the masked-autoencoding training objective; the released
# decoder never reads it, so the port does not carry the module and the conversion drops the key.


def convert_video_vae(checkpoint_path, output_path, config, diffusers_version, max_shard_size):
    source_dir = os.path.join(checkpoint_path, "video_vae")
    with open(os.path.join(source_dir, "config.json")) as handle:
        wrapper = json.load(handle)
    path = os.path.join(source_dir, wrapper["source_path"], wrapper["source_safetensors_path"])
    config = {**config, "latents_mean": wrapper["latents_mean"], "latents_std": wrapper["latents_std"]}
    return convert_checkpoint(
        path, output_path, config=config, model_class="AutoencoderKLMiniMaxH3", max_shard_size=max_shard_size
    )


# Not present in `audio_vae/metadata.json`: the reference implementation hardcodes these in its DAC audio VAE and its
# attention projection, keyed off the sample rate.


def convert_audio_vae(checkpoint_path, output_path, diffusers_version):
    return convert_checkpoint(
        os.path.join(checkpoint_path, "audio_vae", "model.safetensors"),
        output_path,
        config=get_audio_vae_config(checkpoint_path),
        model_class="AutoencoderKLMiniMaxH3Audio",
    )


def write_scheduler_configs(checkpoint_path: str, output_path: str, diffusers_version: str) -> None:
    """Emit the two `MiniMaxH3Scheduler` configs, one per modality.

    The source `model_index.json` leaves `scheduler` null and instead carries the schedule constants in its
    `_minimax_h3.sigma_shift_scales` block. The sigma shift is the only per-modality difference, so it becomes two
    scheduler folders holding the same class at different `shift` values.
    """
    with open(os.path.join(checkpoint_path, "model_index.json")) as f:
        shift_scales = json.load(f)["_minimax_h3"]["sigma_shift_scales"]

    for folder, modality in (("scheduler", "video"), ("audio_scheduler", "audio")):
        folder_path = os.path.join(output_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, "scheduler_config.json"), "w") as f:
            json.dump(
                {
                    "_class_name": "MiniMaxH3Scheduler",
                    "_diffusers_version": diffusers_version,
                    "shift": float(shift_scales[modality]),
                },
                f,
                indent=2,
            )
    print(f"scheduler: shift={shift_scales['video']} (video), audio_scheduler: shift={shift_scales['audio']} (audio).")


def read_safetensors_header(path: str) -> dict[str, Any]:
    """Read the metadata header of a safetensors file without touching the tensor payload."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    return header


def dry_run(checkpoint_path: str, config: dict[str, Any]) -> None:
    plan = get_transformer_key_plan(config)

    transformer_dir = os.path.join(checkpoint_path, "transformer")
    shards = sorted(glob.glob(os.path.join(transformer_dir, "*.safetensors")))
    header: dict[str, Any] = {}
    for shard in shards:
        header.update(read_safetensors_header(shard))
    if shards:
        print(f"Read headers of {len(shards)} shard(s) in {transformer_dir}: {len(header)} keys present.\n")
    else:
        print(f"No shards found under {transformer_dir}; validating the plan against the config only.\n")

    print(f"{'original key':<48} {'->':^4} {'diffusers key':<52} {'shape':<24} dtype")
    print("-" * 150)
    num_target_keys = 0
    shape_mismatches: list[str] = []
    for source_key, targets in plan.items():
        present = source_key in header
        if not targets:
            print(f"{source_key:<48} {'-x':^4} {'(dropped, recomputed by the port)':<52}")
            continue
        for index, (target_key, shape) in enumerate(targets):
            num_target_keys += 1
            expected_dtype = "F32" if source_key.startswith(MINIMAX_H3_FP32_SOURCE_PREFIXES) else "BF16"
            if present:
                actual_dtype = header[source_key]["dtype"]
                actual_shape = header[source_key]["shape"]
                if index == 0 and len(targets) == 1 and actual_shape != shape:
                    shape_mismatches.append(f"{source_key}: header {actual_shape} != planned {shape}")
                if actual_dtype != expected_dtype:
                    shape_mismatches.append(f"{source_key}: header dtype {actual_dtype} != expected {expected_dtype}")
                marker = "->"
            else:
                marker = "->?"
            left = source_key if index == 0 else ""
            print(f"{left:<48} {marker:^4} {target_key:<52} {str(shape):<24} {expected_dtype}")

    missing = [key for key in plan if key not in header]
    unexpected = [key for key in header if key not in plan]

    print("\n" + "=" * 150)
    print(f"planned original keys : {len(plan)}")
    print(f"planned diffusers keys: {num_target_keys}")
    print(
        f"dropped original keys : {len(MINIMAX_H3_TRANSFORMER_DROPPED_KEYS)} {list(MINIMAX_H3_TRANSFORMER_DROPPED_KEYS)}"
    )
    print(f"fp32 diffusers keys   : {sum(1 for key in plan if key.startswith(MINIMAX_H3_FP32_SOURCE_PREFIXES))}")
    if shards:
        print(f"keys present in shards: {len(header)}")
        print(f"planned but absent    : {len(missing)}" + (" (shards still downloading?)" if missing else ""))
        print(f"present but unplanned : {len(unexpected)}")
        if unexpected:
            print(f"  {unexpected}")
        print(f"header disagreements  : {len(shape_mismatches)}")
        for line in shape_mismatches:
            print(f"  {line}")
    total_bytes = sum(
        (4 if source_key.startswith(MINIMAX_H3_FP32_SOURCE_PREFIXES) else 2) * torch.Size(shape).numel()
        for source_key, targets in plan.items()
        for _, shape in targets
    )
    print(f"total output bytes    : {total_bytes} ({total_bytes / 1024**3:.2f} GiB)")


def convert_transformer(checkpoint_path, output_path, config, max_shard_size):
    shards = sorted(glob.glob(os.path.join(checkpoint_path, "transformer", "*.safetensors")))
    source = MergedCheckpoint(("", Checkpoint(path)) for path in shards)
    return convert_checkpoint(
        source,
        output_path,
        config={**config, "original_format": "minimax_h3_shards"},
        model_class="MiniMaxH3Transformer3DModel",
        max_shard_size=max_shard_size,
    )


# The components a MiniMax-H3 repository holds, and the class each one loads as. `video_processor` is absent: the
# blocks create it from config rather than loading it.
MINIMAX_H3_COMPONENTS = {
    # The source names a checkpoint-local wrapper class (`MiniMaxH3Qwen3VLHFEncoder`); the conditioner is the
    # released Qwen3-VL, read at its 50th decoder layer with its language-model head unused.
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
    "tokenizer": ["transformers", "Qwen2TokenizerFast"],
    "processor": ["transformers", "Qwen3VLProcessor"],
    # Renamed to the diffusers audio/video VAE convention (see `LTX2Pipeline`).
    "vae": ["diffusers", "AutoencoderKLMiniMaxH3"],
    "audio_vae": ["diffusers", "AutoencoderKLMiniMaxH3Audio"],
    "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"],
    # One repository holds both checkpoint partitions: `transformer/` serves `MiniMaxH3Blocks` (`t2va` / `fl2va`) and
    # `transformer_ref/` serves `MiniMaxH3Ref2VABlocks`, while every other component is shared and converted once.
    "transformer_ref": ["diffusers", "MiniMaxH3Transformer3DModel"],
    # The source leaves `scheduler` null. MiniMax-H3 samples with Euler at eta=0 over shifted flow-matching sigmas, at
    # a different shift per modality, so it needs two scheduler entries (see `write_scheduler_configs`).
    "scheduler": ["diffusers", "MiniMaxH3Scheduler"],
    "audio_scheduler": ["diffusers", "MiniMaxH3Scheduler"],
}


def write_model_index(output_path: str, repo_id: str, diffusers_version: str) -> None:
    """Emit `modular_model_index.json`, the only index a MiniMax-H3 repository carries.

    MiniMax-H3 is integrated as Modular Diffusers blocks only, so there is no `model_index.json`: a modular repository
    declares one entry per component with its full loading spec rather than just its class, and a blockset then fetches
    exactly the subfolders it declares. That is what lets one repository hold both transformer partitions, and the
    original checkpoint folders next to the converted ones, without either half pulling the rest down.

    `_class_name` and `_blocks_class_name` name the `t2va` / `fl2va` half, which is what
    `ModularPipeline.from_pretrained` resolves to. The `ref2va` half reads the very same file through
    `MiniMaxH3Ref2VABlocks().init_pipeline(repo_id)`.

    The component map is the static one above, so this needs no source checkpoint: an index can be regenerated for a
    repository that is already published.
    """
    modular_index = {
        "_class_name": "MiniMaxH3ModularPipeline",
        "_diffusers_version": diffusers_version,
        "_blocks_class_name": "MiniMaxH3Blocks",
    }
    for name, (library, class_name) in MINIMAX_H3_COMPONENTS.items():
        modular_index[name] = [
            library,
            class_name,
            {
                "type_hint": [library, class_name],
                "pretrained_model_name_or_path": repo_id,
                "subfolder": name,
                "variant": None,
                "revision": None,
            },
        ]
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "modular_model_index.json"), "w") as f:
        json.dump(modular_index, f, indent=2)
    print(f"modular_model_index.json: {len(MINIMAX_H3_COMPONENTS)} components load from {repo_id}.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Local path to an original MiniMax-H3 variant folder (the one holding `transformer/`, `audio_vae/`, ...).",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Where the diffusers checkpoint is written.")
    parser.add_argument(
        "--modular_repo_id",
        type=str,
        default=None,
        help=(
            "Repository the component entries of `modular_model_index.json` point at. Defaults to `--output_path`, so "
            "pass the Hub id the checkpoint is published under. Every entry carries its own loading spec, so a "
            "blockset fetches exactly the subfolders it declares out of that repository."
        ),
    )
    parser.add_argument(
        "--version",
        type=str,
        default="h3",
        choices=["h3", "test"],
        help="`test` emits the tiny config used for fixtures.",
    )
    parser.add_argument(
        "--max_shard_size",
        type=int,
        default=5 * 1024**3,
        help="Maximum size of an output safetensors shard, in bytes.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the full planned key mapping (and cross-check any shard headers already present) without writing.",
    )
    return parser.parse_args()


def main(args):
    from diffusers import __version__ as diffusers_version

    config = MINIMAX_H3_TEST_TRANSFORMER_CONFIG if args.version == "test" else MINIMAX_H3_TRANSFORMER_CONFIG

    if args.dry_run:
        dry_run(args.checkpoint_path, config)
        return

    transformer_path = os.path.join(args.output_path, "transformer")
    convert_transformer(args.checkpoint_path, transformer_path, config, args.max_shard_size)
    video_vae_config = MINIMAX_H3_TEST_VIDEO_VAE_CONFIG if args.version == "test" else MINIMAX_H3_VIDEO_VAE_CONFIG
    convert_video_vae(
        args.checkpoint_path,
        os.path.join(args.output_path, "vae"),
        video_vae_config,
        diffusers_version,
        args.max_shard_size,
    )
    convert_audio_vae(args.checkpoint_path, os.path.join(args.output_path, "audio_vae"), diffusers_version)
    write_scheduler_configs(args.checkpoint_path, args.output_path, diffusers_version)
    write_model_index(args.output_path, args.modular_repo_id or args.output_path, diffusers_version)


if __name__ == "__main__":
    main(get_args())
