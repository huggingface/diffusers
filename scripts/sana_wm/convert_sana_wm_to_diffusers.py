#!/usr/bin/env python
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
#
# SPDX-License-Identifier: Apache-2.0
"""Convert the public SANA-WM release into a diffusers-loadable directory.

Reads the ``Efficient-Large-Model/SANA-WM_bidirectional`` HF repo (or a local
mirror) and writes a directory ready for ``SanaWMPipeline.from_pretrained(path)``:

    <output_dir>/
    ├── model_index.json
    ├── tokenizer/
    ├── text_encoder/
    ├── vae/
    ├── transformer/
    ├── scheduler/
    └── refiner/
        ├── transformer/
        ├── connectors/
        ├── text_encoder/
        └── tokenizer/

Usage:
    python scripts/sana_wm/convert_sana_wm_to_diffusers.py \\
        --src Efficient-Large-Model/SANA-WM_bidirectional \\
        --dst /path/to/SANA-WM_bidirectional-diffusers \\
        [--no-refiner]

The output is local-only; no upload to the Hub.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


def _copy_subdir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="Efficient-Large-Model/SANA-WM_bidirectional", help="HF repo or local dir")
    parser.add_argument("--dst", required=True, type=Path, help="Output directory")
    parser.add_argument("--no-refiner", action="store_true", help="Skip refiner export")
    parser.add_argument(
        "--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Weight dtype"
    )
    args = parser.parse_args()

    torch_dtype = getattr(torch, args.torch_dtype)
    dst: Path = args.dst.absolute()
    dst.mkdir(parents=True, exist_ok=True)

    # Resolve the source on disk (snapshot_download for HF repos, otherwise use as-is).
    src_path = Path(args.src)
    if not src_path.is_dir():
        print(f"[convert] snapshot_download({args.src}) …")
        src_path = Path(snapshot_download(args.src))
    print(f"[convert] source: {src_path}")

    # 1. VAE (already diffusers format under <src>/vae).
    print("[convert] vae …")
    _copy_subdir(src_path / "vae", dst / "vae")

    # 2. Tokenizer + text encoder — fetch via the configured Gemma-2 repo.
    # We save the full ``Gemma2ForCausalLM``; the pipeline grabs the decoder
    # at runtime via ``self.text_encoder.model(...)``. This matches the sana
    # inference recipe of ``AutoModelForCausalLM.from_pretrained(...).get_decoder()``
    # and avoids subtle state-dict prefix differences when saving just the
    # decoder submodule.
    print("[convert] tokenizer + text_encoder (gemma-2-2b-it) …")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gemma_repo = "Efficient-Large-Model/gemma-2-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(gemma_repo)
    tokenizer.padding_side = "right"
    tokenizer.save_pretrained(dst / "tokenizer")
    text_encoder = AutoModelForCausalLM.from_pretrained(gemma_repo, torch_dtype=torch_dtype)
    text_encoder.save_pretrained(dst / "text_encoder")
    del text_encoder

    # 3. Transformer (SanaWMTransformer3DModel) — load the public DiT, save in diffusers format.
    print("[convert] transformer (SanaWMTransformer3DModel) …")
    from diffusers import SanaWMTransformer3DModel

    transformer = SanaWMTransformer3DModel().to(torch_dtype).eval()
    dit_ckpt = src_path / "dit" / "sana_wm_1600m_720p.safetensors"
    if not dit_ckpt.is_file():
        raise FileNotFoundError(f"DiT checkpoint not found at {dit_ckpt}")
    from safetensors.torch import load_file

    sd = load_file(str(dit_ckpt))
    sd.pop("pos_embed", None)  # unused at inference (wan_rope is computed on-the-fly)
    sd = SanaWMTransformer3DModel.add_inner_prefix(sd)
    missing, unexpected = transformer.load_state_dict(sd, strict=False)
    if missing:
        missing_nontrivial = [k for k in missing if not k.endswith(".pos_embed")]
        if missing_nontrivial:
            print(f"  missing keys: {missing_nontrivial[:10]}{' …' if len(missing_nontrivial) > 10 else ''}")
    if unexpected:
        print(f"  unexpected keys: {unexpected[:10]}{' …' if len(unexpected) > 10 else ''}")
    transformer.save_pretrained(dst / "transformer")
    del transformer, sd

    # 4. Scheduler — FlowMatchEulerDiscreteScheduler config.
    print("[convert] scheduler …")
    from diffusers import FlowMatchEulerDiscreteScheduler

    FlowMatchEulerDiscreteScheduler(shift=9.8).save_pretrained(dst / "scheduler")

    # 5. Refiner (LTX-2): copy the four subfolders (already diffusers-format).
    if not args.no_refiner:
        print("[convert] refiner …")
        from diffusers.pipelines.sana_wm.refiner import SanaWMLTX2Refiner

        refiner_src = src_path / "refiner"
        refiner_dst = dst / "refiner"
        refiner_dst.mkdir(exist_ok=True)
        for sub in ("transformer", "connectors", "text_encoder"):
            if (refiner_src / sub).is_dir():
                _copy_subdir(refiner_src / sub, refiner_dst / sub)
        (refiner_dst / SanaWMLTX2Refiner.config_name).write_text(
            json.dumps({"text_max_sequence_length": 1024}, indent=2)
        )

    # 6. model_index.json — the top-level diffusers manifest.
    print("[convert] model_index.json …")
    index = {
        "_class_name": "SanaWMPipeline",
        "_diffusers_version": "0.38.0",
        "tokenizer": ["transformers", "GemmaTokenizerFast"],
        "text_encoder": ["transformers", "Gemma2ForCausalLM"],
        "vae": ["diffusers", "AutoencoderKLLTX2Video"],
        "transformer": ["diffusers", "SanaWMTransformer3DModel"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }
    if not args.no_refiner:
        index["refiner"] = ["diffusers", "SanaWMLTX2Refiner"]
    (dst / "model_index.json").write_text(json.dumps(index, indent=2))

    print(f"[convert] done — wrote {dst}")


if __name__ == "__main__":
    main()
