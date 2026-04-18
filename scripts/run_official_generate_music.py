"""Invoke the REAL ACE-Step pipeline (`acestep.inference.generate_music`) directly.

This produces the ground-truth `official` wav for a parity comparison. We bypass
the interactive wizard and the LM by disabling `thinking` + all `use_cot_*`
flags. Output ends up as whatever file format the handler writes (usually
`.flac`); we re-read it and re-emit as `.wav` at the requested path.

Usage (jieyue):
    conda activate acestep_v15_train
    export PYTHONPATH=/root/data/repo/gongjunmin/workspace/ACE-Step-1.5
    python scripts/run_official_generate_music.py \\
        --example examples/text2music/example_01.json \\
        --variant turbo \\
        --out /vepfs-d-data/q-ace/gongjunmin/ace_parity_out/turbo_text2music_REAL_official.wav \\
        --duration 20 --seed 42
"""

import argparse
import json
import os
import shutil
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--original-repo",
                   default=os.environ.get("ACESTEP_ORIG", "/root/data/repo/gongjunmin/workspace/ACE-Step-1.5"))
    p.add_argument("--example", required=True, help="Path to JSON example (relative or absolute).")
    p.add_argument("--variant", default="turbo", choices=["turbo", "base", "sft"])
    p.add_argument("--task", default="text2music",
                   choices=["text2music", "cover", "repaint"])
    p.add_argument("--src-audio", default=None,
                   help="Source audio path for cover/repaint tasks.")
    p.add_argument("--repainting-start", type=float, default=5.0)
    p.add_argument("--repainting-end", type=float, default=15.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="Output .wav path.")
    return p.parse_args()


def main():
    args = parse_args()

    sys.path.insert(0, args.original_repo)

    example_paths = [args.example, os.path.join(args.original_repo, args.example)]
    example = None
    for p in example_paths:
        if os.path.exists(p):
            with open(p) as f:
                example = json.load(f)
            break
    if example is None:
        raise FileNotFoundError(f"Example not found: {args.example}")

    caption = example.get("caption", example.get("description", ""))
    lyrics = example.get("lyrics", "") or ""
    bpm = example.get("bpm")
    keyscale = example.get("keyscale") or ""
    timesignature = example.get("timesignature") or ""
    vocal_language = example.get("language", "en")
    duration = args.duration if args.duration is not None else float(example.get("duration", 30.0))

    print(f"[official] example caption: {str(caption)[:80]!r}")
    print(f"[official] variant={args.variant} task={args.task} duration={duration}s seed={args.seed}")

    # Pick checkpoint name for the AceStepHandler. The handler reads
    # `ACESTEP_CONFIG_PATH` (env) / `config_path` (kwarg) to select a DiT variant.
    variant_ckpt = {
        "turbo": "acestep-v15-turbo",
        "base":  "acestep-v15-base",
        "sft":   "acestep-v15-sft",
    }[args.variant]
    os.environ.setdefault("ACESTEP_CONFIG_PATH", variant_ckpt)

    # Make the handler stay entirely in local mode — don't download anything.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[official] initialising handlers ...")
    dit_handler = AceStepHandler()
    # Mirror cli.py:1400+: call initialize_service to bind a specific DiT variant.
    dit_handler.initialize_service(
        project_root=args.original_repo,
        config_path=variant_ckpt,
        device=device,
        use_flash_attention=dit_handler.is_flash_attention_available(device),
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
    )
    # We intentionally skip LM reasoning so the test is deterministic and doesn't
    # pollute the audio with LM-suggested metas/audio-codes we can't reproduce
    # from the diffusers side yet.
    llm_handler = LLMHandler()

    params = GenerationParams(
        task_type=args.task,
        caption=caption,
        lyrics=lyrics,
        vocal_language=vocal_language,
        bpm=bpm,
        keyscale=keyscale,
        timesignature=timesignature,
        duration=duration,
        thinking=False,
        use_cot_caption=False,
        use_cot_language=False,
        use_cot_metas=False,
        use_cot_lyrics=False,
        seed=args.seed,
        src_audio=args.src_audio if args.task in ("cover", "repaint") else None,
        repainting_start=args.repainting_start if args.task == "repaint" else 0.0,
        repainting_end=args.repainting_end if args.task == "repaint" else -1,
    )
    config = GenerationConfig(
        batch_size=1,
        use_random_seed=False,
        seeds=[args.seed],
        audio_format="wav",
    )

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    save_dir = os.path.join(out_dir, "_official_stage")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("[official] calling generate_music ...")
    result = generate_music(dit_handler, llm_handler, params, config, save_dir=save_dir)

    if not getattr(result, "success", True):
        raise RuntimeError(f"generate_music failed: {getattr(result, 'error', '<no error>')}")
    if not result.audios:
        raise RuntimeError("generate_music returned no audios")

    # Pick the first generated audio, copy/rename to --out.
    first = result.audios[0]
    audio_path = first.get("path") or first.get("audio_path")
    if audio_path is None:
        raise RuntimeError(f"no path in audio entry: {first!r}")
    if not os.path.isabs(audio_path):
        audio_path = os.path.join(save_dir, audio_path)
    print(f"[official] generated: {audio_path}")

    # Ensure .wav at the requested out path. If it's already .wav, just copy.
    import soundfile as sf
    a, sr = sf.read(audio_path)
    sf.write(args.out, a, sr)
    print(f"[official] wrote {args.out}  sr={sr} shape={a.shape}")


if __name__ == "__main__":
    main()
