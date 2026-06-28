#!/usr/bin/env python3
# Copyright 2025 Stability AI and The HuggingFace Team. All rights reserved.
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
"""
End-to-end load + inference smoke test for the Stable Audio 3 diffusers pipeline.

This mirrors how a user would actually run the model: load a converted pipeline
directory with ``StableAudio3Pipeline.from_pretrained`` and generate audio. It
verifies the parts that the unit tests (tiny random configs) and the DiT
forward-parity check do *not*: that the real converted weights load into the
full pipeline, that ``__call__`` runs end to end, and that the output is a sane
waveform (right shape/duration, no NaN/Inf, in range).

Get a converted pipeline directory first:

    python scripts/convert_stable_audio_3_to_diffusers.py \\
        --checkpoint_path stabilityai/stable-audio-3-medium-base \\
        --model_config_path <model_config.json> \\
        --output_dir /path/to/sa3-diffusers \\
        --dtype float16

Then run inference:

    python scripts/run_stable_audio_3_inference.py \\
        --model_dir /path/to/sa3-diffusers \\
        --prompt "A gentle piano melody with soft strings in a concert hall" \\
        --duration 10.0 --num_inference_steps 8 --output output.wav

Exit code is non-zero if any sanity check fails, so it can gate CI.
"""

import argparse
import sys

import numpy as np
import torch


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _save_wav(path: str, audio: np.ndarray, sample_rate: int):
    """Write a (channels, samples) float array to a WAV file."""
    # transpose to (samples, channels) for the writers
    data = audio.T
    try:
        import soundfile as sf

        sf.write(path, data, samplerate=sample_rate)
        return
    except ImportError:
        pass
    try:
        from scipy.io import wavfile

        wavfile.write(path, sample_rate, np.clip(data, -1.0, 1.0).astype(np.float32))
        return
    except ImportError:
        pass
    raise RuntimeError("Install `soundfile` or `scipy` to save WAV output (or use --no_save).")


def main():
    args = parse_args()
    device = _pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    print(f"Device: {device} | dtype: {dtype}")

    from diffusers import StableAudio3Pipeline

    # ── 1. Load the converted pipeline ───────────────────────────────────────
    print(f"Loading pipeline from: {args.model_dir}")
    pipe = StableAudio3Pipeline.from_pretrained(args.model_dir, torch_dtype=dtype)
    if args.cpu_offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    print("✓ Pipeline loaded.")
    print(
        f"  vae={type(pipe.vae).__name__}  transformer={type(pipe.transformer).__name__}  "
        f"scheduler={type(pipe.scheduler).__name__}"
    )

    # ── 2. Inference ─────────────────────────────────────────────────────────
    gen_device = "cpu" if device == "mps" else device  # mps generators are flaky
    generator = torch.Generator(gen_device).manual_seed(args.seed)
    print(f"Generating: prompt={args.prompt!r}  duration={args.duration}s  steps={args.num_inference_steps}")

    out = pipe(
        args.prompt,
        duration=args.duration,
        num_inference_steps=args.num_inference_steps,
        num_waveforms_per_prompt=args.num_waveforms_per_prompt,
        generator=generator,
    )
    audio = out.audios  # (num_waveforms, channels, samples) tensor
    print(f"✓ Inference complete. Output tensor: {tuple(audio.shape)}")

    # ── 3. Sanity checks ─────────────────────────────────────────────────────
    sample_rate = int(pipe.vae.config.sampling_rate)
    expected_samples = int(args.duration * sample_rate)
    waveform = audio[0]  # (channels, samples)
    n_channels, n_samples = waveform.shape

    checks = []
    checks.append(("dimensionality (channels, samples)", waveform.ndim == 2))
    checks.append(("stereo (2 channels)", n_channels == 2))
    checks.append((f"sample count == duration*sr ({expected_samples})", n_samples == expected_samples))
    checks.append(("no NaN", not torch.isnan(waveform).any().item()))
    checks.append(("no Inf", not torch.isinf(waveform).any().item()))
    checks.append(("in range [-1, 1]", waveform.abs().max().item() <= 1.0 + 1e-4))
    checks.append(("non-silent (std > 1e-4)", waveform.float().std().item() > 1e-4))

    print("\n── Sanity checks ──")
    all_ok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok

    peak = waveform.abs().max().item()
    rms = waveform.float().pow(2).mean().sqrt().item()
    print(f"\n  peak={peak:.4f}  rms={rms:.4f}  duration={n_samples / sample_rate:.2f}s @ {sample_rate} Hz")

    # ── 4. Save ──────────────────────────────────────────────────────────────
    if not args.no_save:
        arr = waveform.cpu().float().numpy()
        _save_wav(args.output, arr, sample_rate)
        print(f"\n✓ Saved → {args.output}")

    print(f"\n{'✓ ALL CHECKS PASSED' if all_ok else '✗ SOME CHECKS FAILED'}")
    sys.exit(0 if all_ok else 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Load + inference smoke test for StableAudio3Pipeline.")
    parser.add_argument("--model_dir", type=str, required=True, help="Converted diffusers pipeline directory.")
    parser.add_argument("--prompt", type=str, default="A gentle piano melody with soft strings in a concert hall")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--num_waveforms_per_prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--cpu_offload", action="store_true", help="Use enable_model_cpu_offload (CUDA only).")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--no_save", action="store_true", help="Skip writing the WAV file.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
