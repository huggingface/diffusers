"""Full-audio parity harness for the ACE-Step diffusers integration.

Designed for the jieyue dev box (GPU, bf16 + flash_attention_2) but also works
on CPU/SDPA for quick sanity checks.

Runs the SAME example through:
  (1) the ORIGINAL ACE-Step 1.5 pipeline, and
  (2) the DIFFUSERS AceStepPipeline after the bug fixes,
with the same seed and precision, saving .wav files side-by-side for listening
verification by the reviewer.

Supports the three main task types — text2music, cover, repaint — on the
three model variants — turbo, base, sft. The reviewer listens to each
`{variant}_{task}_official.wav` next to the matching `{variant}_{task}_diffusers.wav`
and decides pass / fail.

Quick start (jieyue):
    conda activate acestep_v15_train
    # Single case:
    python scripts/audio_parity_jieyue.py \\
        --example examples/text2music/example_01.json \\
        --variant turbo --task text2music --out /tmp/ace_parity_out

    # Full matrix (9 cases per side):
    python scripts/audio_parity_jieyue.py --matrix --out /tmp/ace_parity_out

The matrix mode iterates {variant × task} and names outputs
`{variant}_{task}_{official|diffusers}.wav`.

Cover / repaint source audio is taken from
`<original_repo>/acestep_output/*.mp3` by default (ACE-Step's own demo
outputs) so the test is self-contained.
"""

import argparse
import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")


TASKS = ["text2music", "cover", "repaint"]
VARIANTS = ["turbo", "base", "sft"]
VARIANT_TO_CHECKPOINT = {
    "turbo": "acestep-v15-turbo",
    "base": "acestep-v15-base",
    "sft": "acestep-v15-sft",
}
# Which ACE-Step python module hosts `AceStepConditionGenerationModel` for a
# given variant. Keeps the original leg honest — don't cross-import modeling files.
VARIANT_TO_MODELING_MOD = {
    "acestep-v15-turbo": "acestep.models.turbo.modeling_acestep_v15_turbo",
    "acestep-v15-base": "acestep.models.base.modeling_acestep_v15_base",
    "acestep-v15-sft": "acestep.models.sft.modeling_acestep_v15_base",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--original-repo",
        default=os.environ.get("ACESTEP_ORIG", "/root/data/repo/gongjunmin/workspace/ACE-Step-1.5"),
    )
    p.add_argument(
        "--diffusers-repo",
        default=os.environ.get("ACESTEP_DIFF", "/root/data/repo/gongjunmin/diffusers"),
    )
    p.add_argument("--checkpoint-dir", default=None,
                   help="Defaults to <original-repo>/checkpoints.")
    p.add_argument("--sft-checkpoint-dir", default=None,
                   help="Override the SFT checkpoint subdir under checkpoint-dir.")
    p.add_argument("--variant", default="turbo", choices=VARIANTS)
    p.add_argument("--task", default="text2music", choices=TASKS)
    p.add_argument("--matrix", action="store_true",
                   help="Run the full variant × task matrix (9 × 2 = 18 wavs).")
    p.add_argument("--example", default="examples/text2music/example_01.json")
    p.add_argument("--src-audio", default=None,
                   help=".mp3/.wav source for cover (reference) and repaint (src). "
                        "Defaults to one of acestep_output/*.mp3.")
    p.add_argument("--repainting-start", type=float, default=5.0)
    p.add_argument("--repainting-end", type=float, default=15.0)
    p.add_argument("--audio-cover-strength", type=float, default=0.5)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--shift", type=float, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--attn", default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--out", default="/tmp/ace_parity_out",
                   help="Where to write .wav outputs.")
    p.add_argument("--converted-root", default=os.environ.get("ACESTEP_CONVERTED_ROOT", "/tmp"),
                   help="Cache dir for converted diffusers checkpoints (~5 GB per variant). "
                        "On jieyue use a vepfs path.")
    p.add_argument("--skip-original", action="store_true")
    p.add_argument("--skip-diffusers", action="store_true")
    return p.parse_args()


def load_example(path, original_repo):
    for c in [path, os.path.join(original_repo, path)]:
        if os.path.exists(c):
            with open(c) as f:
                return json.load(f)
    raise FileNotFoundError(f"Example not found: {path}")


def default_src_audio(original_repo):
    demo_dir = os.path.join(original_repo, "acestep_output")
    if not os.path.isdir(demo_dir):
        return None
    mp3s = sorted(f for f in os.listdir(demo_dir) if f.lower().endswith(".mp3"))
    return os.path.join(demo_dir, mp3s[0]) if mp3s else None


def load_audio_to_48k_stereo(path):
    import torchaudio

    audio, sr = torchaudio.load(path)
    if sr != 48000:
        audio = torchaudio.transforms.Resample(sr, 48000)(audio)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    return audio[:2].clamp(-1.0, 1.0)


def ensure_diffusers_checkpoint(args, variant_ckpt_name):
    # Each converted pipeline is ~5 GB (transformer + condition_encoder +
    # text_encoder + vae). Default the cache to --converted-root so users can
    # point it at a big disk (vepfs on jieyue, /tmp locally).
    out = os.path.join(args.converted_root, f"{variant_ckpt_name}-diffusers")
    if os.path.isdir(os.path.join(out, "transformer")):
        return out
    cmd = [
        "python",
        os.path.join(args.diffusers_repo, "scripts/convert_ace_step_to_diffusers.py"),
        "--checkpoint_dir", args.checkpoint_dir,
        "--dit_config", variant_ckpt_name,
        "--output_dir", out,
        "--dtype", "bf16",
    ]
    print(f"[audio-parity] converting: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    return out


# --------------------------------------------------------------------------- #
#                                 ORIGINAL LEG                                #
# --------------------------------------------------------------------------- #

def run_original_leg(args, variant_ckpt_name, example, task, src_audio_tensor):
    sys.path.insert(0, args.original_repo)
    import importlib
    import torch

    orig_mod = importlib.import_module(VARIANT_TO_MODELING_MOD[variant_ckpt_name])
    checkpoint = os.path.join(args.checkpoint_dir, variant_ckpt_name)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[parity][orig][{args.variant}/{task}] loading {checkpoint} ({dtype}, attn={args.attn}) ...")
    model = orig_mod.AceStepConditionGenerationModel.from_pretrained(
        checkpoint, torch_dtype=dtype
    )
    model.config._attn_implementation = args.attn
    model.eval().to(device)

    silence_latent = torch.load(os.path.join(checkpoint, "silence_latent.pt"),
                                map_location=device).to(dtype)

    from transformers import AutoModel, AutoTokenizer
    te_dir = os.path.join(args.checkpoint_dir, "Qwen3-Embedding-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(te_dir)
    text_encoder = AutoModel.from_pretrained(
        te_dir, torch_dtype=dtype, trust_remote_code=True
    ).eval().to(device)

    caption = example.get("caption", example.get("description", ""))
    lyrics = example.get("lyrics", "") or ""
    bpm = example.get("bpm")
    keyscale = example.get("keyscale") or ""
    timesignature = example.get("timesignature") or ""
    language = example.get("language", "en")
    duration = args.duration or float(example.get("duration", 30.0))

    from acestep.constants import SFT_GEN_PROMPT
    instructions = {
        "text2music": "Fill the audio semantic mask based on the given conditions:",
        "cover": "Generate audio semantic tokens based on the given conditions:",
        "repaint": "Repaint the mask area based on the given conditions:",
    }
    instruction = instructions[task]
    metas = (
        f"- bpm: {bpm if bpm else 'N/A'}\n"
        f"- timesignature: {timesignature or 'N/A'}\n"
        f"- keyscale: {keyscale or 'N/A'}\n"
        f"- duration: {int(duration)} seconds\n"
    )
    text_prompt = SFT_GEN_PROMPT.format(instruction, caption, metas)
    lyrics_prompt = f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"

    def encode_text(prompt_list, max_len):
        tok = tokenizer(prompt_list, padding="longest", truncation=True,
                        max_length=max_len, return_tensors="pt")
        return tok.input_ids.to(device), tok.attention_mask.to(device).bool()

    text_ids, text_mask = encode_text([text_prompt], 256)
    lyric_ids, lyric_mask = encode_text([lyrics_prompt], 2048)
    with torch.no_grad():
        text_hs = text_encoder(input_ids=text_ids).last_hidden_state.to(dtype)
        lyric_hs = text_encoder.get_input_embeddings()(lyric_ids).to(dtype)

    from diffusers import AutoencoderOobleck
    vae = AutoencoderOobleck.from_pretrained(
        os.path.join(args.checkpoint_dir, "vae"), torch_dtype=dtype
    ).eval().to(device)

    refer = torch.zeros(1, model.config.timbre_fix_frame,
                        model.config.timbre_hidden_dim, device=device, dtype=dtype)
    refer_order = torch.tensor([0], device=device, dtype=torch.long)
    is_covers = torch.tensor([0], device=device, dtype=torch.long)

    acoustic_dim = model.config.audio_acoustic_hidden_dim
    chunk_mode_all_mask = 2.0  # model-decided ('auto') repaint default

    if task == "text2music":
        latent_len = int(duration * 25)
        src_latents = torch.zeros(1, latent_len, acoustic_dim, device=device, dtype=dtype)
        chunk_mask = torch.ones(1, latent_len, acoustic_dim, device=device, dtype=dtype) * chunk_mode_all_mask
    elif task == "cover":
        assert src_audio_tensor is not None, "cover needs --src-audio"
        aud = src_audio_tensor.to(device=device, dtype=vae.dtype).unsqueeze(0)
        with torch.no_grad():
            ref_latents = vae.encode(aud).latent_dist.sample()
        ref_latents = ref_latents.transpose(1, 2).to(dtype)
        latent_len = ref_latents.shape[1]
        duration = latent_len / 25.0
        src_latents = ref_latents
        chunk_mask = torch.ones(1, latent_len, acoustic_dim, device=device, dtype=dtype) * chunk_mode_all_mask
        is_covers = torch.tensor([1], device=device, dtype=torch.long)
    elif task == "repaint":
        assert src_audio_tensor is not None, "repaint needs --src-audio"
        aud = src_audio_tensor.to(device=device, dtype=vae.dtype).unsqueeze(0)
        with torch.no_grad():
            src_lat = vae.encode(aud).latent_dist.sample()
        src_lat = src_lat.transpose(1, 2).to(dtype)
        latent_len = src_lat.shape[1]
        duration = latent_len / 25.0
        start_f = int(args.repainting_start * 25)
        end_f = int(min(args.repainting_end, duration) * 25)
        chunk_mask = torch.ones(1, latent_len, acoustic_dim, device=device, dtype=dtype)
        chunk_mask[:, start_f:end_f, :] = 0.0
        src_latents = src_lat * chunk_mask
    else:
        raise ValueError(task)

    attn_mask = torch.ones(1, latent_len, device=device, dtype=dtype)

    is_turbo = args.variant == "turbo"
    steps = args.steps or (8 if is_turbo else 27)
    shift = args.shift if args.shift is not None else (3.0 if is_turbo else 1.0)
    guidance = args.guidance_scale if args.guidance_scale is not None else (
        1.0 if is_turbo else 7.0
    )

    print(f"[parity][orig][{args.variant}/{task}] "
          f"duration={duration:.2f}s steps={steps} shift={shift} guidance={guidance}")

    kwargs = dict(
        text_hidden_states=text_hs,
        text_attention_mask=text_mask,
        lyric_hidden_states=lyric_hs,
        lyric_attention_mask=lyric_mask,
        refer_audio_acoustic_hidden_states_packed=refer,
        refer_audio_order_mask=refer_order,
        src_latents=src_latents,
        chunk_masks=chunk_mask,
        is_covers=is_covers,
        silence_latent=silence_latent,
        attention_mask=attn_mask,
        seed=args.seed,
        shift=shift,
    )
    if not is_turbo:
        kwargs["diffusion_guidance_scale"] = guidance
        kwargs["infer_steps"] = steps

    with torch.no_grad():
        result = model.generate_audio(**kwargs)
    latent = result["target_latents"]

    audio = vae.decode(latent.transpose(1, 2)).sample.float()
    std = audio.std(dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio = audio / std
    return latent.detach().cpu(), audio.detach().cpu(), vae.config.sampling_rate


# --------------------------------------------------------------------------- #
#                                 DIFFUSERS LEG                               #
# --------------------------------------------------------------------------- #

def run_diffusers_leg(args, variant_ckpt_name, example, task, src_audio_tensor):
    sys.path.insert(0, os.path.join(args.diffusers_repo, "src"))
    import torch
    from diffusers import AceStepPipeline

    model_dir = ensure_diffusers_checkpoint(args, variant_ckpt_name)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[parity][diff][{args.variant}/{task}] loading {model_dir} ({dtype}) ...")
    pipe = AceStepPipeline.from_pretrained(model_dir, torch_dtype=dtype, low_cpu_mem_usage=False)
    try:
        pipe.transformer.config._attn_implementation = args.attn
    except Exception:
        pass
    pipe = pipe.to(device)

    caption = example.get("caption", example.get("description", ""))
    lyrics = example.get("lyrics", "") or ""
    duration = args.duration or float(example.get("duration", 30.0))

    print(f"[parity][diff][{args.variant}/{task}] is_turbo={pipe.is_turbo} "
          f"defaults={pipe._variant_defaults()}")

    # Match the ORIGINAL's `prepare_noise`: `torch.Generator(device=device).manual_seed(seed)`.
    # The CUDA vs CPU RNG streams are NOT identical for the same seed; if the diffusers
    # pipeline's noise initialization follows a different device than the original, the
    # denoising trajectories diverge and the outputs aren't comparable.
    generator = torch.Generator(device=device).manual_seed(args.seed)
    call_kwargs = dict(
        prompt=caption,
        lyrics=lyrics,
        vocal_language=example.get("language", "en"),
        audio_duration=duration,
        num_inference_steps=args.steps,
        shift=args.shift,
        guidance_scale=args.guidance_scale,
        bpm=example.get("bpm"),
        keyscale=example.get("keyscale"),
        timesignature=example.get("timesignature"),
        generator=generator,
        output_type="pt",
        task_type=task,
    )
    if task == "cover":
        call_kwargs["reference_audio"] = src_audio_tensor
        call_kwargs["audio_cover_strength"] = args.audio_cover_strength
    elif task == "repaint":
        call_kwargs["src_audio"] = src_audio_tensor
        call_kwargs["repainting_start"] = args.repainting_start
        call_kwargs["repainting_end"] = args.repainting_end

    with torch.no_grad():
        out = pipe(**call_kwargs)
    return out.audios.float().detach().cpu(), pipe.vae.config.sampling_rate


# --------------------------------------------------------------------------- #
#                                    DRIVER                                   #
# --------------------------------------------------------------------------- #

def run_one(args, variant, task, example, src_audio, out_dir):
    import soundfile as sf

    variant_ckpt_name = VARIANT_TO_CHECKPOINT[variant]
    if variant == "sft" and args.sft_checkpoint_dir:
        variant_ckpt_name = args.sft_checkpoint_dir

    # Patch the variant onto args so child fns pick the right defaults.
    args.variant = variant

    src_audio_tensor = None
    if task in ("cover", "repaint"):
        if src_audio is None:
            print(f"[parity][skip] task={task} needs --src-audio; skipping.")
            return
        src_audio_tensor = load_audio_to_48k_stereo(src_audio)

    prefix = os.path.join(out_dir, f"{variant}_{task}")
    if not args.skip_original:
        try:
            _, audio, sr = run_original_leg(args, variant_ckpt_name, example, task, src_audio_tensor)
            sf.write(f"{prefix}_official.wav", audio[0].numpy().T, sr)
            print(f"[parity] wrote {prefix}_official.wav  ({audio.shape[-1] / sr:.2f}s)")
        except Exception as e:
            print(f"[parity][ERROR] original {variant}/{task} failed: {e}")

    if not args.skip_diffusers:
        try:
            audio, sr = run_diffusers_leg(args, variant_ckpt_name, example, task, src_audio_tensor)
            sf.write(f"{prefix}_diffusers.wav", audio[0].numpy().T, sr)
            print(f"[parity] wrote {prefix}_diffusers.wav  ({audio.shape[-1] / sr:.2f}s)")
        except Exception as e:
            print(f"[parity][ERROR] diffusers {variant}/{task} failed: {e}")


def main():
    args = parse_args()
    args.checkpoint_dir = args.checkpoint_dir or os.path.join(args.original_repo, "checkpoints")
    os.makedirs(args.out, exist_ok=True)

    example = load_example(args.example, args.original_repo)
    print(f"[parity] caption: {str(example.get('caption', example.get('description', '')))[:80]!r}")

    src_audio = args.src_audio or default_src_audio(args.original_repo)
    if src_audio:
        print(f"[parity] cover/repaint source audio: {src_audio}")
    else:
        print("[parity] no default source audio found (cover/repaint will be skipped)")

    if args.matrix:
        # Phase A: text2music for every variant (produces the source audio we'll
        # feed into cover / repaint if none was supplied).
        for variant in VARIANTS:
            print(f"\n[parity] === {variant} / text2music ===")
            run_one(args, variant, "text2music", example, src_audio, args.out)

        # If the caller didn't hand us a source file, use the first variant's
        # text2music official output as the shared cover/repaint input.
        if src_audio is None:
            fallback = os.path.join(args.out, f"{VARIANTS[0]}_text2music_official.wav")
            if os.path.exists(fallback):
                src_audio = fallback
                print(f"\n[parity] cover/repaint source audio (fallback): {src_audio}")
            else:
                print(f"\n[parity] WARNING: no source audio found; cover/repaint will be skipped.")

        # Phase B: cover + repaint with shared source audio.
        for variant in VARIANTS:
            for task in ("cover", "repaint"):
                print(f"\n[parity] === {variant} / {task} ===")
                run_one(args, variant, task, example, src_audio, args.out)
    else:
        run_one(args, args.variant, args.task, example, src_audio, args.out)

    print(f"\n[parity] DONE. .wav files in {args.out}")


if __name__ == "__main__":
    main()
