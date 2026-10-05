import argparse
import inspect
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import traceback
from pathlib import Path


CHECKPOINT_REVISIONS = {
    "black-forest-labs/FLUX.2-dev": "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c",
    "black-forest-labs/FLUX.2-klein-4B": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
    "black-forest-labs/FLUX.2-klein-9B": "92196c8e11f7b6cf2b7493e037d8c5345c559216",
}


def check_arguments(args):
    return [
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--image-size",
        str(args.image_size),
        "--checkpoints",
        *args.checkpoints,
    ]


def submit_job(args):
    from huggingface_hub import HfApi, get_token

    token = get_token()
    if token is None:
        raise ValueError("Authenticate with `hf auth login` or set HF_TOKEN before submitting a job.")

    repo_root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    with tempfile.TemporaryDirectory() as directory:
        archive_path = Path(directory) / "diffusers-source.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            for path in sorted((repo_root / "src" / "diffusers").rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    archive.add(path, arcname=path.relative_to(repo_root / "src"))

        job = HfApi(token=token).run_uv_job(
            str(Path(__file__).resolve()),
            script_args=["--source-archive", str(archive_path), *check_arguments(args)],
            dependencies=[
                f"diffusers @ https://github.com/huggingface/diffusers/archive/{revision}.tar.gz",
                "torch>=2.6",
                "accelerate>=0.31.0",
            ],
            python="3.12",
            flavor=args.flavor,
            timeout=args.timeout,
            secrets={"HF_TOKEN": token},
            name="flux2-vae-checkpoint-compatibility",
        )

    print(f"Job: {job.url}", flush=True)
    print(f"Logs: hf jobs logs {job.id}", flush=True)
    print(f"Status: hf jobs inspect {job.id}", flush=True)


def check_checkpoint(checkpoint, device, dtype, image_size):
    import torch

    from diffusers import AutoencoderKLFlux2

    revision = CHECKPOINT_REVISIONS.get(checkpoint)
    config = AutoencoderKLFlux2.load_config(checkpoint, subfolder="vae", revision=revision)
    model, loading_info = AutoencoderKLFlux2.from_pretrained(
        checkpoint,
        subfolder="vae",
        revision=revision,
        torch_dtype=dtype,
        output_loading_info=True,
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading_info[key]:
            raise AssertionError(f"{checkpoint}: {key}={loading_info[key]}")

    model = model.to(device).eval()
    generator = torch.Generator(device=device).manual_seed(0)
    sample = torch.randn(
        1, model.config.in_channels, image_size, image_size, generator=generator, device=device, dtype=dtype
    )
    with torch.inference_mode():
        latents = model.encode(sample).latent_dist.mode()
        reconstructed = model.decode(latents).sample
        scale_factor = 2 ** (len(model.config.block_out_channels) - 1)
        expected_latent_shape = (
            1,
            model.config.latent_channels,
            image_size // scale_factor,
            image_size // scale_factor,
        )
        assert latents.shape == expected_latent_shape, (latents.shape, expected_latent_shape)
        assert reconstructed.shape == (1, model.config.out_channels, image_size, image_size), reconstructed.shape
        assert torch.isfinite(latents).all(), "Non-finite latents"
        assert torch.isfinite(reconstructed).all(), "Non-finite decoded image"

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = AutoencoderKLFlux2.from_pretrained(directory, torch_dtype=dtype).to(device).eval()
            round_trip = reloaded(sample, sample_posterior=False).sample
            torch.testing.assert_close(round_trip, reconstructed)

    print(
        json.dumps(
            {
                "checkpoint": checkpoint,
                "revision": revision,
                "status": "PASS",
                "legacy_force_upcast": config.get("force_upcast"),
                "device": str(device),
                "dtype": str(dtype),
                "latent_shape": list(latents.shape),
                "image_shape": list(reconstructed.shape),
            }
        ),
        flush=True,
    )


def run_checks(args):
    import torch

    import diffusers
    from diffusers import AutoencoderKLFlux2

    if "force_upcast" in inspect.signature(AutoencoderKLFlux2.__init__).parameters:
        raise AssertionError("The loaded AutoencoderKLFlux2 still declares force_upcast.")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    dtype = getattr(torch, args.dtype)
    print(f"Diffusers: {diffusers.__version__} ({diffusers.__file__})", flush=True)
    print(f"PyTorch: {torch.__version__}; device={device}; dtype={dtype}", flush=True)
    failed = []
    for checkpoint in args.checkpoints:
        print(f"Checking {checkpoint}", flush=True)
        try:
            check_checkpoint(checkpoint, device, dtype, args.image_size)
        except Exception:
            failed.append(checkpoint)
            traceback.print_exc()
            print(json.dumps({"checkpoint": checkpoint, "status": "FAIL"}), flush=True)

    if failed:
        raise SystemExit(f"Failed checkpoints: {', '.join(failed)}")
    print(f"All {len(args.checkpoints)} checkpoints passed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Check FLUX.2 VAE loading and inference locally or on HF Jobs.")
    parser.add_argument(
        "--submit", action="store_true", help="Submit the script and current source checkout to HF Jobs."
    )
    parser.add_argument(
        "--flavor", default="cpu-basic", help="HF Jobs hardware flavor; for example, cpu-basic or l4x1."
    )
    parser.add_argument("--timeout", default="30m", help="HF Jobs timeout.")
    parser.add_argument(
        "--device", default="auto", help="PyTorch device; auto selects CUDA when available, otherwise CPU."
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--checkpoints", nargs="+", default=list(CHECKPOINT_REVISIONS))
    parser.add_argument("--source-archive", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.image_size <= 0 or args.image_size % 16:
        parser.error("--image-size must be a positive multiple of 16.")

    if args.submit:
        submit_job(args)
    elif args.source_archive:
        with tempfile.TemporaryDirectory() as directory:
            with tarfile.open(args.source_archive) as archive:
                archive.extractall(directory, filter="data")
            env = dict(os.environ, PYTHONPATH=directory)
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), *check_arguments(args)], env=env, check=True
            )
    else:
        run_checks(args)


if __name__ == "__main__":
    main()
