#!/usr/bin/env python3
"""Run Flux2 text-to-image with tensor parallelism on TPU.

Usage (TPU topology env-vars must be set first):

    eval $(python -m torch_tpu._internal.distributed.launchers.singlehost_wrapper | sed 's/^/export /')
    python run_flux2_tp_tpu_pipeline.py --model-id black-forest-labs/FLUX.2-dev --output out.png
"""

import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401 — registers "tpu" device and "tpu_dist" backend
from torch.distributed.device_mesh import DeviceMesh
from torch_tpu._internal import sync as tpu_sync

from diffusers import Flux2Pipeline, TensorParallelConfig
from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu, retrieve_timesteps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="black-forest-labs/FLUX.2-dev")
    p.add_argument("--output", default="flux2_output.png")
    p.add_argument("--prompt", default="a photo of an astronaut riding a horse on mars")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tp-degree", type=int, default=4)
    args = p.parse_args()

    # self-relaunch under torchrun when not already a distributed worker
    if os.environ.get("LOCAL_RANK") is None:
        cmd = [sys.executable, "-m", "torch.distributed.run",
               f"--nproc-per-node={args.tp_degree}", os.path.abspath(__file__)] + sys.argv[1:]
        raise SystemExit(os.execvp(sys.executable, cmd))

    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    tp_mesh = DeviceMesh("tpu", list(range(args.tp_degree)))

    if rank == 0:
        print(f"Loading pipeline from {args.model_id} ...", flush=True)
    pipe = Flux2Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)

    # encode prompt on CPU — keeps the text encoder off TPU HBM
    if rank == 0:
        print("Encoding prompt ...", flush=True)
    prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=args.prompt, device=torch.device("cpu"), max_sequence_length=512,
    )

    # apply TP to the transformer and move it to TPU
    pipe.transformer.enable_parallelism(config=TensorParallelConfig(mesh=tp_mesh))
    pipe.transformer = pipe.transformer.to("tpu")
    tpu_sync.synchronize(None, wait=True)

    # prepare latents on CPU (avoids XLA generator issues), then move to TPU
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    num_channels = pipe.transformer.config.in_channels // 4
    latents, latent_ids = pipe.prepare_latents(
        batch_size=1, num_latents_channels=num_channels,
        height=args.height, width=args.width,
        dtype=torch.bfloat16, device=torch.device("cpu"), generator=generator,
    )
    latent_ids_cpu = latent_ids.clone()
    latents = latents.to("tpu")
    prompt_embeds = prompt_embeds.to("tpu", dtype=torch.bfloat16)
    text_ids = text_ids.to("tpu")
    latent_ids = latent_ids.to("tpu")

    mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=args.steps)
    sigmas = np.linspace(1.0, 1.0 / args.steps, args.steps)
    timesteps, num_steps = retrieve_timesteps(pipe.scheduler, args.steps, device="tpu", sigmas=sigmas, mu=mu)
    pipe.scheduler.set_begin_index(0)
    guidance = torch.full([1], args.guidance, device="tpu", dtype=torch.float32)

    if rank == 0:
        print(f"Denoising ({num_steps} steps) ...", flush=True)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            noise_pred = pipe.transformer(
                hidden_states=latents,
                timestep=t.expand(1).to(latents.dtype) / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            # flush XLA graph between steps to free HBM activations and avoid OOM
            tpu_sync.synchronize(None, wait=True)

    if rank == 0:
        latents_cpu = latents.detach().to("cpu").to(torch.float32)
        latents_cpu = pipe._unpack_latents_with_ids(latents_cpu, latent_ids_cpu)
        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents_cpu.device, latents_cpu.dtype)
        bn_std = torch.sqrt(
            pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
        ).to(latents_cpu.device, latents_cpu.dtype)
        latents_cpu = pipe._unpatchify_latents(latents_cpu * bn_std + bn_mean)
        with torch.no_grad():
            image = pipe.vae.decode(latents_cpu.to(torch.bfloat16), return_dict=False)[0]
        pipe.image_processor.postprocess(image.detach(), output_type="pil")[0].save(args.output)
        print(f"Saved → {os.path.abspath(args.output)}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
