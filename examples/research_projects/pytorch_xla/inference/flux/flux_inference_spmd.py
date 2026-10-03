"""FLUX inference on TPU using PyTorch/XLA SPMD.

Uses SPMD to shard the transformer across multiple TPU chips, enabling
inference on devices where the model doesn't fit on a single chip (e.g., v5e).
The VAE is loaded on CPU at startup, moved to XLA for decode, then moved back.
"""

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import numpy as np
import structlog
import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.debug.profiler as xp
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.experimental.custom_kernel import FlashAttention

from diffusers import AutoencoderKL, FluxPipeline


cache_path = Path("/tmp/data/compiler_cache_eXp")
cache_path.mkdir(parents=True, exist_ok=True)
xr.initialize_cache(str(cache_path), readonly=False)
xr.use_spmd()

logger = structlog.get_logger()
metrics_filepath = "/tmp/metrics_report.txt"
VAE_SCALE_FACTOR = 8


def _vae_decode(latents, vae, height, width, device):
    """Move VAE to XLA, decode latents, move VAE back to CPU."""
    vae.to(device)
    latents = FluxPipeline._unpack_latents(latents, height, width, VAE_SCALE_FACTOR)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]
    vae.to("cpu")
    return image


def main(args):
    # --- SPMD mesh: 4-way model parallel to fit transformer + VAE on v5e chips ---
    num_devices = xr.global_runtime_device_count()
    if num_devices >= 4:
        mesh = xs.Mesh(np.arange(num_devices), (num_devices // 4, 4), ("data", "model"))
    else:
        NotImplementedError
    xs.set_global_mesh(mesh)
    logger.info(f"SPMD mesh: {mesh.mesh_shape}, axes: {mesh.axis_names}, devices: {num_devices}")

    # --- Profiler ---
    profile_path = Path("/tmp/data/profiler_out_eXp")
    profile_path.mkdir(parents=True, exist_ok=True)
    profiler_port = 9012
    profile_duration = args.profile_duration
    if args.profile:
        logger.info(f"starting profiler on port {profiler_port}")
        _ = xp.start_server(profiler_port)

    device = xm.xla_device()

    # --- Checkpoint ---
    if args.schnell:
        ckpt_id = "black-forest-labs/FLUX.1-schnell"
    else:
        ckpt_id = "black-forest-labs/FLUX.1-dev"

    # --- Text encoding (CPU) ---
    prompt = "photograph of an electronics chip in the shape of a race car with trillium written on its side"
    logger.info("encoding prompt on CPU...")
    text_pipe = FluxPipeline.from_pretrained(ckpt_id, transformer=None, vae=None, torch_dtype=torch.bfloat16).to("cpu")
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, _ = text_pipe.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=512
        )
    image_processor = text_pipe.image_processor
    del text_pipe

    # --- Load VAE on CPU (moved to XLA only for decode) ---
    logger.info("loading VAE on CPU...")
    vae = AutoencoderKL.from_pretrained(ckpt_id, subfolder="vae", torch_dtype=torch.bfloat16)

    # --- Load transformer and shard ---
    logger.info(f"loading flux transformer from {ckpt_id}")
    flux_pipe = FluxPipeline.from_pretrained(
        ckpt_id,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        vae=None,
        torch_dtype=torch.bfloat16,
    ).to(device)

    for name, param in flux_pipe.transformer.named_parameters():
        if param.dim() >= 2:
            spec = [None] * param.dim()
            largest_dim = max(range(param.dim()), key=lambda d: param.shape[d])
            spec[largest_dim] = "model"
            xs.mark_sharding(param, mesh, tuple(spec))

    flux_pipe.transformer.enable_xla_flash_attention(partition_spec=("data", None, None, None), is_flux=True)
    FlashAttention.DEFAULT_BLOCK_SIZES = {
        "block_q": 1536,
        "block_k_major": 1536,
        "block_k": 1536,
        "block_b": 1536,
        "block_q_major_dkv": 1536,
        "block_k_major_dkv": 1536,
        "block_q_dkv": 1536,
        "block_k_dkv": 1536,
        "block_q_dq": 1536,
        "block_k_dq": 1536,
        "block_k_major_dq": 1536,
    }

    width = args.width
    height = args.height
    guidance = args.guidance
    n_steps = 4 if args.schnell else 28

    prompt_embeds = prompt_embeds.to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    xs.mark_sharding(prompt_embeds, mesh, ("data", None, None))
    xs.mark_sharding(pooled_prompt_embeds, mesh, ("data", None))

    # --- Compilation run ---
    logger.info("starting compilation run...")
    ts = perf_counter()
    latents = flux_pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=28,
        guidance_scale=guidance,
        height=height,
        width=width,
        output_type="latent",
    ).images
    image = _vae_decode(latents, vae, height, width, device)
    image = image_processor.postprocess(image)[0]
    logger.info(f"compilation took {perf_counter() - ts} sec.")
    image.save("/tmp/compile_out.png")

    # --- Inference loop ---
    seed = 4096 if args.seed is None else args.seed
    xm.set_rng_state(seed=seed, device=device)
    times = []
    logger.info("starting inference run...")
    for _ in range(args.itters):
        ts = perf_counter()

        if args.profile:
            xp.trace_detached(f"localhost:{profiler_port}", str(profile_path), duration_ms=profile_duration)
        latents = flux_pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=n_steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            output_type="latent",
        ).images
        image = _vae_decode(latents, vae, height, width, device)
        image = image_processor.postprocess(image)[0]
        inference_time = perf_counter() - ts
        logger.info(f"inference time: {inference_time}")
        times.append(inference_time)

    logger.info(f"avg. inference over {args.itters} iterations took {sum(times) / len(times)} sec.")
    image.save("/tmp/inference_out.png")
    metrics_report = met.metrics_report()
    with open(metrics_filepath, "w+") as fout:
        fout.write(metrics_report)
    logger.info(f"saved metric information as {metrics_filepath}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--schnell", action="store_true", help="run flux schnell instead of dev")
    parser.add_argument("--width", type=int, default=1024, help="width of the image to generate")
    parser.add_argument("--height", type=int, default=1024, help="height of the image to generate")
    parser.add_argument("--guidance", type=float, default=3.5, help="guidance strength for dev")
    parser.add_argument("--seed", type=int, default=None, help="seed for inference")
    parser.add_argument("--profile", action="store_true", help="enable profiling")
    parser.add_argument("--profile-duration", type=int, default=10000, help="duration for profiling in msec.")
    parser.add_argument("--itters", type=int, default=15, help="items to run inference and get avg time in sec.")
    args = parser.parse_args()
    main(args)
