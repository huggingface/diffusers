from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import structlog
import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.debug.profiler as xp
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr
from torch_xla.experimental.custom_kernel import FlashAttention

from diffusers import FluxPipeline


logger = structlog.get_logger()
metrics_filepath = "/tmp/metrics_report.txt"


def _main(index, args, text_pipe, ckpt_id):
    cache_path = Path("/tmp/data/compiler_cache_tRiLlium_eXp")
    cache_path.mkdir(parents=True, exist_ok=True)
    xr.initialize_cache(str(cache_path), readonly=False)

    profile_path = Path("/tmp/data/profiler_out_tRiLlium_eXp")
    profile_path.mkdir(parents=True, exist_ok=True)
    profiler_port = 9012
    profile_duration = args.profile_duration
    if args.profile:
        logger.info(f"starting profiler on port {profiler_port}")
        _ = xp.start_server(profiler_port)
    device0 = xm.xla_device()

    logger.info(f"loading flux from {ckpt_id}")
    flux_pipe = FluxPipeline.from_pretrained(
        ckpt_id, text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None, torch_dtype=torch.bfloat16
    ).to(device0)
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

    prompt = "photograph of an electronics chip in the shape of a race car with trillium written on its side"
    width = args.width
    height = args.height
    guidance = args.guidance
    n_steps = 4 if args.schnell else 28

    logger.info("starting compilation run...")
    ts = perf_counter()
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = text_pipe.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=512
        )
    prompt_embeds = prompt_embeds.to(device0)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device0)

    image = flux_pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=28,
        guidance_scale=guidance,
        height=height,
        width=width,
    ).images[0]
    logger.info(f"compilation took {perf_counter() - ts} sec.")
    image.save("/tmp/compile_out.png")

    base_seed = 4096 if args.seed is None else args.seed
    seed_range = 1000
    unique_seed = base_seed + index * seed_range
    xm.set_rng_state(seed=unique_seed, device=device0)
    times = []
    logger.info("starting inference run...")
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = text_pipe.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=512
        )
    prompt_embeds = prompt_embeds.to(device0)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device0)
    for _ in range(args.itters):
        ts = perf_counter()

        if args.profile:
            xp.trace_detached(f"localhost:{profiler_port}", str(profile_path), duration_ms=profile_duration)
        image = flux_pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=n_steps,
            guidance_scale=guidance,
            height=height,
            width=width,
        ).images[0]
        inference_time = perf_counter() - ts
        if index == 0:
            logger.info(f"inference time: {inference_time}")
        times.append(inference_time)
    logger.info(f"avg. inference over {args.itters} iterations took {sum(times) / len(times)} sec.")
    image.save(f"/tmp/inference_out-{index}.png")
    if index == 0:
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
    if args.schnell:
        ckpt_id = "black-forest-labs/FLUX.1-schnell"
    else:
        ckpt_id = "black-forest-labs/FLUX.1-dev"
    text_pipe = FluxPipeline.from_pretrained(ckpt_id, transformer=None, vae=None, torch_dtype=torch.bfloat16).to("cpu")
    xmp.spawn(_main, args=(args, text_pipe, ckpt_id))
