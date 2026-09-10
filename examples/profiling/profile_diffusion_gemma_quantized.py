import argparse
import json
import sys
import time
import traceback

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, DiffusionGemmaForBlockDiffusion

from diffusers import BlockRefinementScheduler, DiffusionGemmaPipeline


def _emit_stage(stage: str):
    print(f"[profile] stage={stage}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Benchmark a quantized DiffusionGemmaPipeline run")
    parser.add_argument("--model_id", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument("--prompt", default="Why is the sky blue?")
    parser.add_argument("--gen_length", type=int, default=64)
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cache_implementation", choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--compile_decoder", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--disable_4bit", action="store_true")
    parser.add_argument("--max_new_tokens_report", type=int, default=120)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    load_in_4bit = device == "cuda" and not args.disable_4bit
    model_kwargs = {
        "dtype": torch.bfloat16 if device == "cuda" else torch.float32,
    }

    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["device_map"] = "auto"

    current_stage = "setup"
    result = {
        "model_id": args.model_id,
        "status": "fail",
        "stage": current_stage,
        "device": device,
        "load_in_4bit": load_in_4bit,
        "cache_implementation": args.cache_implementation,
        "compile_decoder": args.compile_decoder,
        "gen_length": args.gen_length,
        "num_inference_steps": args.num_inference_steps,
        "temperature": args.temperature,
        "wall_time_s": None,
        "tokens_returned": None,
        "tokens_per_second": None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2) if device == "cuda" else None,
        "text_preview": None,
        "error_type": None,
        "error_message": None,
    }

    started_at = time.perf_counter()
    try:
        current_stage = "load_model"
        result["stage"] = current_stage
        _emit_stage(current_stage)
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(args.model_id, **model_kwargs).eval()
        if device == "cpu":
            model.to("cpu")

        current_stage = "load_processor"
        result["stage"] = current_stage
        _emit_stage(current_stage)
        processor = AutoProcessor.from_pretrained(args.model_id)
        scheduler = BlockRefinementScheduler()
        pipe = DiffusionGemmaPipeline(model=model, scheduler=scheduler, processor=processor)

        if args.compile_decoder:
            current_stage = "compile_decoder"
            result["stage"] = current_stage
            _emit_stage(current_stage)
            if device != "cuda":
                raise RuntimeError("Decoder compile benchmark only supports CUDA")
            pipe.model.model.decoder = torch.compile(
                pipe.model.model.decoder,
                mode="reduce-overhead",
                fullgraph=True,
            )

        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        current_stage = "generate"
        result["stage"] = current_stage
        _emit_stage(current_stage)
        output = pipe(
            prompt=args.prompt,
            gen_length=args.gen_length,
            num_inference_steps=args.num_inference_steps,
            temperature=args.temperature,
            cache_implementation=args.cache_implementation,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - started_at
        text = output.texts[0]
        generated_tokens = int(output.sequences.shape[-1])
        result.update(
            {
                "status": "pass",
                "stage": current_stage,
                "wall_time_s": round(elapsed_s, 2),
                "tokens_returned": generated_tokens,
                "tokens_per_second": round(generated_tokens / elapsed_s, 2) if elapsed_s > 0 else None,
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2) if device == "cuda" else None,
                "text_preview": text[: args.max_new_tokens_report],
            }
        )
    except Exception as error:  # noqa: BLE001 - profiling should report failures instead of crashing
        elapsed_s = time.perf_counter() - started_at
        result.update(
            {
                "wall_time_s": round(elapsed_s, 2),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2) if device == "cuda" else None,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback_tail": traceback.format_exc().strip().splitlines()[-1],
            }
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
