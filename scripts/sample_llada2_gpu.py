"""Sample from LLaDA2 with the fixed pipeline to verify a real GPU run.

Exercises:
  - prompt path (carries tokenizer mask through, even for chat-template prompts)
  - eos_early_stop=True with the scheduler EOS-at-first-position fix
  - per-call block_length now drives the scheduler's transfer schedule
  - progress_bar_config disable preservation
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffusers import BlockRefinementScheduler, LLaDA2Pipeline

from token_display import TokenDisplay


MODEL_ID = "inclusionAI/LLaDA2.1-mini"


@contextmanager
def _noop():
    yield


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--prompt", default="What is 2 + 2? Answer in one sentence.")
    parser.add_argument("--gen_length", type=int, default=64)
    parser.add_argument("--block_length", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--visualize", action="store_true", help="Live token-grid display.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device} torch={torch.__version__}", flush=True)

    print(f"[info] loading {args.model_id}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, dtype=torch.bfloat16).to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    scheduler = BlockRefinementScheduler()

    pipe = LLaDA2Pipeline(model=model, scheduler=scheduler, tokenizer=tokenizer)
    pipe.set_progress_bar_config(disable=True)
    before_cfg = dict(pipe._progress_bar_config)

    mask_token_id = pipe.mask_token_id
    display = TokenDisplay(tokenizer, mask_token_id, title="LLaDA2") if args.visualize else None

    print(f"[info] prompt: {args.prompt!r}", flush=True)

    with display or _noop():
        out = pipe(
            prompt=args.prompt,
            gen_length=args.gen_length,
            block_length=args.block_length,
            num_inference_steps=args.num_inference_steps,
            temperature=0.0,
            threshold=0.7,
            eos_early_stop=True,
            callback_on_step_end=display,
            callback_on_step_end_tensor_inputs=["block_x", "transfer_index"] if display else None,
        )
    print("[result] text:", out.texts[0], flush=True)
    print("[result] sequence shape:", tuple(out.sequences.shape), flush=True)

    # Issue #6 regression: progress-bar config must survive the call.
    after_cfg = dict(pipe._progress_bar_config)
    assert before_cfg == after_cfg, (before_cfg, after_cfg)
    print("[ok] progress-bar config preserved", flush=True)

    # --- Batched + padded prompts (Issues #1, #5) ---
    prompts = [
        "What is 2 + 2? Answer in one sentence.",
        "Name a primary color.",
    ]
    print(f"[info] batched prompts: {prompts}", flush=True)
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    out_b = pipe(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        use_chat_template=False,
        gen_length=48,
        block_length=8,
        num_inference_steps=8,
        temperature=0.0,
        threshold=0.7,
        eos_early_stop=True,
    )
    for i, txt in enumerate(out_b.texts):
        print(f"[result-batched] row {i}: {txt!r}", flush=True)
    print("[result-batched] sequence shape:", tuple(out_b.sequences.shape), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
