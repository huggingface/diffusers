"""Sample from DFlash on a GPU to verify the extracted pipeline works end-to-end.

Mirrors scripts/sample_llada2_gpu.py: loads a real draft + target pair, runs a single
prompt through DFlashPipeline, asserts the progress-bar config is not mutated by `__call__`
(LLaDA2 review issue #6), and prints the decoded text.

Pass --visualize to watch tokens materialize in the terminal block-by-block.
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from diffusers import DFlashPipeline

from contextlib import contextmanager

from token_display import TokenDisplay


@contextmanager
def _noop():
    yield


DEFAULT_DRAFT_MODEL_ID = "z-lab/Qwen3-8B-DFlash-b16"
DEFAULT_TARGET_MODEL_ID = "Qwen/Qwen3-8B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft_model_id", default=DEFAULT_DRAFT_MODEL_ID)
    parser.add_argument("--target_model_id", default=DEFAULT_TARGET_MODEL_ID)
    parser.add_argument("--prompt", default="What is 2 + 2? Answer in one sentence.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--dump_tokens", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="Live token-grid display.")
    parser.add_argument("--draft_pause", type=float, default=0.15, help="Seconds to hold draft frame before snapping to verified state.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device} torch={torch.__version__}", flush=True)

    print(f"[info] loading draft={args.draft_model_id}", flush=True)
    draft = AutoModel.from_pretrained(args.draft_model_id, trust_remote_code=True, dtype=torch.bfloat16).to(device)
    print(f"[info] loading target={args.target_model_id}", flush=True)
    target = AutoModelForCausalLM.from_pretrained(args.target_model_id, dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_id)

    pipe = DFlashPipeline(draft_model=draft, target_model=target, tokenizer=tokenizer)
    pipe.set_progress_bar_config(disable=True)
    before_cfg = dict(pipe._progress_bar_config)

    # mask_token_id lives in the draft model config when not on the tokenizer
    dflash_cfg = getattr(getattr(pipe.draft_model, "config", None), "dflash_config", {}) or {}
    mask_token_id = tokenizer.mask_token_id or dflash_cfg.get("mask_token_id")
    print(f"[info] mask_token_id={mask_token_id}", flush=True)
    display = TokenDisplay(tokenizer, mask_token_id, title="DFlash", draft_pause=args.draft_pause) if args.visualize else None

    print(f"[info] prompt: {args.prompt!r}", flush=True)

    with display or _noop():
        out = pipe(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": bool(args.enable_thinking)},
            callback_on_step_end=display,
            callback_on_step_end_tensor_inputs=["output_ids", "block_output_ids", "accepted_length"] if display else None,
        )

    print("[result] text:", out.texts[0], flush=True)
    print("[result] sequence shape:", tuple(out.sequences.shape), flush=True)
    if args.dump_tokens:
        decoded_no_skip = tokenizer.batch_decode(out.sequences, skip_special_tokens=False)[0]
        print("[result] text-no-skip:", repr(decoded_no_skip), flush=True)
        print("[result] tokens:", out.sequences[0].tolist()[:80], flush=True)

    after_cfg = dict(pipe._progress_bar_config)
    assert before_cfg == after_cfg, (before_cfg, after_cfg)
    print("[ok] progress-bar config preserved", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
