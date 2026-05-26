#!/usr/bin/env python
# Small async helper that re-rolls assistant responses for a tiny dataset
# slice through a locally-served target model (vLLM or SGLang — both expose
# the OpenAI /v1/chat/completions API).
#
# This is a smoke-test stand-in for SpecForge's scripts/regenerate_train_data.py
# and Automodel's nemo_automodel.components.speculative.regenerate, kept tiny so
# the full pipeline (serve -> regen -> train) fits in one sbatch.

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from datasets import load_dataset
from openai import AsyncOpenAI


async def regenerate(client, model, messages, max_tokens, temperature):
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True, help="served-model-name from the vLLM/SGLang server")
    p.add_argument("--dataset_name", default="wikitext")
    p.add_argument("--dataset_config_name", default="wikitext-2-raw-v1")
    p.add_argument("--split", default="train[:32]")
    p.add_argument("--text_column", default="text")
    p.add_argument("--output_path", required=True)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)  # SpecForge default
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    ds = load_dataset(args.dataset_name, args.dataset_config_name, split=args.split)
    prompts = [row[args.text_column] for row in ds if row.get(args.text_column, "").strip()]
    print(f"[regen] {len(prompts)} prompts; serving={args.server}; model={args.model}", file=sys.stderr)

    client = AsyncOpenAI(base_url=args.server, api_key="EMPTY")
    sem = asyncio.Semaphore(args.concurrency)

    async def one(i, prompt):
        messages = [{"role": "user", "content": prompt[:1024]}]
        async with sem:
            try:
                assistant = await regenerate(client, args.model, messages, args.max_tokens, args.temperature)
            except Exception as exc:
                print(f"[regen] sample {i} failed: {exc}", file=sys.stderr)
                return None
        # Emit both the flat "text" column (so train_dflash.py can tokenize directly)
        # and "conversations" (so downstream consumers that expect chat format work too).
        return {
            "id": i,
            "text": assistant,
            "conversations": messages + [{"role": "assistant", "content": assistant}],
        }

    results = await asyncio.gather(*[one(i, p) for i, p in enumerate(prompts)])
    rows = [r for r in results if r is not None]

    with open(args.output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[regen] wrote {len(rows)}/{len(prompts)} rows -> {args.output_path}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
