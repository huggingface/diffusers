#!/usr/bin/env python
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
Sample script for BD3LM-style block diffusion decoding.

Example:
    python sample_bd3lm.py \
      --model_id kuleshov-group/bd3-lm \
      --prompt "Explain what reinforcement learning is in simple terms." \
      --max_new_tokens 256
"""

import argparse

import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from diffusers import BD3LMModel, BD3LMPipeline


def main():
    parser = argparse.ArgumentParser(description="Run BD3LM block diffusion decoding.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="kuleshov-group/bd3-lm",
        help="Model ID or local path.",
    )
    parser.add_argument(
        "--use_diffusers_model",
        action="store_true",
        help="Load a diffusers-native BD3LM checkpoint instead of transformers.",
    )
    parser.add_argument(
        "--tokenizer_id",
        type=str,
        default=None,
        help="Tokenizer ID or local path (defaults to `model_id`, falls back to `gpt2`).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain what reinforcement learning is in simple terms.",
        help="Prompt text to generate from.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--denoising_steps", type=int, default=64)
    parser.add_argument("--p_nucleus", type=float, default=1.0)
    parser.add_argument("--first_hitting", action="store_true")
    parser.add_argument("--no_first_hitting", action="store_true")
    parser.add_argument("--use_kv_cache", action="store_true")
    parser.add_argument("--max_window", type=int, default=1024)
    parser.add_argument("--entropy_threshold", type=float, default=4.0)
    parser.add_argument("--var_length", action="store_true")
    parser.add_argument("--mask_token_id", type=int, default=None)
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Use the tokenizer chat template for the prompt.",
    )
    parser.add_argument(
        "--add_generation_prompt",
        action="store_true",
        help="Add the generation prompt when using the chat template.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype.",
    )

    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(args.dtype)

    print(f"Loading model: {args.model_id}")
    if args.use_diffusers_model:
        model_kwargs = {}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        model = BD3LMModel.from_pretrained(args.model_id, **model_kwargs)
        config = model.config
    else:
        config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
        if not hasattr(config, "get"):
            config.get = lambda key, default=None: getattr(config, key, default)

        try:
            model = AutoModelForMaskedLM.from_pretrained(
                args.model_id,
                trust_remote_code=True,
                dtype=torch_dtype if torch_dtype is not None else "auto",
                config=config,
            )
        except AttributeError as err:
            if "all_tied_weights_keys" not in str(err):
                raise
            class_ref = config.auto_map.get("AutoModelForMaskedLM")
            if class_ref is None:
                raise ValueError("BD3LM config does not define AutoModelForMaskedLM in auto_map.") from err
            model_class = get_class_from_dynamic_module(
                class_ref, args.model_id, revision=None, trust_remote_code=True
            )
            if not hasattr(model_class, "all_tied_weights_keys"):
                model_class.all_tied_weights_keys = {}
            model = model_class.from_pretrained(
                args.model_id,
                trust_remote_code=True,
                dtype=torch_dtype if torch_dtype is not None else "auto",
                config=config,
            )
    model = model.to(args.device)
    tokenizer = None
    tokenizer_id = args.tokenizer_id or args.model_id
    tokenizer_kwargs = {}
    if not args.use_diffusers_model:
        tokenizer_kwargs["config"] = config
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True, **tokenizer_kwargs)
    except Exception:
        tokenizer_id = "gpt2" if args.tokenizer_id is None else args.tokenizer_id
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True, **tokenizer_kwargs)

    if tokenizer.mask_token_id is None:
        if getattr(config, "vocab_size", None) == len(tokenizer) + 1:
            tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        elif getattr(config, "vocab_size", None) == len(tokenizer):
            raise ValueError(
                "Tokenizer has no mask token but matches the model vocab size. "
                "Provide a tokenizer that already includes a mask token."
            )
        else:
            tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    first_hitting = args.first_hitting or not args.no_first_hitting

    pipe = BD3LMPipeline(model=model, tokenizer=tokenizer)

    print(f"\nPrompt: {args.prompt}")
    output = pipe(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        p_nucleus=args.p_nucleus,
        first_hitting=first_hitting,
        use_kv_cache=args.use_kv_cache,
        max_window=args.max_window,
        entropy_threshold=args.entropy_threshold,
        var_length=args.var_length,
        mask_token_id=args.mask_token_id,
        use_chat_template=args.use_chat_template,
        add_generation_prompt=args.add_generation_prompt,
    )

    print("\nGenerated text:")
    print(
        output.texts[0]
        if output.texts is not None
        else tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    )
    print(f"\nGenerated {output.sequences.shape[1]} tokens")


if __name__ == "__main__":
    main()
