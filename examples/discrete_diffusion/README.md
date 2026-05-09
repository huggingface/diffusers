# Discrete Token Diffusion (Experimental)

This folder contains **training and sampling examples** for *discrete diffusion over token IDs* (language-model style), built to follow the `diffusers` + `accelerate` training conventions.

## LLaDA2

[LLaDA2](https://huggingface.co/collections/inclusionAI/llada21) generates text through block-wise iterative refinement. Instead of autoregressive token-by-token generation, it starts with a fully masked sequence and progressively unmasks tokens by confidence over multiple refinement steps.

### Train

The training script uses confidence-aware loss and works with any causal LM from the Hub (e.g. Qwen, Llama, Mistral):

```bash
accelerate launch examples/discrete_diffusion/train_llada2.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --text_column text \
  --output_dir llada2-output \
  --max_train_steps 1000 \
  --prompt_length 32 \
  --block_length 32 \
  --lambda_conf 2.0 \
  --conf_temperature 0.5
```

If you don't want to download a dataset, you can use random-token data:

```bash
accelerate launch examples/discrete_diffusion/train_llada2.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --output_dir llada2-output \
  --use_dummy_data \
  --num_dummy_samples 2048
```

### Sample

```bash
python examples/discrete_diffusion/sample_llada2.py \
  --model_id inclusionAI/LLaDA2.1-mini \
  --prompt "Write a short poem about the ocean." \
  --gen_length 256 \
  --num_inference_steps 32 \
  --threshold 0.7 \
  --editing_threshold 0.5 \
  --max_post_steps 16 \
  --use_chat_template \
  --add_generation_prompt
```

## DFlash

[DFlash](https://huggingface.co/collections/z-lab/dflash) is a block-diffusion **speculative decoding** scheme: a small diffusion *draft* model, conditioned on hidden features from a frozen *target* causal LM, proposes a block of tokens that the target verifies in a single forward pass. The pipeline accepts the longest matching prefix and resamples the next token at the rejection point.

### Sample

The published draft pairs with a stock target (no `trust_remote_code` for the target):

```bash
python examples/discrete_diffusion/sample_dflash.py \
  --draft_model_id z-lab/Qwen3.5-4B-DFlash \
  --target_model_id Qwen/Qwen3.5-4B \
  --prompt "How many positive whole-number divisors does 196 have?" \
  --max_new_tokens 4096
```

The draft ships a custom `DFlashDraftModel` class via `auto_map`, so the sample script loads it with `trust_remote_code=True`; the target loads as a stock Qwen3 / Qwen3.5 model. Per-draft thinking-mode defaults from the upstream model cards:

| Draft | `enable_thinking` |
|---|---|
| `z-lab/Qwen3.5-*-DFlash` | `True` |
| `z-lab/Qwen3-*-DFlash-b16` | `False` (drafts are non-thinking-trained) |

### Train

The training loop conditions the draft on intermediate target hidden states and predicts the next `block_size − 1` tokens of each block:

```bash
accelerate launch examples/discrete_diffusion/train_dflash.py \
  --draft_model_id z-lab/Qwen3-4B-DFlash-b16 \
  --target_model_id Qwen/Qwen3-4B \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --text_column text \
  --output_dir dflash-output \
  --max_train_steps 1000 \
  --learning_rate 2e-5
```

`--block_size 0` (default) reads the block size from the draft model's config (16 for the b16 drafts, 16 for `Qwen3.5-*-DFlash`).
