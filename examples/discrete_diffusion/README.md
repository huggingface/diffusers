# Discrete Token Diffusion (Experimental)

This folder contains **training examples** for *discrete diffusion over token IDs* (language-model style), built to follow the `diffusers` + `accelerate` training conventions.

## Quickstart: block refinement with Qwen (causal LM)

If you want a causal-LM example, start here. This trains block refinement with a CAP-style confidence loss.

```bash
accelerate launch examples/discrete_diffusion/train_block_refinement_qwen_cap.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --text_column text \
  --output_dir qwen-block-refinement-output \
  --max_train_steps 1000 \
  --prompt_length 32 \
  --block_length 32 \
  --lambda_conf 2.0 \
  --conf_temperature 0.5
```

## MDLM-style absorbing diffusion

`train_mdlm.py` trains a masked/absorbing discrete diffusion model:
- Forward process: with probability `1 - alpha(t)`, replace tokens with `mask_token_id`
- Noise schedule: log-linear `alpha(t) = 1 - (1 - eps) * t`
- Loss: weighted token reconstruction NLL over masked positions

### Run

```bash
accelerate launch examples/discrete_diffusion/train_mdlm.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --output_dir mdlm-output \
  --max_train_steps 1000 \
  --lambda_conf 0.0 \
  --conf_temperature 1.0
```

The script saves:
- `transformers` model + tokenizer
- `diffusers.TokenDiffusionScheduler`

into `--output_dir` checkpoints and `--output_dir/final`.

### Sample

```bash
python examples/discrete_diffusion/sample_mdlm.py \
  --checkpoint_path mdlm-output/final \
  --num_samples 4 \
  --seq_len 64 \
  --num_inference_steps 128
```

## Block-wise sampling

Block-wise sampling updates the sequence in chunks, refining only the active block at a time.

```bash
python examples/discrete_diffusion/sample_block_token_diffusion.py \
  --checkpoint_path mdlm-output/final \
  --num_samples 4 \
  --seq_len 256 \
  --block_size 32 \
  --num_inference_steps 64 \
  --top_p 0.9
```

## Block refinement (commit-by-confidence) with Qwen

For causal LMs that only support a 2D `attention_mask`, run `BlockRefinementPipeline` with `--attention_mask_mode 2d`.

### Train

```bash
accelerate launch examples/discrete_diffusion/train_block_refinement_qwen_cap.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --text_column text \
  --output_dir qwen-block-refinement-output \
  --max_train_steps 1000 \
  --prompt_length 32 \
  --block_length 32 \
  --lambda_conf 2.0 \
  --conf_temperature 0.5
```

If you don't want to download a dataset, you can use random-token data:

```bash
accelerate launch examples/discrete_diffusion/train_block_refinement_qwen_cap.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --output_dir qwen-block-refinement-output \
  --use_dummy_data \
  --num_dummy_samples 2048
```

### Sample

```bash
python examples/discrete_diffusion/sample_block_refinement.py \
  --checkpoint_path qwen-block-refinement-output/final \
  --device cuda \
  --attention_mask_mode 2d \
  --prompt "Write a short paragraph about diffusion models." \
  --gen_length 128
```

## DFlash speculative decoding

Use a diffusion draft model with a target causal LM for block-wise speculative decoding.

```bash
python examples/discrete_diffusion/sample_dflash.py \
  --draft_model_id z-lab/Qwen3-8B-DFlash-b16 \
  --target_model_id Qwen/Qwen3-8B \
  --prompt "How many positive whole-number divisors does 196 have?" \
  --max_new_tokens 256 \
  --use_chat_template \
  --add_generation_prompt
```

## SDAR block diffusion decoding

Run SDAR-style block diffusion sampling with remasking strategies.

```bash
python examples/discrete_diffusion/sample_sdar.py \
  --model_id JetLM/SDAR-1.7B-Chat \
  --prompt "Explain what reinforcement learning is in simple terms." \
  --max_new_tokens 256 \
  --block_length 4 \
  --denoising_steps 4 \
  --remasking_strategy low_confidence_dynamic \
  --confidence_threshold 0.9 \
  --use_chat_template \
  --add_generation_prompt
```

### Fine-tune (draft model)

```bash
accelerate launch examples/discrete_diffusion/train_dflash.py \
  --draft_model_id z-lab/Qwen3-4B-DFlash-b16 \
  --target_model_id Qwen/Qwen3-4B \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --output_dir dflash-output \
  --max_train_steps 100 \
  --logging_steps 10
```

## Hybrid sampling

Hybrid sampling uses a different transition kernel than absorbing/uniform diffusion and requires a compatible scheduler
configuration saved in the checkpoint directory.

```bash
python examples/discrete_diffusion/sample_hybrid_token_diffusion.py \
  --checkpoint_path hybrid-output/final \
  --num_samples 4 \
  --seq_len 256 \
  --num_inference_steps 64
```

### Train

```bash
accelerate launch examples/discrete_diffusion/train_hybrid_token_diffusion.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --output_dir hybrid-output \
  --max_train_steps 1000 \
  --lambda_conf 0.0 \
  --conf_temperature 1.0
```

## UDLM-style uniform diffusion

`train_udlm.py` trains a uniform token diffusion model:
- Forward process: with probability `1 - alpha(t)`, replace tokens with a uniform random token
- Noise schedule: configurable via `--alpha_schedule` (`log_linear`, `linear`, `cosine`, `geometric`)
- Loss: diffusion loss for uniform token diffusion

### Run

```bash
accelerate launch examples/discrete_diffusion/train_udlm.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --output_dir udlm-output \
  --max_train_steps 1000 \
  --exclude_mask_from_uniform
```
