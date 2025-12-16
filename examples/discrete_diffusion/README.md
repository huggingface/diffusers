# Discrete Token Diffusion (Experimental)

This folder contains **training examples** for *discrete diffusion over token IDs* (language-model style), built to follow the `diffusers` + `accelerate` training conventions.

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
  --max_train_steps 1000
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
  --max_train_steps 1000
```

## UDLM-style uniform diffusion

`train_udlm.py` trains a uniform token diffusion model:
- Forward process: with probability `1 - alpha(t)`, replace tokens with a uniform random token
- Noise schedule: log-linear `alpha(t) = 1 - (1 - eps) * t`
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
