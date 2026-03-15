# Discrete Token Diffusion (Experimental)

This folder contains **training and sampling examples** for *discrete diffusion over token IDs* (language-model style), built to follow the `diffusers` + `accelerate` training conventions.

## Block refinement (commit-by-confidence)

Block refinement iteratively generates text in fixed-size blocks. At each step the model predicts all tokens in the block, commits the most confident ones, and re-masks the rest for further refinement.

### Train

The training script works with any causal LM from the Hub (e.g. Qwen, Llama, Mistral):

```bash
accelerate launch examples/discrete_diffusion/train_block_refinement.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --text_column text \
  --output_dir block-refinement-output \
  --max_train_steps 1000 \
  --prompt_length 32 \
  --block_length 32 \
  --lambda_conf 2.0 \
  --conf_temperature 0.5
```

If you don't want to download a dataset, you can use random-token data:

```bash
accelerate launch examples/discrete_diffusion/train_block_refinement.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --output_dir block-refinement-output \
  --use_dummy_data \
  --num_dummy_samples 2048
```

### Sample

```bash
python examples/discrete_diffusion/sample_block_refinement.py \
  --checkpoint_path block-refinement-output/final \
  --device cuda \
  --attention_mask_mode 2d \
  --prompt "Write a short paragraph about diffusion models." \
  --gen_length 128
```

For causal LMs that only support a 2D `attention_mask`, use `--attention_mask_mode 2d`.

## LLaDA2 sampling

[LLaDA2](https://huggingface.co/collections/inclusionAI/llada21) uses block refinement with a masked language model backbone. The `LLaDA2Pipeline` wraps `BlockRefinementPipeline` with LLaDA2-specific defaults.

```bash
python examples/discrete_diffusion/sample_llada2.py \
  --model_id inclusionAI/LLaDA2.1-mini \
  --prompt "Write a short poem about the ocean." \
  --gen_length 256 \
  --steps 32 \
  --threshold 0.7 \
  --editing_threshold 0.5 \
  --max_post_steps 16 \
  --use_chat_template \
  --add_generation_prompt
```
