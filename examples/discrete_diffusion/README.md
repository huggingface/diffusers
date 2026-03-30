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
