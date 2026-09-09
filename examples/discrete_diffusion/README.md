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

## Visualizing the sampling process

`animate_sampling.py` records the intermediate canvas of every denoising step (through each pipeline's
`callback_on_step_end`, without changing the pipelines) and renders it as an animation. It makes the difference
between the two families of discrete diffusion easy to see:

* Masked diffusion (LLaDA2): every slot starts as `[MASK]` and is progressively unmasked, revealing text from blanks.
* Uniform block diffusion (DiffusionGemma): every slot always holds a real token, so a canvas of random tokens is
  sharpened into text, one cached block at a time.

The script has two subcommands. `capture` runs a real pipeline and saves one trajectory as JSON. `render` turns one or
more trajectories into a self-contained HTML animation (and, with `--gif`, an animated GIF). Highlighted states use a
color-blind-safe palette that also stays legible in grayscale.

### Capture a trajectory

```bash
# LLaDA2 (masked diffusion)
python examples/discrete_diffusion/animate_sampling.py capture \
  --method llada2 --model_id inclusionAI/LLaDA2.1-mini \
  --prompt "Why is the sky blue? Explain in detail." --use_chat_template \
  --gen_length 512 --block_length 32 --num_inference_steps 32 \
  --out traj_llada2.json

# DiffusionGemma (uniform block diffusion); --scheduler picks one of the three supported samplers
python examples/discrete_diffusion/animate_sampling.py capture \
  --method diffusion_gemma --model_id google/diffusiongemma-26B-A4B-it \
  --prompt "Why is the sky blue?" --gen_length 512 --num_inference_steps 32 \
  --scheduler entropy_bound \
  --out traj_dg.json
```

`--scheduler` accepts `entropy_bound` (the released checkpoint's sampler), `discrete_ddim` (exact D3PM posterior,
ancestral), or `block_refinement` (confidence-committed refinement). Capture one trajectory per scheduler to compare
them on the same prompt and seed.

### Render an animation

```bash
# One or more trajectories become a single side-by-side animation
python examples/discrete_diffusion/animate_sampling.py render \
  traj_llada2.json traj_dg.json \
  --out sampling_animation.html --gif sampling_animation.gif
```

Trajectories with different step counts are resampled onto a shared timeline, so every panel starts and finishes
together regardless of how many steps each sampler took. Use `--max_frames` to cap the GIF length and `--cols` to set
the characters per line.

The script is also runnable with [uv](https://docs.astral.sh/uv/): `uv run examples/discrete_diffusion/animate_sampling.py ...`.
