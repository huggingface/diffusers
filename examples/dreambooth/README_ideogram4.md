# DreamBooth training example for Ideogram 4

[DreamBooth](https://huggingface.co/papers/2208.12242) is a method to personalize image generation models given just a few (3~5) images of a subject/concept.
[LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) is a popular parameter-efficient fine-tuning technique that allows you to achieve full-finetuning-like performance with a fraction of the learnable parameters.

`train_dreambooth_lora_ideogram4.py` shows how to implement LoRA DreamBooth training for [Ideogram 4](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/ideogram4.md).

> [!NOTE]
> **About the model**
>
> Ideogram 4 is a flow-matching text-to-image model with a few characteristics that are relevant for training:
> - It uses **two** transformers — a text-conditional `transformer` and an `unconditional_transformer` blended at inference via asymmetric classifier-free guidance. This trainer adds LoRA to the **conditional `transformer` only**; the unconditional one stays frozen.
> - Text conditioning comes from a **Qwen3-VL** multimodal text encoder (a fixed set of decoder layers is concatenated into the per-token features).

## Running locally with PyTorch

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the `examples/dreambooth` folder and run

```bash
pip install -r requirements_ideogram4.txt
```

Initialize an [🤗 Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config default
```

We use the PEFT library as the backend for LoRA training; make sure `peft>=0.11.1` is installed.

### Quantized (nf4) base — QLoRA

Ideogram 4 is a large model, so a pre-quantized **nf4** checkpoint (`bitsandbytes`) is a convenient base for low-memory LoRA training. When the base checkpoint is already quantized, the trainer detects it automatically — you do **not** need to pass `--bnb_quantization_config_path` (that flag is for quantizing a full-precision checkpoint on the fly). The LoRA adapter is trained on top of the frozen 4-bit base (QLoRA) and saved in full precision.

### FP8 base — SDNQ checkpoint

Ideogram 4 is also distributed as an **SDNQ fp8** checkpoint ([`Disty0/Ideogram-4-SDNQ-FP8`](https://huggingface.co/Disty0/Ideogram-4-SDNQ-FP8)), about half the size of the bf16 weights. Training from it requires the [`sdnq`](https://github.com/Disty0/sdnq) library (`pip install sdnq`), which registers the backend needed to load the checkpoint. There are two ways to train from it:

**1. Train directly in fp8 — `--do_fp8_training`.** Pass `--do_fp8_training` with the SDNQ checkpoint as the base. The transformer is converted in place to SDNQ's training format and trained with fp8 scaled matmul on the forward and backward pass, keeping the weights in fp8 — the lowest-VRAM option. The LoRA adapter is still trained and saved in full precision.

```bash
accelerate launch train_dreambooth_lora_ideogram4.py \
  --pretrained_model_name_or_path="Disty0/Ideogram-4-SDNQ-FP8" \
  --do_fp8_training \
  --dataset_name="Norod78/Yarn-art-style" \
  --output_dir="trained-ideogram4-lora-fp8" \
  --instance_prompt="$INSTANCE_PROMPT" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --rank=16 \
  --optimizer="adamw" \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --mixed_precision="bf16" \
  --disable_training_autocast \
  --seed="0"
```

> [!NOTE]
> `--do_fp8_training` chooses its path from the checkpoint. An already-SDNQ-quantized base (fp8 or 4-bit) is trained in-place in fp8 as above. A **full-precision** base is instead converted to fp8 with [torchao](https://github.com/pytorch/ao), which gives fp8 *compute* but keeps bf16 storage (no memory saving). A `bitsandbytes` nf4 checkpoint is rejected — it is already a 4-bit QLoRA base, so train it without this flag (see above).

**2. Dequantize to bf16, then train.** Without `--do_fp8_training`, the trainer dequantizes the SDNQ transformer to bf16 on load and trains a standard bf16 LoRA — the same command as above, just omitting `--do_fp8_training`. This follows the regular bf16 training path but loads the transformer in full precision, so it uses more memory than option 1; combine it with the [memory optimizations](#memory-optimizations) below as needed.

## Prompt format

> [!IMPORTANT]
> Ideogram 4 is trained on structured **JSON captions** — a single-line JSON object that exhaustively describes the image — rather than free-form text. Plain text works, but the model understands the JSON structure natively, so captions in the schema generally train and generate best.

A caption is a JSON object; commonly used fields (see the upstream [ideogram-oss/ideogram4](https://github.com/ideogram-oss/ideogram4) prompt docs for the full schema) include:
- `high_level_description` — a one-line summary of the whole image.
- `compositional_deconstruction` — spatial layout, with a `background` string and an `elements` array; each element has a `type` (e.g. `"obj"`, `"text"`) and a `desc`.
- `colour_palette` — an array of hex colors to steer the image's color scheme.
- `bbox` — bounding-box coordinates for explicit placement of subjects, text, and background regions.

For best results, make each training caption describe its image as exhaustively as the schema allows.

For `--caption_column` / `--instance_prompt` (and at inference):
- **Recommended:** provide captions already in Ideogram 4's JSON caption schema.
- Or pass `--upsample_prompt` to rewrite free-form captions into the JSON schema during caching. This loads the prompt-enhancer LM head (`--prompt_enhancer_head_id`, default [`diffusers/qwen3-vl-8b-instruct-lm-head`](https://huggingface.co/diffusers/qwen3-vl-8b-instruct-lm-head)) as the pipeline's `prompt_enhancer_head`; install `outlines` for schema-constrained output.
- At inference, pass a short prompt with `prompt_upsampling=True` to rewrite it into the schema.

## Training example

For this example we use the [`Norod78/Yarn-art-style`](https://huggingface.co/datasets/Norod78/Yarn-art-style) dataset:

```bash
export MODEL_NAME="ideogram-ai/ideogram-4-nf4-diffusers"
export OUTPUT_DIR="trained-ideogram4-lora"
# Ideogram 4 expects a structured JSON caption (see "Prompt format" above).
export INSTANCE_PROMPT='{"high_level_description":"A puppy in a soft yarn-art style","compositional_deconstruction":{"background":"a plain cream studio backdrop","elements":[{"type":"obj","desc":"a small fluffy puppy crocheted from multicolored yarn, sitting upright and facing the viewer"}]}}'

accelerate launch train_dreambooth_lora_ideogram4.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name="Norod78/Yarn-art-style" \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="$INSTANCE_PROMPT" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --rank=16 \
  --optimizer="adamw" \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --mixed_precision="bf16" \
  --disable_training_autocast \
  --seed="0"
```

> [!IMPORTANT]
> Pass `--disable_training_autocast` when training Ideogram 4. Its forward is sensitive to mixed-precision autocast (bf16 → gray/noisy, fp16 → NaN), so training under accelerate's autocast makes the LoRA learn against corrupted predictions and produce fried outputs at inference. The flag disables the autocast wrapper and casts the transformer inputs to the weight dtype explicitly, matching the (autocast-free) inference path.

To track training with Weights & Biases add `--report_to="wandb"`, and to add periodic samples add `--validation_prompt="$INSTANCE_PROMPT" --validation_epochs=25` (a JSON caption, like the training prompt).

> [!NOTE]
> By default the LoRA weights are saved locally to `--output_dir`. To upload them to the Hub, add `--push_to_hub` (and `--hub_model_id`). Keep private datasets/LoRAs in private repos.

## Memory optimizations

Many of these can be combined:

- `--cache_latents` — pre-encode images with the VAE, then free it.
- `--offload` — offload the VAE / text encoder to CPU when not in use.
- `--gradient_accumulation_steps` — accumulate gradients to use a smaller effective batch.
- `--gradient_checkpointing` — recompute activations in the backward pass to save memory (slower).
- `--use_8bit_adam` — 8-bit AdamW optimizer (`bitsandbytes`); only applies to the `adamw` optimizer.
- `--resolution` — lower the training resolution (images are resized/cropped to this). Must be a multiple of 16; Ideogram 4 supports 256–2048.
- `--rank` — lower the LoRA rank for fewer trainable parameters.

### Precision of saved LoRA layers

By default the trained LoRA layers are saved in the training precision (e.g. `bf16` with `--mixed_precision="bf16"`). Pass `--upcast_before_saving` to save them in `float32` instead.

## Inference

After training, load the base pipeline and your LoRA:

```python
import torch
from diffusers import Ideogram4Pipeline

pipeline = Ideogram4Pipeline.from_pretrained("ideogram-ai/ideogram-4-nf4-diffusers", torch_dtype=torch.bfloat16)
pipeline.to("cuda")
pipeline.load_lora_weights("trained-ideogram4-lora", weight_name="pytorch_lora_weights.safetensors")

# Ideogram 4 expects a structured JSON caption (or pass a short prompt with prompt_upsampling=True).
prompt = '{"high_level_description":"A puppy in a soft yarn-art style","compositional_deconstruction":{"background":"a plain cream studio backdrop","elements":[{"type":"obj","desc":"a small fluffy puppy crocheted from multicolored yarn, sitting upright and facing the viewer"}]}}'
image = pipeline(prompt, height=1024, width=1024).images[0]
image.save("ideogram4_lora.png")
```

Ideogram 4 uses a guidance *schedule* by default; to use a constant scale instead, pass `guidance_scale=<value>, guidance_schedule=None` (exactly one of the two must be set, and a `guidance_schedule` must have length `num_inference_steps`).
