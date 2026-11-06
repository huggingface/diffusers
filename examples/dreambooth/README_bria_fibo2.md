# DreamBooth LoRA training for fibo-2 (Bria)

[fibo-2](https://huggingface.co/briaai) is Bria's flow-matching text-to-image and image-editing model
(`BriaFibo2Pipeline`): a `BriaFibo2Transformer2DModel` denoiser conditioned on a Qwen3-VL text encoder, with an
`AutoencoderKLFlux2` VAE. This directory has two LoRA trainers:

- `train_dreambooth_lora_bria_fibo2.py` — text-to-image.
- `train_dreambooth_lora_bria_fibo2_img2img.py` — image editing (image-to-image).

Both train a LoRA adapter on the transformer only; the text encoder and VAE are frozen.

> [!NOTE]
> fibo-2 is trained on **structured JSON captions** and renders them best, but a plain-text short description is
> accepted too (and used as-is). Keep your training captions in the form you intend to prompt with. The trainers
> clean captions the way the pipeline does, so a JSON caption and its cleaned, compact form train the same.

## Captions

fibo-2 was trained on **structured JSON captions**, so prompt and train in the same form. The trainers pass each
caption through the pipeline's own helpers, cleaning and compacting a JSON caption exactly as inference does (empty
values and the `aesthetic_score`/`preference_score` keys are dropped), so the model sees the same text both times.

**Text-to-image.** A caption is a JSON object; include only the keys you need:

```json
{
  "short_description": "A red bicycle leaning against a white brick wall on a sunny morning.",
  "objects": [{"description": "a red city bicycle with a wicker basket", "location": "center"}],
  "background_setting": "a white brick wall with a green door",
  "lighting": {"conditions": "bright morning sun", "direction": "from the right"},
  "style_medium": "photograph"
}
```

Pass one caption per image through `--caption_column` (each value a JSON string, or a plain sentence used as the
`short_description`), or a single `--instance_prompt` for the whole set. To place literal text in the image, fibo-2
uses a `text_render` field; the trainers here do not train that path, but the base model still renders short quoted
text in a prompt reasonably well.

**Image editing.** The edit instruction (`--caption_column`, with `--instance_prompt` as the fallback) takes either
form the pipeline accepts, and the two give different text embeddings — train in the one you will prompt with:

- a **plain instruction** — `"Make the background a snowy forest"`; or
- a **JSON object** with the instruction under `edit_instruction` and a structured caption of the *result*:
  `{"short_description": "...", "edit_instruction": "Make the background a snowy forest"}` (the caption may instead
  sit under a `structured_caption` key).

## Running locally

### Installing the dependencies

Install `diffusers` from source and then the trainer requirements:

```bash
pip install -e .
cd examples/dreambooth
pip install -r requirements_bria_fibo2.txt
```

And initialize an [🤗 Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment:

```bash
accelerate config default
```

fibo-2's Qwen3-VL text encoder needs `transformers>=5.20.1`.

## Text-to-image

Train a LoRA on a folder of instance images (one trigger prompt for all of them):

```bash
export MODEL_NAME="briaai/fibo-2"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-fibo2-lora"

accelerate launch train_dreambooth_lora_bria_fibo2.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed=0 \
  --push_to_hub
```

To train on a dataset with per-image captions, pass `--dataset_name` and `--caption_column` (the captions can be
structured JSON captions or plain short descriptions) instead of `--instance_data_dir`:

```bash
accelerate launch train_dreambooth_lora_bria_fibo2.py \
  --pretrained_model_name_or_path="briaai/fibo-2" \
  --dataset_name="your-dataset" \
  --caption_column="text" \
  --instance_prompt="in the style of TOK" \
  --output_dir="trained-fibo2-lora" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --max_train_steps=1000
```

### Inference

```python
from diffusers import BriaFibo2Pipeline
import torch

pipeline = BriaFibo2Pipeline.from_pretrained("briaai/fibo-2", torch_dtype=torch.bfloat16).to("cuda")
pipeline.load_lora_weights("trained-fibo2-lora", weight_name="pytorch_lora_weights.safetensors")
image = pipeline("A photo of sks dog in a bucket").images[0]
image.save("fibo2.png")
```

## Image editing (image-to-image)

The editing trainer learns from **(conditioning image, target image, edit instruction)** triplets, so it needs a
dataset (`--dataset_name`) with a source-image column, a target-image column and a caption column holding the
instruction. The conditioning and target images are given the same geometry so each pair stays aligned; the
conditioning image is both shown to the Qwen3-VL text encoder (as the image to edit) and VAE-encoded into the
transformer's clean context tokens, exactly as `BriaFibo2Pipeline` does at inference.

```bash
accelerate launch train_dreambooth_lora_bria_fibo2_img2img.py \
  --pretrained_model_name_or_path="briaai/fibo-2" \
  --dataset_name="your-edit-dataset" \
  --image_column="target_image" \
  --cond_image_column="source_image" \
  --caption_column="instruction" \
  --instance_prompt="make it a watercolor painting" \
  --output_dir="trained-fibo2-edit-lora" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --max_train_steps=2000 \
  --validation_prompt="make it a watercolor painting" \
  --validation_image="path/to/source.png" \
  --validation_epochs=25
```

`--instance_prompt` is the fallback instruction used when a pair has no caption. `--caption_column` holds the edit
instruction as plain text.

> [!NOTE]
> For editing, the conditioning image is shown to Qwen3-VL, and its vision tokens count towards
> `--max_sequence_length` (a 768px image is up to ~576 tokens). The editing trainer defaults it to 1024 for that
> reason; raise it further only if you pair large images with long structured captions.

### Inference

```python
from diffusers import BriaFibo2Pipeline
from diffusers.utils import load_image
import torch

pipeline = BriaFibo2Pipeline.from_pretrained("briaai/fibo-2", torch_dtype=torch.bfloat16).to("cuda")
pipeline.load_lora_weights("trained-fibo2-edit-lora", weight_name="pytorch_lora_weights.safetensors")
image = load_image("path/to/source.png")
edited = pipeline("make it a watercolor painting", image=image).images[0]
edited.save("fibo2_edit.png")
```

## Notes

- **LoRA layers.** By default the adapter targets **every linear layer** of the transformer (PEFT's `"all-linear"`),
  matching Bria's reference trainer. Pass `--lora_layers` with a comma-separated list to target a narrower set, e.g.
  `--lora_layers="to_q,to_k,to_v,to_out.0,w1,w2,w3"` for attention and feed-forward only, or
  `--lora_layers="to_q,to_k,to_v,to_out.0"` for attention only.
- **Aspect-ratio bucketing.** Pass `--use_aspect_ratio_buckets` to bucket images by aspect ratio (optionally with an
  explicit `--aspect_ratio_buckets` list), instead of resizing everything to a square `--resolution`.
- **Caption dropout.** `--caption_dropout 0.1` trains a fraction of samples unconditionally, which improves
  classifier-free guidance. For editing, the conditioning image is always kept; only the instruction is dropped.
- **Memory.** Use `--cache_latents` to pre-encode images, `--gradient_checkpointing`, `--use_8bit_adam`, and
  `--offload` to move the VAE and text encoder to CPU when idle. `--bnb_quantization_config_path` enables
  bitsandbytes-quantized training of the transformer, and `--do_fp8_training` enables FP8 (TorchAO) training.
- **Prodigy.** Pass `--optimizer="prodigy"` (with `--learning_rate=1.0`) to use the Prodigy optimizer.

> [!NOTE]
> These scripts are provided as a starting point; tune hyperparameters (rank, learning rate, steps) for your data.

## More resources

Bria publishes a single-GPU reference LoRA trainer, the full caption schema, and their recommended settings in the
model's [`fine_tuning` folder](https://huggingface.co/briaai/Fibo-2.0-dev/tree/main/fine_tuning). Their suggested
starting point is rank 64 (alpha = rank), learning rate 1e-4 constant, 1024px, gradient accumulation 8, and LoRA on
every linear layer. The trainers here also default to every linear layer; consider raising `--rank`/`--lora_alpha`
(the defaults are small) and lowering the target set with `--lora_layers` if you want a lighter adapter.
