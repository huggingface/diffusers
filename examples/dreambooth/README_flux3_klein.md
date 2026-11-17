# DreamBooth LoRA training for the FLUX.3 image model

These scripts train LoRA adapters for the FLUX.3 image model (`Flux3Transformer2DModel`, served by
`Flux3ModularPipeline`) with the DreamBooth recipe:

- `train_dreambooth_lora_flux3_klein.py` — text-to-image (t2i)
- `train_dreambooth_lora_flux3_klein_img2img.py` — image-to-image / instruction editing (i2i)

> [!NOTE]
> LoRA **loading and inference** for this model live in the pipeline/loader PR (`Flux3LoraLoaderMixin`); these
> scripts only cover **training**. Adapters are saved through `Flux3LoraLoaderMixin.save_lora_weights`, so they load
> back with `Flux3ModularPipeline.load_lora_weights`.

A few things that are specific to FLUX.3 and differ from the FLUX.1/FLUX.2 trainers:

- **Text encoder is Qwen3-VL.** The text embedding is the stack of several Qwen3-VL hidden layers
  (`--text_encoder_out_layers`, default `4 8 12 16 20 24 28 32`), concatenated along the channel dim. The prompt goes
  through the Qwen chat template and the text window is right-padded to a multiple of 80 tokens (capped at
  `--max_sequence_length`), matching the pipeline's `Flux3TextEncoderStep`.
- **No guidance distillation.** The model is not guidance-distilled, so there is no guidance argument in training;
  classifier-free guidance happens at inference. Use `--caption_dropout` for unconditional training (t2i and i2i).
- **VAE normalization is a batch-norm.** Latents are 2×2-patchified to 128 channels and normalized with the VAE's
  stored batch-norm statistics (not a scalar `scaling_factor`/`shift_factor`).
- **Flow matching with a fixed shift of 6.93** (no resolution-dependent dynamic shifting); the velocity target is
  `noise - image`.
- **Validation reuses the in-memory trained transformer** via `Flux3AutoBlocks().init_pipeline()` +
  `update_components(...)`, so the DiT is never loaded a second time.

## Running locally with PyTorch

### Installing the dependencies

Install `diffusers` from source and keep it up to date:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then `cd` into `examples/dreambooth` and install the requirements:

```bash
pip install -r requirements_flux3_klein.txt
```

Initialize an [🤗 Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or, for a default configuration:

```bash
accelerate config default
```

We use PEFT as the LoRA backend, so make sure `peft>=0.11.1` is installed.

## Text-to-image (t2i)

### Style LoRA example

This trains a *yarn-art* style LoRA on [Norod78/Yarn-art-style](https://huggingface.co/datasets/Norod78/Yarn-art-style),
whose captions already contain the trigger phrase `yarn art style`.

```bash
export MODEL_NAME="black-forest-labs/flux-3-image-klein-base"
export OUTPUT_DIR="trained-flux3-klein-yarn-lora"

accelerate launch train_dreambooth_lora_flux3_klein.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name="Norod78/Yarn-art-style" \
  --caption_column="text" \
  --instance_prompt="yarn art style" \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --cache_latents \
  --rank=32 \
  --lora_alpha=32 \
  --optimizer="adamW" \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_prompt="a cat, yarn art style" \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub
```

For a subject/object LoRA, point `--instance_data_dir` at a local folder of images and set `--instance_prompt` to a
caption with a trigger word (e.g. `a photo of sks dog`) instead of `--dataset_name`/`--caption_column`.

> [!NOTE]
> For long prompts, raise `--max_sequence_length` (the Qwen3-VL token cap, default 512). Longer windows use more
> memory and can slow down training.

## Image-to-image (i2i)

The i2i script trains an instruction-editing / relighting-style LoRA on **(condition image, target image,
instruction)** triplets. The reference image is VAE-encoded and fed to the model as a separate, clean conditioning
stream (its own position ids and timestep 0); the loss is computed on the generated image.

This example uses [kontext-community/relighting](https://huggingface.co/datasets/kontext-community/relighting)
(`file_name` = source, `output` = relit target, `instruction` = edit prompt):

```bash
export MODEL_NAME="black-forest-labs/flux-3-image-klein-base"
export OUTPUT_DIR="trained-flux3-klein-relight-lora"

accelerate launch train_dreambooth_lora_flux3_klein_img2img.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name="kontext-community/relighting" \
  --image_column="output" \
  --cond_image_column="file_name" \
  --caption_column="instruction" \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --cache_latents \
  --rank=16 \
  --lora_alpha=16 \
  --optimizer="adamW" \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --max_train_steps=1500 \
  --seed="0" \
  --validation_prompt="relight the scene with warm sunset light" \
  --validation_image="path/to/a/source/image.png" \
  --validation_epochs=25 \
  --push_to_hub
```

i2i reads its data through `--dataset_name` (a dataset with an image column, a conditioning-image column and a caption
column), not a plain image folder.

### Addressing the reference image in the prompt

FLUX.3 refers to each conditioning image passed to `image=` as `<ref_image_0>`, `<ref_image_1>`, ... in the order
they are passed. i2i training uses a single conditioning image, so captions address it as `<ref_image_0>`, e.g.
`"Relight <ref_image_0> with warm sunset light."` Matching this grammar in your captions keeps the adapter aligned
with how the base model reads references.

For explicit placement you can tag regions with an `<id>` and append a JSON list of regions to the caption. Each
region has an `id`, a `bbox` of `[y0, x0, y1, x1]` (top-left and bottom-right corners, origin top-left) normalized to
0-1000 on both axes, and a `desc`. When editing, a region also names its source reference with `from` (e.g.
`"ref_image_0"`), where it starts with `src_bbox`, and where it should end up with `tgt_bbox` (equal boxes keep it in
place):

```text
In <ref_image_0>, move the knight <knight_1> up and to the left, keeping everything else.
[{"id": "knight_1", "from": "ref_image_0", "src_bbox": [500, 150, 850, 350], "tgt_bbox": [194, 55, 544, 255],
  "desc": "A miniature knight crocheted from grey yarn."}]
```

See the [FLUX.3 pipeline docs](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/flux3.md)
for the full prompt grammar.

## Tracking

Pass `--report_to="wandb"` (or `tensorboard`) and `--validation_prompt` (plus `--validation_image` for i2i) to log
validation samples during training.

## Memory optimizations

- **CPU offloading** — `--offload` moves the VAE and text encoder to CPU except while they are used.
- **Latent caching** — `--cache_latents` pre-encodes images to latents once (the VAE is then freed), at the cost of no
  random image augmentations.
- **Low-precision training** — `--do_fp8_training` (torchao FP8) or `--bnb_quantization_config_path` (bitsandbytes
  QLoRA); the two are mutually exclusive.
- **Gradient checkpointing / accumulation** — `--gradient_checkpointing`, `--gradient_accumulation_steps`.
- **8-bit Adam** — `--optimizer="adamW" --use_8bit_adam` (or `--optimizer="prodigy"`).
- **Saved-LoRA precision** — `--upcast_before_saving` saves the adapter in fp32 instead of the training dtype.

## LoRA rank, alpha and target modules

`--rank` / `--lora_alpha` / `--lora_dropout` control the adapter. By default the scripts adapt the attention and
feed-forward projections across every block:

```
to_q, to_k, to_v, to_out.0, ff.linear_in, ff.linear_out
```

These are exactly the leaf modules the non-diffusers (ostris / ai-toolkit) FLUX.3 LoRA converter maps, so adapters
trained here stay interchangeable with that layout. These names match across the per-mode blocks (`mode_blocks.image`,
`mode_blocks.image_cond`, `txt_mode_blocks`) and the joint `single_transformer_blocks`, so a single target set reaches
every stream. Use `--lora_layers` (a comma-separated list) to target a different set, e.g. to also adapt
`context_embedder`, `x_embedders.image` or `proj_out`.

## Aspect-ratio bucketing and caption dropout

- `--aspect_ratio_buckets` / `--use_aspect_ratio_buckets` group images of similar aspect ratio so a batch shares a
  resolution (t2i).
- `--caption_dropout` replaces a fraction of captions (t2i) or instructions (i2i) with the empty prompt during
  training, which improves classifier-free guidance at inference. For i2i the conditioning image is kept.

## Inference

Load a trained adapter onto the modular pipeline and generate:

```python
import torch
from diffusers.modular_pipelines import Flux3AutoBlocks

pipe = Flux3AutoBlocks().init_pipeline("black-forest-labs/flux-3-image-klein-base")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

pipe.load_lora_weights("trained-flux3-klein-yarn-lora", weight_name="pytorch_lora_weights.safetensors")

image = pipe(prompt="a cat, yarn art style", output="images")[0]
image.save("yarn_cat.png")
```

For i2i, pass the reference image through the pipeline's `image=` input.
