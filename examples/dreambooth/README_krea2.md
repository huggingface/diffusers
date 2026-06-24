# DreamBooth training example for Krea 2

[DreamBooth](https://huggingface.co/papers/2208.12242) is a method to personalize image generation models given just a few (3~5) images of a subject/concept.
[LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) is a popular parameter-efficient fine-tuning technique that allows you to achieve full-finetuning like performance but with a fraction of learnable parameters.

The `train_dreambooth_lora_krea2.py` script shows how to implement the training procedure for [LoRAs](https://huggingface.co/blog/lora) and adapt it for [Krea 2](https://www.krea.ai/).

> [!NOTE]
> **About Krea 2: RAW vs Turbo**
>
> Krea 2 ships as two checkpoints that are designed to work together:
> - **Krea 2 RAW** is the base model — a pre-trained checkpoint with **no distillation**. It is diverse and highly malleable, and it is the checkpoint you should use for **fine-tuning, post-training, and LoRA training**. It is *not* meant to be used for inference directly (do not expect high-quality outputs from it).
> - **Krea 2 Turbo** is an **8-step distilled** checkpoint built for fast, high-quality text-to-image **inference**.
>
> The recommended workflow is to **train your LoRA on RAW and run inference (and validation) on Turbo** — LoRAs trained on RAW express strongly on Turbo, so you get the best of both worlds: a malleable base to fine-tune and a fast, high-quality model to generate with.
>
> Architecturally, Krea 2 uses the Qwen-Image VAE, a 12B DiT (dense), and a Qwen3-VL text encoder with multi-layer feature aggregation.
>
> 📖 Read more here: Krea 2 release blog <!-- TODO: link to the Krea 2 release blog once it's published -->.

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
pip install -r requirements_krea2.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Note that we use the PEFT library as backend for LoRA training, so make sure to have `peft>=0.11.1` installed in your environment.

### Dog toy example

Now let's get our dataset. For this example we will use some dog images: https://huggingface.co/datasets/diffusers/dog-example.

Let's first download it locally:

```python
from huggingface_hub import snapshot_download

local_dir = "./dog"
snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

## Training

We train the LoRA on the **RAW** checkpoint. Because RAW is not meant for inference, validation and final inference are run on the **Turbo** checkpoint via `--validation_model_path` (see [Validation on Turbo](#validation-on-turbo)).

```bash
export MODEL_NAME="krea/Krea-2-Raw"
export TURBO_NAME="krea/Krea-2-Turbo"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-krea2-lora"

accelerate launch train_dreambooth_lora_krea2.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --cache_latents \
  --rank=32 \
  --lora_alpha=32 \
  --optimizer="adamW" \
  --use_8bit_adam \
  --learning_rate=3e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_model_path=$TURBO_NAME \
  --validation_prompt="a photo of sks dog" \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub
```

To better track our training experiments, we're using the following flags in the command above:

* `report_to="wandb"` will ensure the training runs are tracked on [Weights and Biases](https://wandb.ai/site). To use it, be sure to install `wandb` with `pip install wandb`.
* `validation_prompt`, `validation_epochs` and `validation_model_path` allow the script to run validation inference on Turbo during training (see below).

> [!NOTE]
> If you want to train using long prompts, you can use `--max_sequence_length` to set the token limit (default 512). Note that this uses more resources and may slow down training.

## Validation on Turbo

Since RAW is a non-distilled base that is **not meant for inference**, validating on RAW is misleading. Instead, pass `--validation_model_path` pointing at the **Turbo** checkpoint: at every validation step the script transplants the adapter currently being trained on RAW onto the Turbo pipeline and generates with it, so your validation images reflect what the final result will actually look like.

The Turbo inference recipe is the default for validation:

* `--validation_num_inference_steps` (default `8`) — Turbo is an 8-step distilled model.
* `--validation_guidance_scale` (default `0.0`) — Turbo runs without classifier-free guidance.
* `--validation_mu` (default `1.15`) — Turbo uses a fixed `mu` for the timestep shift instead of computing it from the resolution.

If `--validation_model_path` is omitted, validation and final inference fall back to the training checkpoint (using the pipeline defaults).

## Memory Optimizations

> [!NOTE]
> Many of these techniques complement each other and can be combined to further reduce memory consumption. Some are mutually exclusive, so check before launching.

### CPU Offloading
Pass `--offload` to offload the VAE and text encoder to CPU memory and only move them to GPU when needed.

### Latent Caching
Pre-encode the training images with the VAE and then free it. Enable with `--cache_latents`.

### Low-precision training with quantization
- **NF4 / 4-bit (QLoRA)** with `bitsandbytes`: pass `--bnb_quantization_config_path` pointing at a JSON of `BitsAndBytesConfig` kwargs (e.g. `{"load_in_4bit": true, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16"}`). This is the biggest single VRAM saver and lets a full Krea 2 LoRA run fit on a single mid-range GPU.
- **FP8 training** with `torchao`: pass `--do_fp8_training`. This uses FP8 scaled-matmul on a bf16-loaded transformer — it speeds up compute on supported hardware but, because the weights stay in bf16, it does not by itself reduce memory. Requires a GPU with compute capability ≥ 8.9. (`--do_fp8_training` and `--bnb_quantization_config_path` are mutually exclusive.)

### Gradient Checkpointing and Accumulation
* `--gradient_accumulation_steps` accumulates gradients over several steps before an update, reducing the number of backward/update passes.
* `--gradient_checkpointing` saves memory by recomputing intermediate activations during the backward pass instead of storing them (at the cost of a slower backward pass).

### 8-bit Adam Optimizer
When training with `AdamW` (not `prodigy`) pass `--use_8bit_adam` to reduce optimizer memory. Make sure `bitsandbytes` is installed.

### Image Resolution
`--resolution` sets the resolution all train/validation images are resized to (default 1024). Lowering it reduces memory.

### Precision of saved LoRA layers
By default trained layers are saved in the training precision (e.g. `bf16` under `--mixed_precision="bf16"`). Pass `--upcast_before_saving` to save them in `float32` instead (more memory).

## LoRA Rank, Alpha and Target Modules

Two key LoRA hyperparameters are rank and alpha:

- `--rank`: dimension of the trainable LoRA matrices. Higher rank = more capacity (and more parameters).
- `--lora_alpha`: scaling factor; the LoRA update is scaled by `lora_alpha / rank`. With `lora_alpha == rank` the scale is 1.0.

`--lora_layers` lets you choose exactly which modules to adapt (comma-separated). By default the script adapts the recommended layer set at rank/alpha 32:

```
img_in, final_layer.linear, to_q, to_k, to_v, to_out.0, to_gate,
ff.up, ff.down, text_fusion.projector, txt_in.linear_1, txt_in.linear_2,
time_embed.linear_1, time_embed.linear_2, time_mod_proj
```

> [!TIP]
> **Capacity: rank vs. target modules.** The default (rank/alpha **32** on the full layer set above) fits most styles, including ones with heavy high-frequency detail. For **long training runs**, it's recommended to add capacity by **increasing the rank and narrowing the target modules to the attention layers** — `--lora_layers="to_q,to_k,to_v,to_out.0,to_gate"` — rather than keeping the full layer set, so that prompt adherence doesn't degrade. In general, flat illustrative styles prefer **low-capacity** LoRAs (lower rank, fewer layers) and converge faster, while high-frequency styles (ink-brush paintings, etc.) benefit from more capacity.

> [!TIP]
> Standard learning rates of `3e-4 ~ 7e-4` with a `constant` schedule work well, and you can go a bit higher with a `cosine` schedule.

## Captioning for style LoRAs

For training a style, it's recommended to use captions that **describe the parts of the image you do *not* want baked into the LoRA, while omitting the stylistic parts you *do* want it to learn**, and add a descriptive **trigger phrase** as a style anchor. For example, for a hand-drawn-illustration style:

> "An astronaut standing beside a space rover on a flat landscape with cacti in the background while a large planet and stars are visible in the background. hand-drawn children's book illustration"

Here the phrase *"hand-drawn children's book illustration"* anchors the style and is preferred over a random rare token (e.g. `Ill3$tr@te`). For object/character training a trigger word is fine, as long as the captions broadly get the class of the subject right.

## Inference

Train on RAW, then load your LoRA into **Turbo** for fast, high-quality generation:

```python
import torch
from diffusers import Krea2Pipeline

pipe = Krea2Pipeline.from_pretrained("krea/Krea-2-Turbo", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Load your trained LoRA (trained on Krea 2 RAW)
pipe.load_lora_weights("path/to/your/trained-krea2-lora")

image = pipe(
    prompt="a photo of sks dog",
    height=1024,
    width=1024,
    num_inference_steps=8,
    guidance_scale=0.0,
    mu=1.15,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]

image.save("output.png")
```
