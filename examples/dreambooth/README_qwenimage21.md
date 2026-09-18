# DreamBooth training example for Qwen-Image 2.1

[DreamBooth](https://huggingface.co/papers/2208.12242) is a method to personalize text-to-image models given just a few (3~5) images of a subject.

The `train_dreambooth_lora_qwenimage21.py` script shows how to implement the training procedure with [LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) and adapt it for [Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1).

This will also allow us to push the trained model parameters to the Hugging Face Hub platform.

Qwen-Image 2.1 also takes condition images. That task has its own script, `train_dreambooth_lora_qwenimage21_img2img.py`,
described in [Image-to-image (editing)](#image-to-image-editing) below.

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
pip install -r requirements_flux.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell (e.g., a notebook)

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

When running `accelerate config`, if we specify torch compile mode to True there can be dramatic speedups.
Note also that we use PEFT library as backend for LoRA training, make sure to have `peft>=0.14.0` installed in your environment.

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

This will also allow us to push the trained LoRA parameters to the Hugging Face Hub platform.

Now, we can launch training using:

```bash
export MODEL_NAME="Qwen/Qwen-Image-2.1"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-qwenimage21-lora"

accelerate launch train_dreambooth_lora_qwenimage21.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --use_8bit_adam \
  --learning_rate=2e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub
```

For using `push_to_hub`, make you're logged into your Hugging Face account:

```bash
hf auth login
```

To better track our training experiments, we're using the following flags in the command above:

* `report_to="wandb` will ensure the training runs are tracked on [Weights and Biases](https://wandb.ai/site). To use it, be sure to install `wandb` with `pip install wandb`. Don't forget to call `wandb login <your_api_key>` before training if you haven't done it before.
* `validation_prompt` and `validation_epochs` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.

### LoRA rank and alpha

`--rank` sets the dimension of the trainable LoRA matrices, and `--lora_alpha` scales what they contribute:
PEFT multiplies the LoRA update by `lora_alpha / rank`. Both default to 16 here, so the update is applied at
full strength out of the box.

Change one and the ratio moves with it:

* `lora_alpha == rank` - scale 1, the LoRA is applied at the strength it learned.
* `lora_alpha < rank` - scale below 1, a weaker LoRA. `--rank 16` on its own with `--lora_alpha 4` is scale
  0.25, which mostly shows up as a run that looks undertrained at a step count that should have been enough.
* `lora_alpha > rank` - scale above 1, a stronger effect without adding parameters.

> [!TIP]
> Raise `--rank` for capacity, and raise `--lora_alpha` with it unless you mean to change the strength.
> If the style takes but subjects start losing their shape, the run is overcooked: cut the steps or the
> learning rate before reaching for a smaller alpha.

## Model specifics

A few things differ from the other DreamBooth LoRA trainers, all of them following the model rather than a choice made here:

* **Resolutions are multiples of 32.** One latent token covers a 16x16 pixel tile and the transformer groups latents into 2x2 slots, so `--resolution` and every `--aspect_ratio_buckets` entry must divide by 32. The script raises on anything else rather than resizing silently.
* **Images are read as RGBA.** This VAE takes and returns four channels, so a three-channel tensor fails at its first convolution. Images without an alpha channel get an opaque one.
* **The prompt goes through Qwen3-VL**, as a processor rather than a tokenizer, and comes back as variable-length embeddings with a mask. There is no `--max_sequence_length`: the checkpoint's processor does not truncate.
* **The scheduler is used as shipped.** It sets `use_dynamic_shifting`, so the shift is derived from the sequence length at sampling time and the training sigmas stay unshifted.
* **`flex_attention` is worth having.** With it the block-causal mask runs as one compiled block-sparse pass; without it the model falls back to an exact multi-pass SDPA prefill, which gives the same results but costs more.

Validation images are generated at `--resolution`, with `--validation_num_inference_steps` (default 40) and classifier-free guidance off, which is the recipe the model's own docs use.

## Image-to-image (editing)

`train_dreambooth_lora_qwenimage21_img2img.py` trains the same transformer on pairs: a condition image, the image it
should become, and the instruction that describes the change. The condition image enters twice, as vision tokens in the
prompt and as clean latents ahead of the noisy target in the sequence, and the loss is taken on the target alone.

It needs a dataset that holds both images, so `--dataset_name` and `--cond_image_column` are required:

```bash
accelerate launch train_dreambooth_lora_qwenimage21_img2img.py \
  --pretrained_model_name_or_path="Qwen/Qwen-Image-2.1" \
  --dataset_name="my-username/my-edit-pairs" \
  --cond_image_column="cond_image" \
  --image_column="image" \
  --caption_column="caption" \
  --instance_prompt="make it snow" \
  --output_dir="trained-qwenimage21-edit-lora" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_prompt="make it snow" \
  --validation_image="path/to/a/photo.png" \
  --validation_epochs=25 \
  --seed="0"
```

What differs from the text-to-image script:

* **A batch shares one image-pad layout.** The transformer reads the layout from the first row of `img_mask`, so
  every sample in a batch has to place the condition image's tokens identically. That holds when the samples share a
  prompt and a bucket; otherwise train with `--train_batch_size 1`. The script raises rather than training on a
  misaligned batch.
* **Condition images cannot be small.** The vision-language processor upsamples images below its minimum pixel count,
  and then produces more vision tokens than the transformer has slots for. The script checks the two counts and says
  so. 256px is the smallest size that lines up; train at 1024 in practice.
* **Prompt embeddings are per sample**, since each one is encoded together with its own condition image, and they are
  cached that way.
* **A pair keeps its geometry.** The condition image is resized to the target's grid and takes the same crop and flip,
  so pairs that were aligned stay aligned.
* `--with_prior_preservation` and `--caption_dropout` are rejected: both introduce a second prompt layout in a batch.

## Notes

Additionally, we welcome you to explore the following CLI arguments:

* `--lora_layers`: The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v" will result in lora training of attention layers only. The default is `to_k,to_q,to_v,to_out.0`; the feed-forward layers are named `img_mlp.proj`, `img_mlp.gate_layer` and `img_mlp.out`.
* `--use_aspect_ratio_buckets` / `--aspect_ratio_buckets`: train on a set of aspect ratios instead of one square crop. Each batch is drawn from a single bucket.
* `--caption_dropout`: drop an instance caption in favour of the empty prompt with this probability.

We provide several options for optimizing memory optimization:

* `--offload`: When enabled, we will offload the text encoder and VAE to CPU, when they are not used.
* `cache_latents`: When enabled, we will pre-compute the latents from the input images with the VAE and remove the VAE from memory once done.
* `--use_8bit_adam`: When enabled, we will use the 8bit version of AdamW provided by the `bitsandbytes` library.

Refer to the [official documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage21) of the `QwenImage21Pipeline` to know more about the model and its preferred dtypes during inference.

## Using quantization

You can quantize the base model with [`bitsandbytes`](https://huggingface.co/docs/bitsandbytes/index) to reduce memory usage. To do so, pass a JSON file path to `--bnb_quantization_config_path`. This file should hold the configuration to initialize `BitsAndBytesConfig`. Below is an example JSON file:

```json
{
    "load_in_4bit": true,
    "bnb_4bit_quant_type": "nf4"
}
```
