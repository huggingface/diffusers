# DreamBooth training example for Stable Diffusion XL (SDXL)

[DreamBooth](https://huggingface.co/papers/2208.12242) is a method to personalize text2image models like stable diffusion given just a few (3~5) images of a subject.

The `train_dreambooth_lora_sdxl.py` script shows how to implement the training procedure and adapt it for [Stable Diffusion XL](https://huggingface.co/papers/2307.01952).

> 💡 **Note**: For now, we only allow DreamBooth fine-tuning of the SDXL UNet via LoRA. LoRA is a parameter-efficient fine-tuning technique introduced in [LoRA: Low-Rank Adaptation of Large Language Models](https://huggingface.co/papers/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*.

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
pip install -r requirements_sdxl.txt
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
Note also that we use PEFT library as backend for LoRA training, make sure to have `peft>=0.6.0` installed in your environment.

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
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="lora-trained-xl"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub
```

To better track our training experiments, we're using the following flags in the command above:

* `report_to="wandb` will ensure the training runs are tracked on [Weights and Biases](https://wandb.ai/site). To use it, be sure to install `wandb` with `pip install wandb`. Don't forget to call `wandb login <your_api_key>` before training if you haven't done it before.
* `validation_prompt` and `validation_epochs` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.

Our experiments were conducted on a single 40GB A100 GPU.

### Dog toy example with < 16GB VRAM

By making use of [`gradient_checkpointing`](https://pytorch.org/docs/stable/checkpoint.html) (which is natively supported in Diffusers), [`xformers`](https://github.com/facebookresearch/xformers), and [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes) libraries, you can train SDXL LoRAs with less than 16GB of VRAM by adding the following flags to your accelerate launch command:

```diff
+  --enable_xformers_memory_efficient_attention \
+  --gradient_checkpointing \
+  --use_8bit_adam \
+  --mixed_precision="fp16" \
```

and making sure that you have the following libraries installed:

```
bitsandbytes>=0.40.0
xformers>=0.0.20
```

### Inference

Once training is done, we can perform inference like so:

```python
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline
import torch

lora_model_id = <"lora-sdxl-dreambooth-id">
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_model_id)
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
image.save("sks_dog.png")
```

We can further refine the outputs with the [Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0):

```python
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch

lora_model_id = <"lora-sdxl-dreambooth-id">
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

# Load the base pipeline and load the LoRA parameters into it.
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_model_id)

# Load the refiner.
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
refiner.to("cuda")

prompt = "A picture of a sks dog in a bucket"
generator = torch.Generator("cuda").manual_seed(0)

# Run inference.
image = pipe(prompt=prompt, output_type="latent", generator=generator).images[0]
image = refiner(prompt=prompt, image=image[None, :], generator=generator).images[0]
image.save("refined_sks_dog.png")
```

Here's a side-by-side comparison of the with and without Refiner pipeline outputs:

| Without Refiner | With Refiner |
|---|---|
| ![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/sks_dog.png) | ![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/refined_sks_dog.png) |

### Training with text encoder(s)

Alongside the UNet, LoRA fine-tuning of the text encoders is also supported. To do so, just specify `--train_text_encoder` while launching training. Please keep the following points in mind:

* SDXL has two text encoders. So, we fine-tune both using LoRA.
* When not fine-tuning the text encoders, we ALWAYS precompute the text embeddings to save memory.

### Specifying a better VAE

SDXL's VAE is known to suffer from numerical instability issues. This is why we also expose a CLI argument namely `--pretrained_vae_model_name_or_path` that lets you specify the location of a better VAE (such as [this one](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)).

## Notes

In our experiments, we found that SDXL yields good initial results without extensive hyperparameter tuning. For example, without fine-tuning the text encoders and without using prior-preservation, we observed decent results. We didn't explore further hyper-parameter tuning experiments, but we do encourage the community to explore this avenue further and share their results with us 🤗

## Results

You can explore the results from a couple of our internal experiments by checking out this link: [https://wandb.ai/sayakpaul/dreambooth-lora-sd-xl](https://wandb.ai/sayakpaul/dreambooth-lora-sd-xl). Specifically, we used the same script with the exact same hyperparameters on the following datasets:

* [Dogs](https://huggingface.co/datasets/diffusers/dog-example)
* [Starbucks logo](https://huggingface.co/datasets/diffusers/starbucks-example)
* [Mr. Potato Head](https://huggingface.co/datasets/diffusers/potato-head-example)
* [Keramer face](https://huggingface.co/datasets/diffusers/keramer-face-example)

## Running on a free-tier Colab Notebook

Check out [this notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_DreamBooth_LoRA_.ipynb).

## Conducting EDM-style training

It's now possible to perform EDM-style training as proposed in [Elucidating the Design Space of Diffusion-Based Generative Models](https://huggingface.co/papers/2206.00364).

For the SDXL model, simple set:

```diff
+  --do_edm_style_training \
```

Other SDXL-like models that use the EDM formulation, such as [playgroundai/playground-v2.5-1024px-aesthetic](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic), can also be DreamBooth'd with the script. Below is an example command:

```bash
accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="playgroundai/playground-v2.5-1024px-aesthetic"  \
  --instance_data_dir="dog" \
  --output_dir="dog-playground-lora" \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --use_8bit_adam \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub
```

> [!CAUTION]
> Min-SNR gamma is not supported with the EDM-style training yet. When training with the PlaygroundAI model, it's recommended to not pass any "variant".

### DoRA training
The script now supports DoRA training too!
> Proposed in [DoRA: Weight-Decomposed Low-Rank Adaptation](https://huggingface.co/papers/2402.09353),
**DoRA** is very similar to LoRA, except it decomposes the pre-trained weight into two components, **magnitude** and **direction** and employs LoRA for _directional_ updates to efficiently minimize the number of trainable parameters.
The authors found that by using DoRA, both the learning capacity and training stability of LoRA are enhanced without any additional overhead during inference.

> [!NOTE]
> 💡DoRA training is still _experimental_
> and is likely to require different hyperparameter values to perform best compared to a LoRA.
> Specifically, we've noticed 2 differences to take into account your training:
> 1. **LoRA seem to converge faster than DoRA** (so a set of parameters that may lead to overfitting when training a LoRA may be working well for a DoRA)
> 2. **DoRA quality superior to LoRA especially in lower ranks** the difference in quality of DoRA of rank 8 and LoRA of rank 8 appears to be more significant than when training ranks of 32 or 64 for example.
> This is also aligned with some of the quantitative analysis shown in the paper.

**Usage**
1. To use DoRA you need to upgrade the installation of `peft`:
```bash
pip install -U peft
```
2. Enable DoRA training by adding this flag
```bash
--use_dora
```
**Inference**
The inference is the same as if you train a regular LoRA 🤗

## Format compatibility

You can pass `--output_kohya_format` to additionally generate a state dictionary which should be compatible with other platforms and tools such as Automatic 1111, Comfy, Kohya, etc. The `output_dir` will contain a file named "pytorch_lora_weights_kohya.safetensors".