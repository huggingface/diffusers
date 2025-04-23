# Training Flux Fill LoRA for Inpainting

Thanks to [Sebastian-Zok](https://github.com/Sebastian-Zok/FLUX-Fill-LoRa-Training) for the origin training code.

This (experimental) example shows how to train an Inpainting LoRA with [Flux Fill](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev). FLUX has very strong in-context capabilities, making it a suitable choice for a range of tasks, including inpainting, beside classical character LoRA training.

To know more about the Flux family, refer to the following resources:

*   [Flux Docs](https://github.com/black-forest-labs/flux/blob/main/docs/structural-conditioning.md) by Black Forest Labs
*   [Diffusers Flux Docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux)

Flux Fill models are designed for inpainting and outpainting tasks. They take an image and a corresponding mask as input, where white areas in the mask indicate the regions to be filled or repainted based on the provided text prompt.

> [!NOTE]
> **Gated model**
>
> As the base model is gated, before using it with diffusers you first need to go to the [FLUX.1 [dev] Hugging Face page](https://huggingface.co/black-forest-labs/FLUX.1-dev), fill in the form and accept the gate. Once you are in, you need to log in so that your system knows you've accepted the gate. Use the command below to log in:

```bash
huggingface-cli login
```

## Installation

First, clone the diffusers repository and install it in editable mode:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then navigate to the example directory and install the specific requirements:

```bash
cd examples/research_projects/dreambooth_inpaint # Make sure you are in the correct directory if you cloned diffusers elsewhere
pip install -r requirements_flux.txt
```

Initialize an 🤗Accelerate environment:

```bash
accelerate config
```

Or for a default configuration:

```bash
accelerate config default
```

Or non-interactively (e.g., in a notebook):

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

When running `accelerate config`, specifying `torch compile` mode to `True` can lead to significant speedups. Note that this script uses the PEFT library for LoRA training, ensure you have `peft>=0.6.0` installed.

Optional Installations
You might also find the following steps useful:

Install Weights & Biases for experiment tracking:

```bash
pip install wandb
wandb login
```

Install ProdigyOpt for optimization:

```bash
pip install prodigyopt
```

Login to Huggingface Hub (if not already done) to push your model or download gated/private datasets:

```bash
huggingface-cli login
```

Load your Dataset
This example uses the `diffusers/dog-example` dataset and corresponding masks from `sebastianzok/dog-example-masks`. Download them using:

```python
from huggingface_hub import snapshot_download

# Download images
snapshot_download(
    "diffusers/dog-example",
    local_dir= "./dog", repo_type="dataset",
    ignore_patterns=".gitattributes",
)

# Download masks
snapshot_download(
    "sebastianzok/dog-example-masks",
    local_dir= "./dog_masks", repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

Disclaimer: Before proceeding, ensure you move the `dog` and `dog_masks` folders to the `examples/research_projects/dreambooth_inpaint` directory and delete their `.cache` directories if they exist (e.g., `rm -r ./dog/.cache ./dog_masks/.cache`). The mask images should have the same filenames as their corresponding original images.

Training
Set environment variables for your model, data, and output directories:

```bash
export MODEL_NAME="black-forest-labs/FLUX.1-Fill-dev"
export INSTANCE_DIR="dog" # Directory with your images
export MASK_DIR="dog_masks" # Directory with your masks
export OUTPUT_DIR="flux-fill-dog-lora" # Where to save the trained LoRA
```

Now, launch the LoRA training using accelerate:

```bash
accelerate launch train_dreambooth_lora_flux_inpaint.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-Fill-dev" \  # Note: this is the Flux Fill model address
  --instance_data_dir="dog" \
  --mask_data_dir="dog_masks" \
  --output_dir="flux-fill-dog-lora_path" \
  --mixed_precision="bf16" \
  --instance_prompt="A TOK dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="adamw" \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --checkpointing_epochs=2 \
  --num_train_epochs=10 \
  --validation_prompt="A TOK dog" \
  --validation_epochs=1 \
  --validation_image="dog/alvan-nee-9M0tSjb-cpA-unsplash.jpeg" \
  --validation_mask="dog_masks/alvan-nee-9M0tSjb-cpA-unsplash.jpeg" \
  --seed="0" \
  --repeats=15 \
  --rank=64 \
  --alpha=32
```

Note: The script uses a placeholder token `TOK` in the `instance_prompt`. Replace this with a unique identifier for your subject if training on a specific concept (Dreambooth style), or adjust the prompt as needed for general inpainting finetuning.

Known Issue: Validation epochs (`--validation_epochs`) might not function as expected, but validation occurs at each checkpointing step regardless.

Inference
Once training is complete, you can use the trained LoRA weights with the `FluxFillPipeline` for inference:

```python
import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image

# Load the base Flux Fill pipeline
pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

# Load your trained LoRA weights
lora_path = "flux-fill-dog-lora/checkpoint-500" # Or path to your final LoRA weights
pipe.load_lora_weights(lora_path)

# Prepare input image and mask
# Use the same validation images or provide new ones
image = load_image("./dog/alvan-nee-9M0tSjb-cpA-unsplash.jpeg")
mask = load_image("./dog_masks/alvan-nee-9M0tSjb-cpA-unsplash.jpeg")

# Define the prompt
prompt = "A TOK dog wearing sunglasses" # Use the same instance prompt token if applicable

# Run inference
gen_images = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask,
    height=512, # Or the resolution used during training/inference
    width=512,  # Or the resolution used during training/inference
    guidance_scale=5.0, # Adjust as needed
    num_inference_steps=50, # Adjust as needed
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

gen_images.save("output_inpainted.png")

# Remember to unload LoRA weights if you want to use the base pipeline afterwards
# pipe.unload_lora_weights()
```
