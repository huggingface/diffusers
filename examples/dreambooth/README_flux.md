# DreamBooth training example for FLUX.1 [dev]

[DreamBooth](https://arxiv.org/abs/2208.12242) is a method to personalize text2image models like stable diffusion given just a few (3~5) images of a subject.

The `train_dreambooth_flux.py` script shows how to implement the training procedure and adapt it for [FLUX.1 [dev]](https://blackforestlabs.ai/announcing-black-forest-labs/). We also provide a LoRA implementation in the `train_dreambooth_lora_flux.py` script.
> [!NOTE]
> **Memory consumption**
>
> Flux can be quite expensive to run on consumer hardware devices and as a result finetuning it comes with high memory requirements -
> a LoRA with a rank of 16 (w/ all components trained) can exceed 40GB of VRAM for training.

> For more tips & guidance on training on a resource-constrained device and general good practices please check out these great guides and trainers for FLUX: 
> 1) [`@bghira`'s guide](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX.md)
> 2) [`ostris`'s guide](https://github.com/ostris/ai-toolkit?tab=readme-ov-file#flux1-training)

> [!NOTE]
> **Gated model**
>
> As the model is gated, before using it with diffusers you first need to go to the [FLUX.1 [dev] Hugging Face page](https://huggingface.co/black-forest-labs/FLUX.1-dev), fill in the form and accept the gate. Once you are in, you need to log in so that your system knows you’ve accepted the gate. Use the command below to log in:

```bash
huggingface-cli login
```

This will also allow us to push the trained model parameters to the Hugging Face Hub platform.

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
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-flux"

accelerate launch train_dreambooth_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1. \
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

* `report_to="wandb` will ensure the training runs are tracked on Weights and Biases. To use it, be sure to install `wandb` with `pip install wandb`.
* `validation_prompt` and `validation_epochs` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.

> [!NOTE]
> If you want to train using long prompts with the T5 text encoder, you can use `--max_sequence_length` to set the token limit. The default is 77, but it can be increased to as high as 512. Note that this will use more resources and may slow down the training in some cases.

## LoRA + DreamBooth

[LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) is a popular parameter-efficient fine-tuning technique that allows you to achieve full-finetuning like performance but with a fraction of learnable parameters.

Note also that we use PEFT library as backend for LoRA training, make sure to have `peft>=0.6.0` installed in your environment.

### Prodigy Optimizer
Prodigy is an adaptive optimizer that dynamically adjusts the learning rate learned parameters based on past gradients, allowing for more efficient convergence. 
By using prodigy we can "eliminate" the need for manual learning rate tuning. read more [here](https://huggingface.co/blog/sdxl_lora_advanced_script#adaptive-optimizers).

to use prodigy, specify
```bash
--optimizer="prodigy"
```
> [!TIP]
> When using prodigy it's generally good practice to set- `--learning_rate=1.0`

To perform DreamBooth with LoRA, run:

```bash
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-flux-lora"

accelerate launch train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub
```

### Text Encoder Training

Alongside the transformer, fine-tuning of the CLIP text encoder is also supported.
To do so, just specify `--train_text_encoder` while launching training. Please keep the following points in mind:

> [!NOTE]
> This is still an experimental feature. 
> FLUX.1 has 2 text encoders (CLIP L/14 and T5-v1.1-XXL).
By enabling `--train_text_encoder`, fine-tuning of the **CLIP encoder** is performed.
> At the moment, T5 fine-tuning is not supported and weights remain frozen when text encoder training is enabled.

To perform DreamBooth LoRA with text-encoder training, run:
```bash
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export OUTPUT_DIR="trained-flux-dev-dreambooth-lora"

accelerate launch train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --train_text_encoder\
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --seed="0" \
  --push_to_hub
```

## Memory Optimizations
As mentioned, Flux Dreambooth LoRA training is very memory intensive Here are some options (some still experimental) for a more memory efficient training.
### Image Resolution
An easy way to mitigate some of the memory requirements is through `--resolution`. `--resolution` refers to the resolution for input images, all the images in the train/validation dataset are resized to this.
Note that by default, images are resized to resolution of 512, but it's good to keep in mind in case you're accustomed to training on higher resolutions. 
### Gradient Checkpointing and Accumulation
* `--gradient accumulation` refers to the number of updates steps to accumulate before performing a backward/update pass.
by passing a value > 1 you can reduce the amount of backward/update passes and hence also memory reqs.
* with `--gradient checkpointing` we can save memory by not storing all intermediate activations during the forward pass.
Instead, only a subset of these activations (the checkpoints) are stored and the rest is recomputed as needed during the backward pass. Note that this comes at the expanse of a slower backward pass.
### 8-bit-Adam Optimizer
When training with `AdamW`(doesn't apply to `prodigy`) You can pass `--use_8bit_adam` to reduce the memory requirements of training. 
Make sure to install `bitsandbytes` if you want to do so.
### Latent caching
When training w/o validation runs, we can pre-encode the training images with the vae, and then delete it to free up some memory. 
to enable `latent_caching` simply pass `--cache_latents`.
### Precision of saved LoRA layers
By default, trained transformer layers are saved in the precision dtype in which training was performed. E.g. when training in mixed precision is enabled with `--mixed_precision="bf16"`, final finetuned layers will be saved in `torch.bfloat16` as well. 
This reduces memory requirements significantly w/o a significant quality loss. Note that if you do wish to save the final layers in float32 at the expanse of more memory usage, you can do so by passing `--upcast_before_saving`.

## Other notes
Thanks to `bghira` and `ostris` for their help with reviewing & insight sharing ♥️