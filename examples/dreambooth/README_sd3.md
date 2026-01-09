# DreamBooth training example for Stable Diffusion 3 (SD3)

[DreamBooth](https://huggingface.co/papers/2208.12242) is a method to personalize text2image models like stable diffusion given just a few (3~5) images of a subject.

The `train_dreambooth_sd3.py` script shows how to implement the training procedure and adapt it for [Stable Diffusion 3](https://huggingface.co/papers/2403.03206). We also provide a LoRA implementation in the `train_dreambooth_lora_sd3.py` script.

> [!NOTE]
> As the model is gated, before using it with diffusers you first need to go to the [Stable Diffusion 3 Medium Hugging Face page](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers), fill in the form and accept the gate. Once you are in, you need to log in so that your system knows you’ve accepted the gate. Use the command below to log in:

```bash
hf auth login
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
pip install -r requirements_sd3.txt
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
export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-sd3"

accelerate launch train_dreambooth_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
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

> [!NOTE]
> If you want to train using long prompts with the T5 text encoder, you can use `--max_sequence_length` to set the token limit. The default is 77, but it can be increased to as high as 512. Note that this will use more resources and may slow down the training in some cases.

> [!TIP]
> You can pass `--use_8bit_adam` to reduce the memory requirements of training. Make sure to install `bitsandbytes` if you want to do so.

## LoRA + DreamBooth

[LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) is a popular parameter-efficient fine-tuning technique that allows you to achieve full-finetuning like performance but with a fraction of learnable parameters.

Note also that we use PEFT library as backend for LoRA training, make sure to have `peft>=0.6.0` installed in your environment.

To perform DreamBooth with LoRA, run:

```bash
export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-sd3-lora"

accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=4e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub
```

### Targeting Specific Blocks & Layers
As image generation models get bigger & more powerful, more fine-tuners come to find that training only part of the 
transformer blocks (sometimes as little as two) can be enough to get great results. 
In some cases, it can be even better to maintain some of the blocks/layers frozen.

For **SD3.5-Large** specifically, you may find this information useful (taken from: [Stable Diffusion 3.5 Large Fine-tuning Tutorial](https://stabilityai.notion.site/Stable-Diffusion-3-5-Large-Fine-tuning-Tutorial-11a61cdcd1968027a15bdbd7c40be8c6#12461cdcd19680788a23c650dab26b93):
> [!NOTE]
> A commonly believed heuristic that we verified once again during the construction of the SD3.5 family of models is that later/higher layers (i.e. `30 - 37`)* impact tertiary details more heavily. Conversely, earlier layers (i.e. `12 - 24` )* influence the overall composition/primary form more. 
> So, freezing other layers/targeting specific layers is a viable approach.
> `*`These suggested layers are speculative and not 100% guaranteed. The tips here are more or less a general idea for next steps.
> **Photorealism**
> In preliminary testing, we observed that freezing the last few layers of the architecture significantly improved model training when using a photorealistic dataset, preventing detail degradation introduced by small dataset from happening.
> **Anatomy preservation**
> To dampen any possible degradation of anatomy, training only the attention layers and **not** the adaptive linear layers could help. For reference, below is one of the transformer blocks.


We've added `--lora_layers` and `--lora_blocks` to make LoRA training modules configurable. 
- with `--lora_blocks` you can specify the block numbers for training. E.g. passing - 
```diff
--lora_blocks "12,13,14,15,16,17,18,19,20,21,22,23,24,30,31,32,33,34,35,36,37"
```
will trigger LoRA training of transformer blocks 12-24 and 30-37. By default, all blocks are trained. 
- with `--lora_layers` you can specify the types of layers you wish to train. 
By default, the trained layers are -  
`attn.add_k_proj,attn.add_q_proj,attn.add_v_proj,attn.to_add_out,attn.to_k,attn.to_out.0,attn.to_q,attn.to_v`
If you wish to have a leaner LoRA / train more blocks over layers you could pass - 
```diff
+ --lora_layers attn.to_k,attn.to_q,attn.to_v,attn.to_out.0
```
This will reduce LoRA size by roughly 50% for the same rank compared to the default. 
However, if you're after compact LoRAs, it's our impression that maintaining the default setting for `--lora_layers` and
freezing some of the early & blocks is usually better. 


### Text Encoder Training
Alongside the transformer, LoRA fine-tuning of the CLIP text encoders is now also supported.
To do so, just specify `--train_text_encoder` while launching training. Please keep the following points in mind:

> [!NOTE]
> SD3 has three text encoders (CLIP L/14, OpenCLIP bigG/14, and T5-v1.1-XXL).
By enabling `--train_text_encoder`, LoRA fine-tuning of both **CLIP encoders** is performed. At the moment, T5 fine-tuning is not supported and weights remain frozen when text encoder training is enabled.

To perform DreamBooth LoRA with text-encoder training, run:
```bash
export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export OUTPUT_DIR="trained-sd3-lora"

accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --dataset_name="Norod78/Yarn-art-style" \
  --instance_prompt="a photo of TOK yarn art dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --train_text_encoder\
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy"\
  --learning_rate=1.0 \
  --text_encoder_lr=1.0 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1500 \
  --rank=32 \
  --seed="0" \
  --push_to_hub
```

## Other notes

1. We default to the "logit_normal" weighting scheme for the loss following the SD3 paper. Thanks to @bghira for helping us discover that for other weighting schemes supported from the training script, training may incur numerical instabilities.
2. Thanks to `bghira`, `JinxuXiang`, and `bendanzzc` for helping us discover a bug in how VAE encoding was being done previously. This has been fixed in [#8917](https://github.com/huggingface/diffusers/pull/8917).
3. Additionally, we now have the option to control if we want to apply preconditioning to the model outputs via a `--precondition_outputs` CLI arg. It affects how the model `target` is calculated as well.