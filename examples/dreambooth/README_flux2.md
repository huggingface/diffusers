# DreamBooth training example for FLUX.2 [dev]

[DreamBooth](https://huggingface.co/papers/2208.12242) is a method to personalize image generation models given just a few (3~5) images of a subject/concept.

The `train_dreambooth_lora_flux2.py` script shows how to implement the training procedure for [LoRAs](https://huggingface.co/blog/lora) and adapt it for [FLUX.2 [dev]](https://github.com/black-forest-labs/flux2).

> [!NOTE]
> **Memory consumption**
>
> Flux can be quite expensive to run on consumer hardware devices and as a result finetuning it comes with high memory requirements -
> a LoRA with a rank of 16 can exceed XXGB of VRAM for training. below we provide some tips and tricks to reduce memory consumption during training.

> For more tips & guidance on training on a resource-constrained device and general good practices please check out these great guides and trainers for FLUX: 
> 1) [`@bghira`'s guide](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX2.md)
> 2) [`ostris`'s guide](https://github.com/ostris/ai-toolkit?tab=readme-ov-file#flux2-training)

> [!NOTE]
> **Gated model**
>
> As the model is gated, before using it with diffusers you first need to go to the [FLUX.2 [dev] Hugging Face page](https://huggingface.co/black-forest-labs/FLUX.2-dev), fill in the form and accept the gate. Once you are in, you need to log in so that your system knows you’ve accepted the gate. Use the command below to log in:

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

As mentioned, Flux2 LoRA training is *very* memory intensive. Here are memory optimizations we can use (some still experimental) for a more memory efficient training:

## Memory Optimizations
> [!NOTE] many of these techniques complement each other and can be used together to further reduce memory consumption. 
> However some techniques may be mutually exclusive so be sure to check before launching a training run.
### Remote Text Encoder 
Flux.2 uses  Mistral Small 3.1 as text encoder which is quite large and can take up a lot of memory. To mitigate this, we can use the `--remote_text_encoder` flag to enable remote computation of the prompt embeddings using the HuggingFace Inference API. 
This way, the text encoder model is not loaded into memory during training.
> [!NOTE] 
> to enable remote text encoding you must either be logged in to your HuggingFace account (`hf auth login`) OR pass a token with `--hub_token`.
### FSDP Text Encoder 
Flux.2 uses  Mistral Small 3.1 as text encoder which is quite large and can take up a lot of memory. To mitigate this, we can use the `--fsdp_text_encoder` flag to enable distributed computation of the prompt embeddings. 
This way, it distributes the memory cost across multiple nodes.
### CPU Offloading 
To offload parts of the model to CPU memory, you can use `--offload` flag. This will offload the vae and text encoder to CPU memory and only move them to GPU when needed.
### Latent Caching 
Pre-encode the training images with the vae, and then delete it to free up some memory. To enable `latent_caching` simply pass `--cache_latents`.
### QLoRA: Low Precision Training with Quantization
Perform low precision training using 8-bit or 4-bit quantization to reduce memory usage. You can use the following flags:
- **FP8 training** with `torchao`: 
enable FP8 training by passing `--do_fp8_training`. 
> [!IMPORTANT] Since we are utilizing FP8 tensor cores we need CUDA GPUs with compute capability at least 8.9 or greater. 
> If you're looking for memory-efficient training on relatively older cards, we encourage you to check out other trainers like SimpleTuner, ai-toolkit, etc.
- **NF4 training** with `bitsandbytes`: 
Alternatively, you can use 8-bit or 4-bit quantization with `bitsandbytes` by passing:
`--bnb_quantization_config_path` to enable 4-bit NF4 quantization.
### Gradient Checkpointing and Accumulation
* `--gradient accumulation` refers to the number of updates steps to accumulate before performing a backward/update pass.
by passing a value > 1 you can reduce the amount of backward/update passes and hence also memory reqs.
* with `--gradient checkpointing` we can save memory by not storing all intermediate activations during the forward pass.
Instead, only a subset of these activations (the checkpoints) are stored and the rest is recomputed as needed during the backward pass. Note that this comes at the expanse of a slower backward pass.
### 8-bit-Adam Optimizer
When training with `AdamW`(doesn't apply to `prodigy`) You can pass `--use_8bit_adam` to reduce the memory requirements of training. 
Make sure to install `bitsandbytes` if you want to do so.
### Image Resolution
An easy way to mitigate some of the memory requirements is through `--resolution`. `--resolution` refers to the resolution for input images, all the images in the train/validation dataset are resized to this.
Note that by default, images are resized to resolution of 512, but it's good to keep in mind in case you're accustomed to training on higher resolutions.
### Precision of saved LoRA layers
By default, trained transformer layers are saved in the precision dtype in which training was performed. E.g. when training in mixed precision is enabled with `--mixed_precision="bf16"`, final finetuned layers will be saved in `torch.bfloat16` as well. 
This reduces memory requirements significantly w/o a significant quality loss. Note that if you do wish to save the final layers in float32 at the expanse of more memory usage, you can do so by passing `--upcast_before_saving`.


```bash
export MODEL_NAME="black-forest-labs/FLUX.2-dev"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-flux2"

accelerate launch train_dreambooth_lora_flux2.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --do_fp8_training \
  --gradient_checkpointing \
  --remote_text_encoder \
  --cache_latents \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --use_8bit_adam \
  --gradient_accumulation_steps=4 \
  --optimizer="adamW" \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
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

### FSDP on the transformer
By setting the accelerate configuration with FSDP, the transformer block will be wrapped automatically. E.g. set the configuration to:

```shell
distributed_type: FSDP
fsdp_config:
  fsdp_version: 2
  fsdp_offload_params: false
  fsdp_sharding_strategy: HYBRID_SHARD
  fsdp_auto_wrap_policy: TRANSFOMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: Flux2TransformerBlock, Flux2SingleTransformerBlock
  fsdp_forward_prefetch: true
  fsdp_sync_module_states: false
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_use_orig_params: false
  fsdp_activation_checkpointing: true
  fsdp_reshard_after_forward: true
  fsdp_cpu_ram_efficient_loading: false
```

## LoRA + DreamBooth

[LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) is a popular parameter-efficient fine-tuning technique that allows you to achieve full-finetuning like performance but with a fraction of learnable parameters.

Note also that we use PEFT library as backend for LoRA training, make sure to have `peft>=0.6.0` installed in your environment.

### Prodigy Optimizer
Prodigy is an adaptive optimizer that dynamically adjusts the learning rate learned parameters based on past gradients, allowing for more efficient convergence. 
By using prodigy we can "eliminate" the need for manual learning rate tuning. read more [here](https://huggingface.co/blog/sdxl_lora_advanced_script#adaptive-optimizers).

to use prodigy, first make sure to install the prodigyopt library: `pip install prodigyopt`, and then specify -
```bash
--optimizer="prodigy"
```
> [!TIP]
> When using prodigy it's generally good practice to set- `--learning_rate=1.0`

To perform DreamBooth with LoRA, run:

```bash
export MODEL_NAME="black-forest-labs/FLUX.2-dev"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-flux2-lora"

accelerate launch train_dreambooth_lora_flux2.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --do_fp8_training \
  --gradient_checkpointing \
  --remote_text_encoder \
  --cache_latents \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub
```

### LoRA Rank and Alpha
Two key LoRA hyperparameters are LoRA rank and LoRA alpha. 
- `--rank`: Defines the dimension of the trainable LoRA matrices. A higher rank means more expressiveness and capacity to learn (and more parameters).
- `--lora_alpha`: A scaling factor for the LoRA's output. The LoRA update is scaled by lora_alpha / lora_rank.
- lora_alpha vs. rank:
This ratio dictates the LoRA's effective strength:
lora_alpha == rank: Scaling factor is 1. The LoRA is applied with its learned strength. (e.g., alpha=16, rank=16)
lora_alpha < rank: Scaling factor < 1. Reduces the LoRA's impact. Useful for subtle changes or to prevent overpowering the base model. (e.g., alpha=8, rank=16)
lora_alpha > rank: Scaling factor > 1. Amplifies the LoRA's impact. Allows a lower rank LoRA to have a stronger effect. (e.g., alpha=32, rank=16)

> [!TIP]
> A common starting point is to set `lora_alpha` equal to `rank`. 
> Some also set `lora_alpha` to be twice the `rank` (e.g., lora_alpha=32 for lora_rank=16) 
> to give the LoRA updates more influence without increasing parameter count. 
> If you find your LoRA is "overcooking" or learning too aggressively, consider setting `lora_alpha` to half of `rank` 
> (e.g., lora_alpha=8 for rank=16). Experimentation is often key to finding the optimal balance for your use case.

### Target Modules
When LoRA was first adapted from language models to diffusion models, it was applied to the cross-attention layers in the Unet that relate the image representations with the prompts that describe them. 
More recently, SOTA text-to-image diffusion models replaced the Unet with a diffusion Transformer(DiT). With this change, we may also want to explore 
applying LoRA training onto different types of layers and blocks. To allow more flexibility and control over the targeted modules we added `--lora_layers`- in which you can specify in a comma separated string
the exact modules for LoRA training. Here are some examples of target modules you can provide: 
- for attention only layers: `--lora_layers="attn.to_k,attn.to_q,attn.to_v,attn.to_out.0"`
- to train the same modules as in the fal trainer: `--lora_layers="attn.to_k,attn.to_q,attn.to_v,attn.to_out.0,attn.add_k_proj,attn.add_q_proj,attn.add_v_proj,attn.to_add_out,ff.net.0.proj,ff.net.2,ff_context.net.0.proj,ff_context.net.2"`
- to train the same modules as in ostris ai-toolkit / replicate trainer: `--lora_blocks="attn.to_k,attn.to_q,attn.to_v,attn.to_out.0,attn.add_k_proj,attn.add_q_proj,attn.add_v_proj,attn.to_add_out,ff.net.0.proj,ff.net.2,ff_context.net.0.proj,ff_context.net.2,norm1_context.linear, norm1.linear,norm.linear,proj_mlp,proj_out"`
> [!NOTE]
> `--lora_layers` can also be used to specify which **blocks** to apply LoRA training to. To do so, simply add a block prefix to each layer in the comma separated string:
> **single DiT blocks**: to target the ith single transformer block, add the prefix `single_transformer_blocks.i`, e.g. - `single_transformer_blocks.i.attn.to_k`
> **MMDiT blocks**: to target the ith MMDiT block, add the prefix `transformer_blocks.i`, e.g. - `transformer_blocks.i.attn.to_k` 
> [!NOTE]
> keep in mind that while training more layers can improve quality and expressiveness, it also increases the size of the output LoRA weights.



## Training Image-to-Image

Flux.2 lets us perform image editing as well as image generation. We provide a simple script for image-to-image(I2I) LoRA fine-tuning in [train_dreambooth_lora_flux2_img2img.py](./train_dreambooth_lora_flux2_img2img.py) for both T2I and I2I. The optimizations discussed above apply this script, too.

**important**

**Important**
To make sure you can successfully run the latest version of the image-to-image example script, we highly recommend installing from source, specifically from the commit mentioned below. To do this, execute the following steps in a new virtual environment:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .

To start, you must have a dataset containing triplets:

* Condition image - the input image to be transformed.
* Target image - the desired output image after transformation.
* Instruction - a text prompt describing the transformation from the condition image to the target image.

[kontext-community/relighting](https://huggingface.co/datasets/kontext-community/relighting) is a good example of such a dataset. If you are using such a dataset, you can use the command below to launch training:

```bash
accelerate launch train_dreambooth_lora_flux2_img2img.py \
  --pretrained_model_name_or_path=black-forest-labs/FLUX.2-dev  \
  --output_dir="flux2-i2i" \
  --dataset_name="kontext-community/relighting" \
  --image_column="output" --cond_image_column="file_name" --caption_column="instruction" \
  --do_fp8_training \
  --gradient_checkpointing \
  --remote_text_encoder \
  --cache_latents \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --optimizer="adamw" \
  --use_8bit_adam \
  --cache_latents \
  --learning_rate=1e-4 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=200 \
  --max_train_steps=1000 \
  --rank=16\
  --seed="0" 
```

More generally, when performing I2I fine-tuning, we expect you to:

* Have a dataset `kontext-community/relighting`
* Supply `image_column`, `cond_image_column`, and `caption_column` values when launching training

### Misc notes

* By default, we use `mode` as the value of `--vae_encode_mode` argument. This is because Kontext uses `mode()` of the distribution predicted by the VAE instead of sampling from it.
### Aspect Ratio Bucketing
we've added aspect ratio bucketing support which allows training on images with different aspect ratios without cropping them to a single square resolution. This technique helps preserve the original composition of training images and can improve training efficiency.

To enable aspect ratio bucketing, pass `--aspect_ratio_buckets` argument with a semicolon-separated list of height,width pairs, such as:

`--aspect_ratio_buckets="672,1568;688,1504;720,1456;752,1392;800,1328;832,1248;880,1184;944,1104;1024,1024;1104,944;1184,880;1248,832;1328,800;1392,752;1456,720;1504,688;1568,672"
`
Since Flux.2 finetuning is still an experimental phase, we encourage you to explore different settings and share your insights! 🤗
