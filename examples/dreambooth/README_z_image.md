# DreamBooth training example for Z-Image

[DreamBooth](https://huggingface.co/papers/2208.12242) is a method to personalize image generation models given just a few (3~5) images of a subject/concept.
[LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) is a popular parameter-efficient fine-tuning technique that allows you to achieve full-finetuning like performance but with a fraction of learnable parameters.

The `train_dreambooth_lora_z_image.py` script shows how to implement the training procedure for [LoRAs](https://huggingface.co/blog/lora) and adapt it for [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image).

> [!NOTE]
> **About Z-Image**
>
> Z-Image is a high-quality text-to-image generation model from Alibaba's Tongyi Lab. It uses a DiT (Diffusion Transformer) architecture with Qwen3 as the text encoder. The model excels at generating images with accurate text rendering, especially for Chinese characters.

> [!NOTE]
> **Memory consumption**
>
> Z-Image is relatively memory efficient compared to other large-scale diffusion models. Below we provide some tips and tricks to further reduce memory consumption during training.

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
pip install -r requirements_z_image.txt
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

## Memory Optimizations

> [!NOTE] 
> Many of these techniques complement each other and can be used together to further reduce memory consumption. However some techniques may be mutually exclusive so be sure to check before launching a training run.

### CPU Offloading 
To offload parts of the model to CPU memory, you can use `--offload` flag. This will offload the VAE and text encoder to CPU memory and only move them to GPU when needed.

### Latent Caching 
Pre-encode the training images with the VAE, and then delete it to free up some memory. To enable `latent_caching` simply pass `--cache_latents`.

### QLoRA: Low Precision Training with Quantization
Perform low precision training using 8-bit or 4-bit quantization to reduce memory usage. You can use the following flags:

- **FP8 training** with `torchao`: 
Enable FP8 training by passing `--do_fp8_training`. 
> [!IMPORTANT] 
> Since we are utilizing FP8 tensor cores we need CUDA GPUs with compute capability at least 8.9 or greater. If you're looking for memory-efficient training on relatively older cards, we encourage you to check out other trainers.

- **NF4 training** with `bitsandbytes`: 
Alternatively, you can use 8-bit or 4-bit quantization with `bitsandbytes` by passing `--bnb_quantization_config_path` to enable 4-bit NF4 quantization.

### Gradient Checkpointing and Accumulation
* `--gradient_accumulation` refers to the number of updates steps to accumulate before performing a backward/update pass. By passing a value > 1 you can reduce the amount of backward/update passes and hence also memory requirements.
* With `--gradient_checkpointing` we can save memory by not storing all intermediate activations during the forward pass. Instead, only a subset of these activations (the checkpoints) are stored and the rest is recomputed as needed during the backward pass. Note that this comes at the expense of a slower backward pass.

### 8-bit-Adam Optimizer
When training with `AdamW` (doesn't apply to `prodigy`) you can pass `--use_8bit_adam` to reduce the memory requirements of training. Make sure to install `bitsandbytes` if you want to do so.

### Image Resolution
An easy way to mitigate some of the memory requirements is through `--resolution`. `--resolution` refers to the resolution for input images, all the images in the train/validation dataset are resized to this.
Note that by default, images are resized to resolution of 1024, but it's good to keep in mind in case you're training on higher resolutions.

### Precision of saved LoRA layers
By default, trained transformer layers are saved in the precision dtype in which training was performed. E.g. when training in mixed precision is enabled with `--mixed_precision="bf16"`, final finetuned layers will be saved in `torch.bfloat16` as well. 
This reduces memory requirements significantly without a significant quality loss. Note that if you do wish to save the final layers in float32 at the expense of more memory usage, you can do so by passing `--upcast_before_saving`.

## Training Examples

### Z-Image Training

To perform DreamBooth with LoRA on Z-Image, run:

```bash
export MODEL_NAME="Tongyi-MAI/Z-Image"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-z-image-lora"

accelerate launch train_dreambooth_lora_z_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --cache_latents \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=5.0 \
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

* `report_to="wandb"` will ensure the training runs are tracked on [Weights and Biases](https://wandb.ai/site). To use it, be sure to install `wandb` with `pip install wandb`. Don't forget to call `wandb login <your_api_key>` before training if you haven't done it before.
* `validation_prompt` and `validation_epochs` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.

> [!NOTE]
> If you want to train using long prompts, you can use `--max_sequence_length` to set the token limit. The default is 512. Note that this will use more resources and may slow down the training in some cases.

### Training with FP8 Quantization

For reduced memory usage with FP8 training:

```bash
export MODEL_NAME="Tongyi-MAI/Z-Image"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-z-image-lora-fp8"

accelerate launch train_dreambooth_lora_z_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --do_fp8_training \
  --gradient_checkpointing \
  --cache_latents \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=5.0 \
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

### FSDP on the transformer

By setting the accelerate configuration with FSDP, the transformer block will be wrapped automatically. E.g. set the configuration to:

```yaml
distributed_type: FSDP
fsdp_config:
  fsdp_version: 2
  fsdp_offload_params: false
  fsdp_sharding_strategy: HYBRID_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: ZImageTransformerBlock
  fsdp_forward_prefetch: true
  fsdp_sync_module_states: false
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_use_orig_params: false
  fsdp_activation_checkpointing: true
  fsdp_reshard_after_forward: true
  fsdp_cpu_ram_efficient_loading: false
```

### Prodigy Optimizer

Prodigy is an adaptive optimizer that dynamically adjusts the learning rate learned parameters based on past gradients, allowing for more efficient convergence. 
By using prodigy we can "eliminate" the need for manual learning rate tuning. Read more [here](https://huggingface.co/blog/sdxl_lora_advanced_script#adaptive-optimizers).

To use prodigy, first make sure to install the prodigyopt library: `pip install prodigyopt`, and then specify:
```bash
--optimizer="prodigy"
```

> [!TIP]
> When using prodigy it's generally good practice to set `--learning_rate=1.0`

```bash
export MODEL_NAME="Tongyi-MAI/Z-Image"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-z-image-lora-prodigy"

accelerate launch train_dreambooth_lora_z_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --cache_latents \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=5.0 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1.0 \
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

Two key LoRA hyperparameters are LoRA rank and LoRA alpha:

- `--rank`: Defines the dimension of the trainable LoRA matrices. A higher rank means more expressiveness and capacity to learn (and more parameters).
- `--lora_alpha`: A scaling factor for the LoRA's output. The LoRA update is scaled by `lora_alpha / lora_rank`.

**lora_alpha vs. rank:**

This ratio dictates the LoRA's effective strength:
- `lora_alpha == rank`: Scaling factor is 1. The LoRA is applied with its learned strength. (e.g., alpha=16, rank=16)
- `lora_alpha < rank`: Scaling factor < 1. Reduces the LoRA's impact. Useful for subtle changes or to prevent overpowering the base model. (e.g., alpha=8, rank=16)
- `lora_alpha > rank`: Scaling factor > 1. Amplifies the LoRA's impact. Allows a lower rank LoRA to have a stronger effect. (e.g., alpha=32, rank=16)

> [!TIP]
> A common starting point is to set `lora_alpha` equal to `rank`. 
> Some also set `lora_alpha` to be twice the `rank` (e.g., lora_alpha=32 for lora_rank=16) 
> to give the LoRA updates more influence without increasing parameter count. 
> If you find your LoRA is "overcooking" or learning too aggressively, consider setting `lora_alpha` to half of `rank` 
> (e.g., lora_alpha=8 for rank=16). Experimentation is often key to finding the optimal balance for your use case.

### Target Modules

When LoRA was first adapted from language models to diffusion models, it was applied to the cross-attention layers in the UNet that relate the image representations with the prompts that describe them. 
More recently, SOTA text-to-image diffusion models replaced the UNet with a diffusion Transformer (DiT). With this change, we may also want to explore applying LoRA training onto different types of layers and blocks.

To allow more flexibility and control over the targeted modules we added `--lora_layers`, in which you can specify in a comma separated string the exact modules for LoRA training. Here are some examples of target modules you can provide:

- For attention only layers: `--lora_layers="to_k,to_q,to_v,to_out.0"`
- For attention and feed-forward layers: `--lora_layers="to_k,to_q,to_v,to_out.0,ff.net.0.proj,ff.net.2"`

> [!NOTE]
> `--lora_layers` can also be used to specify which **blocks** to apply LoRA training to. To do so, simply add a block prefix to each layer in the comma separated string.

> [!NOTE]
> Keep in mind that while training more layers can improve quality and expressiveness, it also increases the size of the output LoRA weights.

### Aspect Ratio Bucketing

We've added aspect ratio bucketing support which allows training on images with different aspect ratios without cropping them to a single square resolution. This technique helps preserve the original composition of training images and can improve training efficiency.

To enable aspect ratio bucketing, pass `--aspect_ratio_buckets` argument with a semicolon-separated list of height,width pairs, such as:

```bash
--aspect_ratio_buckets="672,1568;688,1504;720,1456;752,1392;800,1328;832,1248;880,1184;944,1104;1024,1024;1104,944;1184,880;1248,832;1328,800;1392,752;1456,720;1504,688;1568,672"
```

### Bilingual Prompts

Z-Image has strong support for both Chinese and English prompts. When training with Chinese prompts, ensure your dataset captions are properly encoded in UTF-8:

```bash
--instance_prompt="一只sks狗的照片"
--validation_prompt="一只sks狗在桶里的照片"
```

> [!TIP]
> Z-Image excels at text rendering in generated images, especially for Chinese characters. If your use case involves generating images with text, consider including text-related examples in your training data.

## Inference

Once you have trained a LoRA, you can load it for inference:

```python
import torch
from diffusers import ZImagePipeline

pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Load your trained LoRA
pipe.load_lora_weights("path/to/your/trained-z-image-lora")

# Generate an image
image = pipe(
    prompt="A photo of sks dog in a bucket",
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=5.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("output.png")
```

---

Since Z-Image finetuning is still in an experimental phase, we encourage you to explore different settings and share your insights! 🤗