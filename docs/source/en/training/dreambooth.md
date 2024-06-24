<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DreamBooth

[DreamBooth](https://huggingface.co/papers/2208.12242) is a training technique that updates the entire diffusion model by training on just a few images of a subject or style. It works by associating a special word in the prompt with the example images.

If you're training on a GPU with limited vRAM, you should try enabling the `gradient_checkpointing` and `mixed_precision` parameters in the training command. You can also reduce your memory footprint by using memory-efficient attention with [xFormers](../optimization/xformers). JAX/Flax training is also supported for efficient training on TPUs and GPUs, but it doesn't support gradient checkpointing or xFormers. You should have a GPU with >30GB of memory if you want to train faster with Flax.

This guide will explore the [train_dreambooth.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py) script to help you become more familiar with it, and how you can adapt it for your own use-case.

Before running the script, make sure you install the library from source:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Navigate to the example folder with the training script and install the required dependencies for the script you're using:

<hfoptions id="installation">
<hfoption id="PyTorch">

```bash
cd examples/dreambooth
pip install -r requirements.txt
```

</hfoption>
<hfoption id="Flax">

```bash
cd examples/dreambooth
pip install -r requirements_flax.txt
```

</hfoption>
</hfoptions>

<Tip>

🤗 Accelerate is a library for helping you train on multiple GPUs/TPUs or with mixed-precision. It'll automatically configure your training setup based on your hardware and environment. Take a look at the 🤗 Accelerate [Quick tour](https://huggingface.co/docs/accelerate/quicktour) to learn more.

</Tip>

Initialize an 🤗 Accelerate environment:

```bash
accelerate config
```

To setup a default 🤗 Accelerate environment without choosing any configurations:

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell, like a notebook, you can use:

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

Lastly, if you want to train a model on your own dataset, take a look at the [Create a dataset for training](create_dataset) guide to learn how to create a dataset that works with the training script.

<Tip>

The following sections highlight parts of the training script that are important for understanding how to modify it, but it doesn't cover every aspect of the script in detail. If you're interested in learning more, feel free to read through the [script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py) and let us know if you have any questions or concerns.

</Tip>

## Script parameters

<Tip warning={true}>

DreamBooth is very sensitive to training hyperparameters, and it is easy to overfit. Read the [Training Stable Diffusion with Dreambooth using 🧨 Diffusers](https://huggingface.co/blog/dreambooth) blog post for recommended settings for different subjects to help you choose the appropriate hyperparameters.

</Tip>

The training script offers many parameters for customizing your training run. All of the parameters and their descriptions are found in the [`parse_args()`](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L228) function. The parameters are set with default values that should work pretty well out-of-the-box, but you can also set your own values in the training command if you'd like.

For example, to train in the bf16 format:

```bash
accelerate launch train_dreambooth.py \
    --mixed_precision="bf16"
```

Some basic and important parameters to know and specify are:

- `--pretrained_model_name_or_path`: the name of the model on the Hub or a local path to the pretrained model
- `--instance_data_dir`: path to a folder containing the training dataset (example images)
- `--instance_prompt`: the text prompt that contains the special word for the example images
- `--train_text_encoder`: whether to also train the text encoder
- `--output_dir`: where to save the trained model
- `--push_to_hub`: whether to push the trained model to the Hub
- `--checkpointing_steps`: frequency of saving a checkpoint as the model trains; this is useful if for some reason training is interrupted, you can continue training from that checkpoint by adding `--resume_from_checkpoint` to your training command

### Min-SNR weighting

The [Min-SNR](https://huggingface.co/papers/2303.09556) weighting strategy can help with training by rebalancing the loss to achieve faster convergence. The training script supports predicting `epsilon` (noise) or `v_prediction`, but Min-SNR is compatible with both prediction types. This weighting strategy is only supported by PyTorch and is unavailable in the Flax training script.

Add the `--snr_gamma` parameter and set it to the recommended value of 5.0:

```bash
accelerate launch train_dreambooth.py \
  --snr_gamma=5.0
```

### Prior preservation loss

Prior preservation loss is a method that uses a model's own generated samples to help it learn how to generate more diverse images. Because these generated sample images belong to the same class as the images you provided, they help the model retain what it has learned about the class and how it can use what it already knows about the class to make new compositions.

- `--with_prior_preservation`: whether to use prior preservation loss
- `--prior_loss_weight`: controls the influence of the prior preservation loss on the model
- `--class_data_dir`: path to a folder containing the generated class sample images
- `--class_prompt`: the text prompt describing the class of the generated sample images

```bash
accelerate launch train_dreambooth.py \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --class_data_dir="path/to/class/images" \
  --class_prompt="text prompt describing class"
```

### Train text encoder

To improve the quality of the generated outputs, you can also train the text encoder in addition to the UNet. This requires additional memory and you'll need a GPU with at least 24GB of vRAM. If you have the necessary hardware, then training the text encoder produces better results, especially when generating images of faces. Enable this option by:

```bash
accelerate launch train_dreambooth.py \
  --train_text_encoder
```

## Training script

DreamBooth comes with its own dataset classes:

- [`DreamBoothDataset`](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L604): preprocesses the images and class images, and tokenizes the prompts for training
- [`PromptDataset`](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L738): generates the prompt embeddings to generate the class images

If you enabled [prior preservation loss](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L842), the class images are generated here:

```py
sample_dataset = PromptDataset(args.class_prompt, num_new_images)
sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

sample_dataloader = accelerator.prepare(sample_dataloader)
pipeline.to(accelerator.device)

for example in tqdm(
    sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
):
    images = pipeline(example["prompt"]).images
```

Next is the [`main()`](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L799) function which handles setting up the dataset for training and the training loop itself. The script loads the [tokenizer](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L898), [scheduler and models](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L912C1-L912C1):

```py
# Load the tokenizer
if args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
elif args.pretrained_model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

# Load scheduler and models
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
text_encoder = text_encoder_cls.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
)

if model_has_vae(args):
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
else:
    vae = None

unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
)
```

Then, it's time to [create the training dataset](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L1073) and DataLoader from `DreamBoothDataset`:

```py
train_dataset = DreamBoothDataset(
    instance_data_root=args.instance_data_dir,
    instance_prompt=args.instance_prompt,
    class_data_root=args.class_data_dir if args.with_prior_preservation else None,
    class_prompt=args.class_prompt,
    class_num=args.num_class_images,
    tokenizer=tokenizer,
    size=args.resolution,
    center_crop=args.center_crop,
    encoder_hidden_states=pre_computed_encoder_hidden_states,
    class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
    tokenizer_max_length=args.tokenizer_max_length,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.train_batch_size,
    shuffle=True,
    collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
    num_workers=args.dataloader_num_workers,
)
```

Lastly, the [training loop](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L1151) takes care of the remaining steps such as converting images to latent space, adding noise to the input, predicting the noise residual, and calculating the loss.

If you want to learn more about how the training loop works, check out the [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline) tutorial which breaks down the basic pattern of the denoising process.

## Launch the script

You're now ready to launch the training script! 🚀

For this guide, you'll download some images of a [dog](https://huggingface.co/datasets/diffusers/dog-example) and store them in a directory. But remember, you can create and use your own dataset if you want (see the [Create a dataset for training](create_dataset) guide).

```py
from huggingface_hub import snapshot_download

local_dir = "./dog"
snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir,
    repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

Set the environment variable `MODEL_NAME` to a model id on the Hub or a path to a local model, `INSTANCE_DIR` to the path where you just downloaded the dog images to, and `OUTPUT_DIR` to where you want to save the model. You'll use `sks` as the special word to tie the training to.

If you're interested in following along with the training process, you can periodically save generated images as training progresses. Add the following parameters to the training command:

```bash
--validation_prompt="a photo of a sks dog"
--num_validation_images=4
--validation_steps=100
```

One more thing before you launch the script! Depending on the GPU you have, you may need to enable certain optimizations to train DreamBooth.

<hfoptions id="gpu-select">
<hfoption id="16GB">

On a 16GB GPU, you can use bitsandbytes 8-bit optimizer and gradient checkpointing to help you train a DreamBooth model. Install bitsandbytes:

```py
pip install bitsandbytes
```

Then, add the following parameter to your training command:

```bash
accelerate launch train_dreambooth.py \
  --gradient_checkpointing \
  --use_8bit_adam \
```

</hfoption>
<hfoption id="12GB">

On a 12GB GPU, you'll need bitsandbytes 8-bit optimizer, gradient checkpointing, xFormers, and set the gradients to `None` instead of zero to reduce your memory-usage.

```bash
accelerate launch train_dreambooth.py \
  --use_8bit_adam \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none \
```

</hfoption>
<hfoption id="8GB">

On a 8GB GPU, you'll need [DeepSpeed](https://www.deepspeed.ai/) to offload some of the tensors from the vRAM to either the CPU or NVME to allow training with less GPU memory.

Run the following command to configure your 🤗 Accelerate environment:

```bash
accelerate config
```

During configuration, confirm that you want to use DeepSpeed. Now it should be possible to train on under 8GB vRAM by combining DeepSpeed stage 2, fp16 mixed precision, and offloading the model parameters and the optimizer state to the CPU. The drawback is that this requires more system RAM (~25 GB). See the [DeepSpeed documentation](https://huggingface.co/docs/accelerate/usage_guides/deepspeed) for more configuration options.

You should also change the default Adam optimizer to DeepSpeed’s optimized version of Adam [`deepspeed.ops.adam.DeepSpeedCPUAdam`](https://deepspeed.readthedocs.io/en/latest/optimizers.html#adam-cpu) for a substantial speedup. Enabling `DeepSpeedCPUAdam` requires your system’s CUDA toolchain version to be the same as the one installed with PyTorch.

bitsandbytes 8-bit optimizers don’t seem to be compatible with DeepSpeed at the moment.

That's it! You don't need to add any additional parameters to your training command.

</hfoption>
</hfoptions>

<hfoptions id="training-inference">
<hfoption id="PyTorch">

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="path_to_saved_model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --push_to_hub
```

</hfoption>
<hfoption id="Flax">

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --max_train_steps=400 \
  --push_to_hub
```

</hfoption>
</hfoptions>

Once training is complete, you can use your newly trained model for inference!

<Tip>

Can't wait to try your model for inference before training is complete? 🤭 Make sure you have the latest version of 🤗 Accelerate installed.

```py
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch

unet = UNet2DConditionModel.from_pretrained("path/to/model/checkpoint-100/unet")

# if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
text_encoder = CLIPTextModel.from_pretrained("path/to/model/checkpoint-100/checkpoint-100/text_encoder")

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", unet=unet, text_encoder=text_encoder, dtype=torch.float16,
).to("cuda")

image = pipeline("A photo of sks dog in a bucket", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("dog-bucket.png")
```

</Tip>

<hfoptions id="training-inference">
<hfoption id="PyTorch">

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("path_to_saved_model", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
image = pipeline("A photo of sks dog in a bucket", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("dog-bucket.png")
```

</hfoption>
<hfoption id="Flax">

```py
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline

pipeline, params = FlaxStableDiffusionPipeline.from_pretrained("path-to-your-trained-model", dtype=jax.numpy.bfloat16)

prompt = "A photo of sks dog in a bucket"
prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

num_samples = jax.device_count()
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
image.save("dog-bucket.png")
```

</hfoption>
</hfoptions>

## LoRA

LoRA is a training technique for significantly reducing the number of trainable parameters. As a result, training is faster and it is easier to store the resulting weights because they are a lot smaller (~100MBs). Use the [train_dreambooth_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py) script to train with LoRA.

The LoRA training script is discussed in more detail in the [LoRA training](lora) guide.

## Stable Diffusion XL

Stable Diffusion XL (SDXL) is a powerful text-to-image model that generates high-resolution images, and it adds a second text-encoder to its architecture. Use the [train_dreambooth_lora_sdxl.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sdxl.py) script to train a SDXL model with LoRA.

The SDXL training script is discussed in more detail in the [SDXL training](sdxl) guide.

## DeepFloyd IF

DeepFloyd IF is a cascading pixel diffusion model with three stages. The first stage generates a base image and the second and third stages progressively upscales the base image into a high-resolution 1024x1024 image. Use the [train_dreambooth_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py) or [train_dreambooth.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py) scripts to train a DeepFloyd IF model with LoRA or the full model.

DeepFloyd IF uses predicted variance, but the Diffusers training scripts uses predicted error so the trained DeepFloyd IF models are switched to a fixed variance schedule. The training scripts will update the scheduler config of the fully trained model for you. However, when you load the saved LoRA weights you must also update the pipeline's scheduler config.

```py
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", use_safetensors=True)

pipe.load_lora_weights("<lora weights path>")

# Update scheduler config to fixed variance schedule
pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, variance_type="fixed_small")
```

The stage 2 model requires additional validation images to upscale. You can download and use a downsized version of the training images for this.

```py
from huggingface_hub import snapshot_download

local_dir = "./dog_downsized"
snapshot_download(
    "diffusers/dog-example-downsized",
    local_dir=local_dir,
    repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

The code samples below provide a brief overview of how to train a DeepFloyd IF model with a combination of DreamBooth and LoRA. Some important parameters to note are:

* `--resolution=64`, a much smaller resolution is required because DeepFloyd IF is a pixel diffusion model and to work on uncompressed pixels, the input images must be smaller
* `--pre_compute_text_embeddings`, compute the text embeddings ahead of time to save memory because the [`~transformers.T5Model`] can take up a lot of memory
* `--tokenizer_max_length=77`, you can use a longer default text length with T5 as the text encoder but the default model encoding procedure uses a shorter text length
* `--text_encoder_use_attention_mask`, to pass the attention mask to the text encoder

<hfoptions id="IF-DreamBooth">
<hfoption id="Stage 1 LoRA DreamBooth">

Training stage 1 of DeepFloyd IF with LoRA and DreamBooth requires ~28GB of memory.

```bash
export MODEL_NAME="DeepFloyd/IF-I-XL-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_dog_lora"

accelerate launch train_dreambooth_lora.py \
  --report_to wandb \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks dog" \
  --resolution=64 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --scale_lr \
  --max_train_steps=1200 \
  --validation_prompt="a sks dog" \
  --validation_epochs=25 \
  --checkpointing_steps=100 \
  --pre_compute_text_embeddings \
  --tokenizer_max_length=77 \
  --text_encoder_use_attention_mask
```

</hfoption>
<hfoption id="Stage 2 LoRA DreamBooth">

For stage 2 of DeepFloyd IF with LoRA and DreamBooth, pay attention to these parameters:

* `--validation_images`, the images to upscale during validation
* `--class_labels_conditioning=timesteps`, to additionally conditional the UNet as needed in stage 2
* `--learning_rate=1e-6`, a lower learning rate is used compared to stage 1
* `--resolution=256`, the expected resolution for the upscaler

```bash
export MODEL_NAME="DeepFloyd/IF-II-L-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_dog_upscale"
export VALIDATION_IMAGES="dog_downsized/image_1.png dog_downsized/image_2.png dog_downsized/image_3.png dog_downsized/image_4.png"

python train_dreambooth_lora.py \
    --report_to wandb \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --instance_prompt="a sks dog" \
    --resolution=256 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-6 \
    --max_train_steps=2000 \
    --validation_prompt="a sks dog" \
    --validation_epochs=100 \
    --checkpointing_steps=500 \
    --pre_compute_text_embeddings \
    --tokenizer_max_length=77 \
    --text_encoder_use_attention_mask \
    --validation_images $VALIDATION_IMAGES \
    --class_labels_conditioning=timesteps
```

</hfoption>
<hfoption id="Stage 1 DreamBooth">

For stage 1 of DeepFloyd IF with DreamBooth, pay attention to these parameters:

* `--skip_save_text_encoder`, to skip saving the full T5 text encoder with the finetuned model
* `--use_8bit_adam`, to use 8-bit Adam optimizer to save memory due to the size of the optimizer state when training the full model
* `--learning_rate=1e-7`, a really low learning rate should be used for full model training otherwise the model quality is degraded (you can use a higher learning rate with a larger batch size)

Training with 8-bit Adam and a batch size of 4, the full model can be trained with ~48GB of memory.

```bash
export MODEL_NAME="DeepFloyd/IF-I-XL-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_if"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=64 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-7 \
  --max_train_steps=150 \
  --validation_prompt "a photo of sks dog" \
  --validation_steps 25 \
  --text_encoder_use_attention_mask \
  --tokenizer_max_length 77 \
  --pre_compute_text_embeddings \
  --use_8bit_adam \
  --set_grads_to_none \
  --skip_save_text_encoder \
  --push_to_hub
```

</hfoption>
<hfoption id="Stage 2 DreamBooth">

For stage 2 of DeepFloyd IF with DreamBooth, pay attention to these parameters:

* `--learning_rate=5e-6`, use a lower learning rate with a smaller effective batch size
* `--resolution=256`, the expected resolution for the upscaler
* `--train_batch_size=2` and `--gradient_accumulation_steps=6`, to effectively train on images wiht faces requires larger batch sizes

```bash
export MODEL_NAME="DeepFloyd/IF-II-L-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_dog_upscale"
export VALIDATION_IMAGES="dog_downsized/image_1.png dog_downsized/image_2.png dog_downsized/image_3.png dog_downsized/image_4.png"

accelerate launch train_dreambooth.py \
  --report_to wandb \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks dog" \
  --resolution=256 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=6 \
  --learning_rate=5e-6 \
  --max_train_steps=2000 \
  --validation_prompt="a sks dog" \
  --validation_steps=150 \
  --checkpointing_steps=500 \
  --pre_compute_text_embeddings \
  --tokenizer_max_length=77 \
  --text_encoder_use_attention_mask \
  --validation_images $VALIDATION_IMAGES \
  --class_labels_conditioning timesteps \
  --push_to_hub
```

</hfoption>
</hfoptions>

### Training tips

Training the DeepFloyd IF model can be challenging, but here are some tips that we've found helpful:

- LoRA is sufficient for training the stage 1 model because the model's low resolution makes representing finer details difficult regardless.
- For common or simple objects, you don't necessarily need to finetune the upscaler. Make sure the prompt passed to the upscaler is adjusted to remove the new token from the instance prompt. For example, if your stage 1 prompt is "a sks dog" then your stage 2 prompt should be "a dog".
- For finer details like faces, fully training the stage 2 upscaler is better than training the stage 2 model with LoRA. It also helps to use lower learning rates with larger batch sizes.
- Lower learning rates should be used to train the stage 2 model.
- The [`DDPMScheduler`] works better than the DPMSolver used in the training scripts.

## Next steps

Congratulations on training your DreamBooth model! To learn more about how to use your new model, the following guide may be helpful:

- Learn how to [load a DreamBooth](../using-diffusers/loading_adapters) model for inference if you trained your model with LoRA.