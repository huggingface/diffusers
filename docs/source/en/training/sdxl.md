<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Stable Diffusion XL

> [!WARNING]
> This script is experimental, and it's easy to overfit and run into issues like catastrophic forgetting. Try exploring different hyperparameters to get the best results on your dataset.

[Stable Diffusion XL (SDXL)](https://hf.co/papers/2307.01952) is a larger and more powerful iteration of the Stable Diffusion model, capable of producing higher resolution images.

SDXL's UNet is 3x larger and the model adds a second text encoder to the architecture. Depending on the hardware available to you, this can be very computationally intensive and it may not run on a consumer GPU like a Tesla T4. To help fit this larger model into memory and to speedup training, try enabling `gradient_checkpointing`, `mixed_precision`, and `gradient_accumulation_steps`. You can reduce your memory-usage even more by enabling memory-efficient attention with [xFormers](../optimization/xformers) and using [bitsandbytes'](https://github.com/TimDettmers/bitsandbytes) 8-bit optimizer.

This guide will explore the [train_text_to_image_sdxl.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py) training script to help you become more familiar with it, and how you can adapt it for your own use-case.

Before running the script, make sure you install the library from source:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then navigate to the example folder containing the training script and install the required dependencies for the script you're using:

```bash
cd examples/text_to_image
pip install -r requirements_sdxl.txt
```

> [!TIP]
> 🤗 Accelerate is a library for helping you train on multiple GPUs/TPUs or with mixed-precision. It'll automatically configure your training setup based on your hardware and environment. Take a look at the 🤗 Accelerate [Quick tour](https://huggingface.co/docs/accelerate/quicktour) to learn more.

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

## Script parameters

> [!TIP]
> The following sections highlight parts of the training script that are important for understanding how to modify it, but it doesn't cover every aspect of the script in detail. If you're interested in learning more, feel free to read through the [script](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py) and let us know if you have any questions or concerns.

The training script provides many parameters to help you customize your training run. All of the parameters and their descriptions are found in the [`parse_args()`](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L129) function. This function provides default values for each parameter, such as the training batch size and learning rate, but you can also set your own values in the training command if you'd like.

For example, to speedup training with mixed precision using the bf16 format, add the `--mixed_precision` parameter to the training command:

```bash
accelerate launch train_text_to_image_sdxl.py \
  --mixed_precision="bf16"
```

Most of the parameters are identical to the parameters in the [Text-to-image](text2image#script-parameters) training guide, so you'll focus on the parameters that are relevant to training SDXL in this guide.

- `--pretrained_vae_model_name_or_path`: path to a pretrained VAE; the SDXL VAE is known to suffer from numerical instability, so this parameter allows you to specify a better [VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
- `--proportion_empty_prompts`: the proportion of image prompts to replace with empty strings
- `--timestep_bias_strategy`: where (earlier vs. later) in the timestep to apply a bias, which can encourage the model to either learn low or high frequency details
- `--timestep_bias_multiplier`: the weight of the bias to apply to the timestep
- `--timestep_bias_begin`: the timestep to begin applying the bias
- `--timestep_bias_end`: the timestep to end applying the bias
- `--timestep_bias_portion`: the proportion of timesteps to apply the bias to

### Min-SNR weighting

The [Min-SNR](https://huggingface.co/papers/2303.09556) weighting strategy can help with training by rebalancing the loss to achieve faster convergence. The training script supports predicting either `epsilon` (noise) or `v_prediction`, but Min-SNR is compatible with both prediction types. This weighting strategy is only supported by PyTorch.

Add the `--snr_gamma` parameter and set it to the recommended value of 5.0:

```bash
accelerate launch train_text_to_image_sdxl.py \
  --snr_gamma=5.0
```

## Training script

The training script is also similar to the [Text-to-image](text2image#training-script) training guide, but it's been modified to support SDXL training. This guide will focus on the code that is unique to the SDXL training script.

It starts by creating functions to [tokenize the prompts](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L478) to calculate the prompt embeddings, and to compute the image embeddings with the [VAE](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L519). Next, you'll a function to [generate the timesteps weights](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L531) depending on the number of timesteps and the timestep bias strategy to apply.

Within the [`main()`](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L572) function, in addition to loading a tokenizer, the script loads a second tokenizer and text encoder because the SDXL architecture uses two of each:

```py
tokenizer_one = AutoTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False
)
tokenizer_two = AutoTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False
)

text_encoder_cls_one = import_model_class_from_model_name_or_path(
    args.pretrained_model_name_or_path, args.revision
)
text_encoder_cls_two = import_model_class_from_model_name_or_path(
    args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
)
```

The [prompt and image embeddings](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L857) are computed first and kept in memory, which isn't typically an issue for a smaller dataset, but for larger datasets it can lead to memory problems. If this is the case, you should save the pre-computed embeddings to disk separately and load them into memory during the training process (see this [PR](https://github.com/huggingface/diffusers/pull/4505) for more discussion about this topic).

```py
text_encoders = [text_encoder_one, text_encoder_two]
tokenizers = [tokenizer_one, tokenizer_two]
compute_embeddings_fn = functools.partial(
    encode_prompt,
    text_encoders=text_encoders,
    tokenizers=tokenizers,
    proportion_empty_prompts=args.proportion_empty_prompts,
    caption_column=args.caption_column,
)

train_dataset = train_dataset.map(compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint)
train_dataset = train_dataset.map(
    compute_vae_encodings_fn,
    batched=True,
    batch_size=args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
    new_fingerprint=new_fingerprint_for_vae,
)
```

After calculating the embeddings, the text encoder, VAE, and tokenizer are deleted to free up some memory:

```py
del text_encoders, tokenizers, vae
gc.collect()
torch.cuda.empty_cache()
```

Finally, the [training loop](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L943) takes care of the rest. If you chose to apply a timestep bias strategy, you'll see the timestep weights are calculated and added as noise:

```py
weights = generate_timestep_weights(args, noise_scheduler.config.num_train_timesteps).to(
        model_input.device
    )
    timesteps = torch.multinomial(weights, bsz, replacement=True).long()

noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
```

If you want to learn more about how the training loop works, check out the [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline) tutorial which breaks down the basic pattern of the denoising process.

## Launch the script

Once you’ve made all your changes or you’re okay with the default configuration, you’re ready to launch the training script! 🚀

Let’s train on the [Naruto BLIP captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) dataset to generate your own Naruto characters. Set the environment variables `MODEL_NAME` and `DATASET_NAME` to the model and the dataset (either from the Hub or a local path). You should also specify a VAE other than the SDXL VAE (either from the Hub or a local path) with `VAE_NAME` to avoid numerical instabilities.

> [!TIP]
> To monitor training progress with Weights & Biases, add the `--report_to=wandb` parameter to the training command. You’ll also need to add the `--validation_prompt` and `--validation_epochs` to the training command to keep track of results. This can be really useful for debugging the model and viewing intermediate results.

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --enable_xformers_memory_efficient_attention \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=10000 \
  --use_8bit_adam \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --validation_prompt="a cute Sundar Pichai creature" \
  --validation_epochs 5 \
  --checkpointing_steps=5000 \
  --output_dir="sdxl-naruto-model" \
  --push_to_hub
```

After you've finished training, you can use your newly trained SDXL model for inference!

<hfoptions id="inference">
<hfoption id="PyTorch">

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("path/to/your/model", torch_dtype=torch.float16).to("cuda")

prompt = "A naruto with green eyes and red legs."
image = pipeline(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("naruto.png")
```

</hfoption>
<hfoption id="PyTorch XLA">

[PyTorch XLA](https://pytorch.org/xla) allows you to run PyTorch on XLA devices such as TPUs, which can be faster. The initial warmup step takes longer because the model needs to be compiled and optimized. However, subsequent calls to the pipeline on an input **with the same length** as the original prompt are much faster because it can reuse the optimized graph.

```py
from diffusers import DiffusionPipeline
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0").to(device)

prompt = "A naruto with green eyes and red legs."
start = time()
image = pipeline(prompt, num_inference_steps=inference_steps).images[0]
print(f'Compilation time is {time()-start} sec')
image.save("naruto.png")

start = time()
image = pipeline(prompt, num_inference_steps=inference_steps).images[0]
print(f'Inference time is {time()-start} sec after compilation')
```

</hfoption>
</hfoptions>

## Next steps

Congratulations on training a SDXL model! To learn more about how to use your new model, the following guides may be helpful:

- Read the [Stable Diffusion XL](../using-diffusers/sdxl) guide to learn how to use it for a variety of different tasks (text-to-image, image-to-image, inpainting), how to use it's refiner model, and the different types of micro-conditionings.
- Check out the [DreamBooth](dreambooth) and [LoRA](lora) training guides to learn how to train a personalized SDXL model with just a few example images. These two training techniques can even be combined!