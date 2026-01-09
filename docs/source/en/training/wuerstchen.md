<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Wuerstchen

The [Wuerstchen](https://hf.co/papers/2306.00637) model drastically reduces computational costs by compressing the latent space by 42x, without compromising image quality and accelerating inference. During training, Wuerstchen uses two models (VQGAN + autoencoder) to compress the latents, and then a third model (text-conditioned latent diffusion model) is conditioned on this highly compressed space to generate an image.

To fit the prior model into GPU memory and to speedup training, try enabling `gradient_accumulation_steps`, `gradient_checkpointing`, and `mixed_precision` respectively.

This guide explores the [train_text_to_image_prior.py](https://github.com/huggingface/diffusers/blob/main/examples/wuerstchen/text_to_image/train_text_to_image_prior.py) script to help you become more familiar with it, and how you can adapt it for your own use-case.

Before running the script, make sure you install the library from source:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then navigate to the example folder containing the training script and install the required dependencies for the script you're using:

```bash
cd examples/wuerstchen/text_to_image
pip install -r requirements.txt
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

> [!TIP]
> The following sections highlight parts of the training scripts that are important for understanding how to modify it, but it doesn't cover every aspect of the [script](https://github.com/huggingface/diffusers/blob/main/examples/wuerstchen/text_to_image/train_text_to_image_prior.py) in detail. If you're interested in learning more, feel free to read through the scripts and let us know if you have any questions or concerns.

## Script parameters

The training scripts provides many parameters to help you customize your training run. All of the parameters and their descriptions are found in the [`parse_args()`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L192) function. It provides default values for each parameter, such as the training batch size and learning rate, but you can also set your own values in the training command if you'd like.

For example, to speedup training with mixed precision using the fp16 format, add the `--mixed_precision` parameter to the training command:

```bash
accelerate launch train_text_to_image_prior.py \
  --mixed_precision="fp16"
```

Most of the parameters are identical to the parameters in the [Text-to-image](text2image#script-parameters) training guide, so let's dive right into the Wuerstchen training script!

## Training script

The training script is also similar to the [Text-to-image](text2image#training-script) training guide, but it's been modified to support Wuerstchen. This guide focuses on the code that is unique to the Wuerstchen training script.

The [`main()`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L441) function starts by initializing the image encoder - an [EfficientNet](https://github.com/huggingface/diffusers/blob/main/examples/wuerstchen/text_to_image/modeling_efficient_net_encoder.py) - in addition to the usual scheduler and tokenizer.

```py
with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    pretrained_checkpoint_file = hf_hub_download("dome272/wuerstchen", filename="model_v2_stage_b.pt")
    state_dict = torch.load(pretrained_checkpoint_file, map_location="cpu")
    image_encoder = EfficientNetEncoder()
    image_encoder.load_state_dict(state_dict["effnet_state_dict"])
    image_encoder.eval()
```

You'll also load the [`WuerstchenPrior`] model for optimization.

```py
prior = WuerstchenPrior.from_pretrained(args.pretrained_prior_model_name_or_path, subfolder="prior")

optimizer = optimizer_cls(
    prior.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```

Next, you'll apply some [transforms](https://github.com/huggingface/diffusers/blob/65ef7a0c5c594b4f84092e328fbdd73183613b30/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L656) to the images and [tokenize](https://github.com/huggingface/diffusers/blob/65ef7a0c5c594b4f84092e328fbdd73183613b30/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L637) the captions:

```py
def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["effnet_pixel_values"] = [effnet_transforms(image) for image in images]
    examples["text_input_ids"], examples["text_mask"] = tokenize_captions(examples)
    return examples
```

Finally, the [training loop](https://github.com/huggingface/diffusers/blob/65ef7a0c5c594b4f84092e328fbdd73183613b30/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L656) handles compressing the images to latent space with the `EfficientNetEncoder`, adding noise to the latents, and predicting the noise residual with the [`WuerstchenPrior`] model.

```py
pred_noise = prior(noisy_latents, timesteps, prompt_embeds)
```

If you want to learn more about how the training loop works, check out the [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline) tutorial which breaks down the basic pattern of the denoising process.

## Launch the script

Once you’ve made all your changes or you’re okay with the default configuration, you’re ready to launch the training script! 🚀

Set the `DATASET_NAME` environment variable to the dataset name from the Hub. This guide uses the [Naruto BLIP captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) dataset, but you can create and train on your own datasets as well (see the [Create a dataset for training](create_dataset) guide).

> [!TIP]
> To monitor training progress with Weights & Biases, add the `--report_to=wandb` parameter to the training command. You’ll also need to add the `--validation_prompt` to the training command to keep track of results. This can be really useful for debugging the model and viewing intermediate results.

```bash
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch  train_text_to_image_prior.py \
  --mixed_precision="fp16" \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --dataloader_num_workers=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_prompts="A robot naruto, 4k photo" \
  --report_to="wandb" \
  --push_to_hub \
  --output_dir="wuerstchen-prior-naruto-model"
```

Once training is complete, you can use your newly trained model for inference!

```py
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

pipeline = AutoPipelineForText2Image.from_pretrained("path/to/saved/model", torch_dtype=torch.float16).to("cuda")

caption = "A cute bird naruto holding a shield"
images = pipeline(
    caption,
    width=1024,
    height=1536,
    prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
    prior_guidance_scale=4.0,
    num_images_per_prompt=2,
).images
```

## Next steps

Congratulations on training a Wuerstchen model! To learn more about how to use your new model, the following may be helpful:

- Take a look at the [Wuerstchen](../api/pipelines/wuerstchen#text-to-image-generation) API documentation to learn more about how to use the pipeline for text-to-image generation and its limitations.
