<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# T2I-Adapter

[T2I-Adapter]((https://hf.co/papers/2302.08453)) is a lightweight adapter model that provides an additional conditioning input image (line art, canny, sketch, depth, pose) to better control image generation. It is similar to a ControlNet, but it is a lot smaller (~77M parameters and ~300MB file size) because its only inserts weights into the UNet instead of copying and training it.

The T2I-Adapter is only available for training with the Stable Diffusion XL (SDXL) model.

This guide will explore the [train_t2i_adapter_sdxl.py](https://github.com/huggingface/diffusers/blob/main/examples/t2i_adapter/train_t2i_adapter_sdxl.py) training script to help you become familiar with it, and how you can adapt it for your own use-case.

Before running the script, make sure you install the library from source:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then navigate to the example folder containing the training script and install the required dependencies for the script you're using:

```bash
cd examples/t2i_adapter
pip install -r requirements.txt
```

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

```bash
from accelerate.utils import write_basic_config

write_basic_config()
```

Lastly, if you want to train a model on your own dataset, take a look at the [Create a dataset for training](create_dataset) guide to learn how to create a dataset that works with the training script.

<Tip>

The following sections highlight parts of the training script that are important for understanding how to modify it, but it doesn't cover every aspect of the script in detail. If you're interested in learning more, feel free to read through the [script](https://github.com/huggingface/diffusers/blob/main/examples/t2i_adapter/train_t2i_adapter_sdxl.py) and let us know if you have any questions or concerns.

</Tip>

## Script parameters

The training script provides many parameters to help you customize your training run. All of the parameters and their descriptions are found in the [`parse_args()`](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L233) function. It provides default values for each parameter, such as the training batch size and learning rate, but you can also set your own values in the training command if you'd like.

For example, to activate gradient accumulation, add the `--gradient_accumulation_steps` parameter to the training command:

```bash
accelerate launch train_t2i_adapter_sdxl.py \
  ----gradient_accumulation_steps=4
```

Many of the basic and important parameters are described in the [Text-to-image](text2image#script-parameters) training guide, so this guide just focuses on the relevant T2I-Adapter parameters:

- `--pretrained_vae_model_name_or_path`: path to a pretrained VAE; the SDXL VAE is known to suffer from numerical instability, so this parameter allows you to specify a better [VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
- `--crops_coords_top_left_h` and `--crops_coords_top_left_w`: height and width coordinates to include in SDXL's crop coordinate embeddings
- `--conditioning_image_column`: the column of the conditioning images in the dataset
- `--proportion_empty_prompts`: the proportion of image prompts to replace with empty strings

## Training script

As with the script parameters, a walkthrough of the training script is provided in the [Text-to-image](text2image#training-script) training guide. Instead, this guide takes a look at the T2I-Adapter relevant parts of the script.

The training script begins by preparing the dataset. This incudes [tokenizing](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L674) the prompt and [applying transforms](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L714) to the images and conditioning images.

```py
conditioning_image_transforms = transforms.Compose(
    [
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
    ]
)
```

Within the [`main()`](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L770) function, the T2I-Adapter is either loaded from a pretrained adapter or it is randomly initialized:

```py
if args.adapter_model_name_or_path:
    logger.info("Loading existing adapter weights.")
    t2iadapter = T2IAdapter.from_pretrained(args.adapter_model_name_or_path)
else:
    logger.info("Initializing t2iadapter weights.")
    t2iadapter = T2IAdapter(
        in_channels=3,
        channels=(320, 640, 1280, 1280),
        num_res_blocks=2,
        downscale_factor=16,
        adapter_type="full_adapter_xl",
    )
```

The [optimizer](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L952) is initialized for the T2I-Adapter parameters:

```py
params_to_optimize = t2iadapter.parameters()
optimizer = optimizer_class(
    params_to_optimize,
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```

Lastly, in the [training loop](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L1086), the adapter conditioning image and the text embeddings are passed to the UNet to predict the noise residual:

```py
t2iadapter_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
down_block_additional_residuals = t2iadapter(t2iadapter_image)
down_block_additional_residuals = [
    sample.to(dtype=weight_dtype) for sample in down_block_additional_residuals
]

model_pred = unet(
    inp_noisy_latents,
    timesteps,
    encoder_hidden_states=batch["prompt_ids"],
    added_cond_kwargs=batch["unet_added_conditions"],
    down_block_additional_residuals=down_block_additional_residuals,
).sample
```

If you want to learn more about how the training loop works, check out the [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline) tutorial which breaks down the basic pattern of the denoising process.

## Launch the script

Now you’re ready to launch the training script! 🚀

For this example training, you'll use the [fusing/fill50k](https://huggingface.co/datasets/fusing/fill50k) dataset. You can also create and use your own dataset if you want (see the [Create a dataset for training](https://moon-ci-docs.huggingface.co/docs/diffusers/pr_5512/en/training/create_dataset) guide).

Set the environment variable `MODEL_DIR` to a model id on the Hub or a path to a local model and `OUTPUT_DIR` to where you want to save the model.

Download the following images to condition your training with:

```bash
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

<Tip>

To monitor training progress with Weights & Biases, add the `--report_to=wandb` parameter to the training command. You'll also need to add the `--validation_image`, `--validation_prompt`, and `--validation_steps` to the training command to keep track of results. This can be really useful for debugging the model and viewing intermediate results.

</Tip>

```bash
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="path to save model"

accelerate launch train_t2i_adapter_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --validation_steps=100 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --seed=42 \
 --push_to_hub
```

Once training is complete, you can use your T2I-Adapter for inference:

```py
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteSchedulerTest
from diffusers.utils import load_image
import torch

adapter = T2IAdapter.from_pretrained("path/to/adapter", torch_dtype=torch.float16)
pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", adapter=adapter, torch_dtype=torch.float16
)

pipeline.scheduler = EulerAncestralDiscreteSchedulerTest.from_config(pipe.scheduler.config)
pipeline.enable_xformers_memory_efficient_attention()
pipeline.enable_model_cpu_offload()

control_image = load_image("./conditioning_image_1.png")
prompt = "pale golden rod circle with old lace background"

generator = torch.manual_seed(0)
image = pipeline(
    prompt, image=control_image, generator=generator
).images[0]
image.save("./output.png")
```

## Next steps

Congratulations on training a T2I-Adapter model! 🎉 To learn more:

- Read the [Efficient Controllable Generation for SDXL with T2I-Adapters](https://www.cs.cmu.edu/~custom-diffusion/) blog post to learn more details about the experimental results from the T2I-Adapter team.
