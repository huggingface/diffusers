<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Kandinsky 2.2

<Tip warning={true}>

This script is experimental, and it's easy to overfit and run into issues like catastrophic forgetting. Try exploring different hyperparameters to get the best results on your dataset.

</Tip>

Kandinsky 2.2 is a multilingual text-to-image model capable of producing more photorealistic images. The model includes an image prior model for creating image embeddings from text prompts, and a decoder model that generates images based on the prior model's embeddings. That's why you'll find two separate scripts in Diffusers for Kandinsky 2.2, one for training the prior model and one for training the decoder model. You can train both models separately, but to get the best results, you should train both the prior and decoder models.

Depending on your GPU, you may need to enable `gradient_checkpointing` (⚠️ not supported for the prior model!), `mixed_precision`, and `gradient_accumulation_steps` to help fit the model into memory and to speedup training. You can reduce your memory-usage even more by enabling memory-efficient attention with [xFormers](../optimization/xformers) (version [v0.0.16](https://github.com/huggingface/diffusers/issues/2234#issuecomment-1416931212) fails for training on some GPUs so you may need to install a development version instead).

This guide explores the [train_text_to_image_prior.py](https://github.com/huggingface/diffusers/blob/main/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py) and the [train_text_to_image_decoder.py](https://github.com/huggingface/diffusers/blob/main/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py) scripts to help you become more familiar with it, and how you can adapt it for your own use-case.

Before running the scripts, make sure you install the library from source:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then navigate to the example folder containing the training script and install the required dependencies for the script you're using:

```bash
cd examples/kandinsky2_2/text_to_image
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

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

Lastly, if you want to train a model on your own dataset, take a look at the [Create a dataset for training](create_dataset) guide to learn how to create a dataset that works with the training script.

<Tip>

The following sections highlight parts of the training scripts that are important for understanding how to modify it, but it doesn't cover every aspect of the scripts in detail. If you're interested in learning more, feel free to read through the scripts and let us know if you have any questions or concerns.

</Tip>

## Script parameters

The training scripts provides many parameters to help you customize your training run. All of the parameters and their descriptions are found in the [`parse_args()`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py#L190) function. The training scripts provides default values for each parameter, such as the training batch size and learning rate, but you can also set your own values in the training command if you'd like.

For example, to speedup training with mixed precision using the fp16 format, add the `--mixed_precision` parameter to the training command:

```bash
accelerate launch train_text_to_image_prior.py \
  --mixed_precision="fp16"
```

Most of the parameters are identical to the parameters in the [Text-to-image](text2image#script-parameters) training guide, so let's get straight to a walkthrough of the Kandinsky training scripts!

### Min-SNR weighting

The [Min-SNR](https://huggingface.co/papers/2303.09556) weighting strategy can help with training by rebalancing the loss to achieve faster convergence. The training script supports predicting `epsilon` (noise) or `v_prediction`, but Min-SNR is compatible with both prediction types. This weighting strategy is only supported by PyTorch and is unavailable in the Flax training script.

Add the `--snr_gamma` parameter and set it to the recommended value of 5.0:

```bash
accelerate launch train_text_to_image_prior.py \
  --snr_gamma=5.0
```

## Training script

The training script is also similar to the [Text-to-image](text2image#training-script) training guide, but it's been modified to support training the prior and decoder models. This guide focuses on the code that is unique to the Kandinsky 2.2 training scripts.

<hfoptions id="script">
<hfoption id="prior model">

The [`main()`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py#L441) function contains the code for preparing the dataset and training the model.

One of the main differences you'll notice right away is that the training script also loads a [`~transformers.CLIPImageProcessor`] - in addition to a scheduler and tokenizer - for preprocessing images and a [`~transformers.CLIPVisionModelWithProjection`] model for encoding the images:

```py
noise_scheduler = DDPMScheduler(beta_schedule="squaredcos_cap_v2", prediction_type="sample")
image_processor = CLIPImageProcessor.from_pretrained(
    args.pretrained_prior_model_name_or_path, subfolder="image_processor"
)
tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_prior_model_name_or_path, subfolder="tokenizer")

with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_prior_model_name_or_path, subfolder="image_encoder", torch_dtype=weight_dtype
    ).eval()
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_prior_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
    ).eval()
```

Kandinsky uses a [`PriorTransformer`] to generate the image embeddings, so you'll want to setup the optimizer to learn the prior mode's parameters.

```py
prior = PriorTransformer.from_pretrained(args.pretrained_prior_model_name_or_path, subfolder="prior")
prior.train()
optimizer = optimizer_cls(
    prior.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```

Next, the input captions are tokenized, and images are [preprocessed](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py#L632) by the [`~transformers.CLIPImageProcessor`]:

```py
def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["clip_pixel_values"] = image_processor(images, return_tensors="pt").pixel_values
    examples["text_input_ids"], examples["text_mask"] = tokenize_captions(examples)
    return examples
```

Finally, the [training loop](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py#L718) converts the input images into latents, adds noise to the image embeddings, and makes a prediction:

```py
model_pred = prior(
    noisy_latents,
    timestep=timesteps,
    proj_embedding=prompt_embeds,
    encoder_hidden_states=text_encoder_hidden_states,
    attention_mask=text_mask,
).predicted_image_embedding
```

If you want to learn more about how the training loop works, check out the [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline) tutorial which breaks down the basic pattern of the denoising process.

</hfoption>
<hfoption id="decoder model">

The [`main()`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py#L440) function contains the code for preparing the dataset and training the model.

Unlike the prior model, the decoder initializes a [`VQModel`] to decode the latents into images and it uses a [`UNet2DConditionModel`]:

```py
with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    vae = VQModel.from_pretrained(
        args.pretrained_decoder_model_name_or_path, subfolder="movq", torch_dtype=weight_dtype
    ).eval()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_prior_model_name_or_path, subfolder="image_encoder", torch_dtype=weight_dtype
    ).eval()
unet = UNet2DConditionModel.from_pretrained(args.pretrained_decoder_model_name_or_path, subfolder="unet")
```

Next, the script includes several image transforms and a [preprocessing](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py#L622) function for applying the transforms to the images and returning the pixel values:

```py
def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["clip_pixel_values"] = image_processor(images, return_tensors="pt").pixel_values
    return examples
```

Lastly, the [training loop](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py#L706) handles converting the images to latents, adding noise, and predicting the noise residual.

If you want to learn more about how the training loop works, check out the [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline) tutorial which breaks down the basic pattern of the denoising process.

```py
model_pred = unet(noisy_latents, timesteps, None, added_cond_kwargs=added_cond_kwargs).sample[:, :4]
```

</hfoption>
</hfoptions>

## Launch the script

Once you’ve made all your changes or you’re okay with the default configuration, you’re ready to launch the training script! 🚀

You'll train on the [Naruto BLIP captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) dataset to generate your own Naruto characters, but you can also create and train on your own dataset by following the [Create a dataset for training](create_dataset) guide. Set the environment variable `DATASET_NAME` to the name of the dataset on the Hub or if you're training on your own files, set the environment variable `TRAIN_DIR` to a path to your dataset.

If you’re training on more than one GPU, add the `--multi_gpu` parameter to the `accelerate launch` command.

<Tip>

To monitor training progress with Weights & Biases, add the `--report_to=wandb` parameter to the training command. You’ll also need to add the `--validation_prompt` to the training command to keep track of results. This can be really useful for debugging the model and viewing intermediate results.

</Tip>

<hfoptions id="training-inference">
<hfoption id="prior model">

```bash
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_prior.py \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_prompts="A robot naruto, 4k photo" \
  --report_to="wandb" \
  --push_to_hub \
  --output_dir="kandi2-prior-naruto-model"
```

</hfoption>
<hfoption id="decoder model">

```bash
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_decoder.py \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_prompts="A robot naruto, 4k photo" \
  --report_to="wandb" \
  --push_to_hub \
  --output_dir="kandi2-decoder-naruto-model"
```

</hfoption>
</hfoptions>

Once training is finished, you can use your newly trained model for inference!

<hfoptions id="training-inference">
<hfoption id="prior model">

```py
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
import torch

prior_pipeline = DiffusionPipeline.from_pretrained(output_dir, torch_dtype=torch.float16)
prior_components = {"prior_" + k: v for k,v in prior_pipeline.components.items()}
pipeline = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", **prior_components, torch_dtype=torch.float16)

pipe.enable_model_cpu_offload()
prompt="A robot naruto, 4k photo"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt).images[0]
```

<Tip>

Feel free to replace `kandinsky-community/kandinsky-2-2-decoder` with your own trained decoder checkpoint!

</Tip>

</hfoption>
<hfoption id="decoder model">

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("path/to/saved/model", torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

prompt="A robot naruto, 4k photo"
image = pipeline(prompt=prompt).images[0]
```

For the decoder model, you can also perform inference from a saved checkpoint which can be useful for viewing intermediate results. In this case, load the checkpoint into the UNet:

```py
from diffusers import AutoPipelineForText2Image, UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("path/to/saved/model" + "/checkpoint-<N>/unet")

pipeline = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", unet=unet, torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

image = pipeline(prompt="A robot naruto, 4k photo").images[0]
```

</hfoption>
</hfoptions>

## Next steps

Congratulations on training a Kandinsky 2.2 model! To learn more about how to use your new model, the following guides may be helpful:

- Read the [Kandinsky](../using-diffusers/kandinsky) guide to learn how to use it for a variety of different tasks (text-to-image, image-to-image, inpainting, interpolation), and how it can be combined with a ControlNet.
- Check out the [DreamBooth](dreambooth) and [LoRA](lora) training guides to learn how to train a personalized Kandinsky model with just a few example images. These two training techniques can even be combined!
