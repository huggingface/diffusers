<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ControlNet

[ControlNet](https://hf.co/papers/2302.05543) models are adapters trained on top of another pretrained model. It allows for a greater degree of control over image generation by conditioning the model with an additional input image. The input image can be a canny edge, depth map, human pose, and many more.

If you're training on a GPU with limited vRAM, you should try enabling the `gradient_checkpointing`, `gradient_accumulation_steps`, and `mixed_precision` parameters in the training command. You can also reduce your memory footprint by using memory-efficient attention with [xFormers](../optimization/xformers). JAX/Flax training is also supported for efficient training on TPUs and GPUs, but it doesn't support gradient checkpointing or xFormers. You should have a GPU with >30GB of memory if you want to train faster with Flax.

This guide will explore the [train_controlnet.py](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py) training script to help you become familiar with it, and how you can adapt it for your own use-case.

Before running the script, make sure you install the library from source:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then navigate to the example folder containing the training script and install the required dependencies for the script you're using:

<hfoptions id="installation">
<hfoption id="PyTorch">
```bash
cd examples/controlnet
pip install -r requirements.txt
```
</hfoption>
<hfoption id="Flax">

If you have access to a TPU, the Flax training script runs even faster! Let's run the training script on the [Google Cloud TPU VM](https://cloud.google.com/tpu/docs/run-calculation-jax). Create a single TPU v4-8 VM and connect to it:

```bash
ZONE=us-central2-b
TPU_TYPE=v4-8
VM_NAME=hg_flax

gcloud alpha compute tpus tpu-vm create $VM_NAME \
 --zone $ZONE \
 --accelerator-type $TPU_TYPE \
 --version  tpu-vm-v4-base

gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone $ZONE -- \
```

Install JAX 0.4.5:

```bash
pip install "jax[tpu]==0.4.5" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Then install the required dependencies for the Flax script:

```bash
cd examples/controlnet
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

The following sections highlight parts of the training script that are important for understanding how to modify it, but it doesn't cover every aspect of the script in detail. If you're interested in learning more, feel free to read through the [script](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py) and let us know if you have any questions or concerns.

</Tip>

## Script parameters

The training script provides many parameters to help you customize your training run. All of the parameters and their descriptions are found in the [`parse_args()`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/controlnet/train_controlnet.py#L231) function. This function provides default values for each parameter, such as the training batch size and learning rate, but you can also set your own values in the training command if you'd like.

For example, to speedup training with mixed precision using the fp16 format, add the `--mixed_precision` parameter to the training command:

```bash
accelerate launch train_controlnet.py \
  --mixed_precision="fp16"
```

Many of the basic and important parameters are described in the [Text-to-image](text2image#script-parameters) training guide, so this guide just focuses on the relevant parameters for ControlNet:

- `--max_train_samples`: the number of training samples; this can be lowered for faster training, but if you want to stream really large datasets, you'll need to include this parameter and the `--streaming` parameter in your training command
- `--gradient_accumulation_steps`: number of update steps to accumulate before the backward pass; this allows you to train with a bigger batch size than your GPU memory can typically handle

### Min-SNR weighting

The [Min-SNR](https://huggingface.co/papers/2303.09556) weighting strategy can help with training by rebalancing the loss to achieve faster convergence. The training script supports predicting `epsilon` (noise) or `v_prediction`, but Min-SNR is compatible with both prediction types. This weighting strategy is only supported by PyTorch and is unavailable in the Flax training script.

Add the `--snr_gamma` parameter and set it to the recommended value of 5.0:

```bash
accelerate launch train_controlnet.py \
  --snr_gamma=5.0
```

## Training script

As with the script parameters, a general walkthrough of the training script is provided in the [Text-to-image](text2image#training-script) training guide. Instead, this guide takes a look at the relevant parts of the ControlNet script.

The training script has a [`make_train_dataset`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/controlnet/train_controlnet.py#L582) function for preprocessing the dataset with image transforms and caption tokenization. You'll see that in addition to the usual caption tokenization and image transforms, the script also includes transforms for the conditioning image.

<Tip>

If you're streaming a dataset on a TPU, performance may be bottlenecked by the 🤗 Datasets library which is not optimized for images. To ensure maximum throughput, you're encouraged to explore other dataset formats like [WebDataset](https://webdataset.github.io/webdataset/), [TorchData](https://github.com/pytorch/data), and [TensorFlow Datasets](https://www.tensorflow.org/datasets/tfless_tfds).

</Tip>

```py
conditioning_image_transforms = transforms.Compose(
    [
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
    ]
)
```

Within the [`main()`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/controlnet/train_controlnet.py#L713) function, you'll find the code for loading the tokenizer, text encoder, scheduler and models. This is also where the ControlNet model is loaded either from existing weights or randomly initialized from a UNet:

```py
if args.controlnet_model_name_or_path:
    logger.info("Loading existing controlnet weights")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
else:
    logger.info("Initializing controlnet weights from unet")
    controlnet = ControlNetModel.from_unet(unet)
```

The [optimizer](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/controlnet/train_controlnet.py#L871) is set up to update the ControlNet parameters:

```py
params_to_optimize = controlnet.parameters()
optimizer = optimizer_class(
    params_to_optimize,
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```

Finally, in the [training loop](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/controlnet/train_controlnet.py#L943), the conditioning text embeddings and image are passed to the down and mid-blocks of the ControlNet model:

```py
encoder_hidden_states = text_encoder(batch["input_ids"])[0]
controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

down_block_res_samples, mid_block_res_sample = controlnet(
    noisy_latents,
    timesteps,
    encoder_hidden_states=encoder_hidden_states,
    controlnet_cond=controlnet_image,
    return_dict=False,
)
```

If you want to learn more about how the training loop works, check out the [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline) tutorial which breaks down the basic pattern of the denoising process.

## Launch the script

Now you're ready to launch the training script! 🚀

This guide uses the [fusing/fill50k](https://huggingface.co/datasets/fusing/fill50k) dataset, but remember, you can create and use your own dataset if you want (see the [Create a dataset for training](create_dataset) guide).

Set the environment variable `MODEL_NAME` to a model id on the Hub or a path to a local model and `OUTPUT_DIR` to where you want to save the model.

Download the following images to condition your training with:

```bash
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

One more thing before you launch the script! Depending on the GPU you have, you may need to enable certain optimizations to train a ControlNet. The default configuration in this script requires ~38GB of vRAM. If you're training on more than one GPU, add the `--multi_gpu` parameter to the `accelerate launch` command.

<hfoptions id="gpu-select">
<hfoption id="16GB">

On a 16GB GPU, you can use bitsandbytes 8-bit optimizer and gradient checkpointing to optimize your training run. Install bitsandbytes:

```py
pip install bitsandbytes
```

Then, add the following parameter to your training command:

```bash
accelerate launch train_controlnet.py \
  --gradient_checkpointing \
  --use_8bit_adam \
```

</hfoption>
<hfoption id="12GB">

On a 12GB GPU, you'll need bitsandbytes 8-bit optimizer, gradient checkpointing, xFormers, and set the gradients to `None` instead of zero to reduce your memory-usage.

```bash
accelerate launch train_controlnet.py \
  --use_8bit_adam \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none \
```

</hfoption>
<hfoption id="8GB">

On a 8GB GPU, you'll need to use [DeepSpeed](https://www.deepspeed.ai/) to offload some of the tensors from the vRAM to either the CPU or NVME to allow training with less GPU memory.

Run the following command to configure your 🤗 Accelerate environment:

```bash
accelerate config
```

During configuration, confirm that you want to use DeepSpeed stage 2. Now it should be possible to train on under 8GB vRAM by combining DeepSpeed stage 2, fp16 mixed precision, and offloading the model parameters and the optimizer state to the CPU. The drawback is that this requires more system RAM (~25 GB). See the [DeepSpeed documentation](https://huggingface.co/docs/accelerate/usage_guides/deepspeed) for more configuration options. Your configuration file should look something like:

```bash
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 4
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
```

You should also change the default Adam optimizer to DeepSpeed’s optimized version of Adam [`deepspeed.ops.adam.DeepSpeedCPUAdam`](https://deepspeed.readthedocs.io/en/latest/optimizers.html#adam-cpu) for a substantial speedup. Enabling `DeepSpeedCPUAdam` requires your system’s CUDA toolchain version to be the same as the one installed with PyTorch.

bitsandbytes 8-bit optimizers don’t seem to be compatible with DeepSpeed at the moment.

That's it! You don't need to add any additional parameters to your training command.

</hfoption>
</hfoptions>

<hfoptions id="training-inference">
<hfoption id="PyTorch">

```bash
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path/to/save/model"

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --push_to_hub
```

</hfoption>
<hfoption id="Flax">

With Flax, you can [profile your code](https://jax.readthedocs.io/en/latest/profiling.html) by adding the `--profile_steps==5` parameter to your training command. Install the Tensorboard profile plugin:

```bash
pip install tensorflow tensorboard-plugin-profile
tensorboard --logdir runs/fill-circle-100steps-20230411_165612/
```

Then you can inspect the profile at [http://localhost:6006/#profile](http://localhost:6006/#profile).

<Tip warning={true}>

If you run into version conflicts with the plugin, try uninstalling and reinstalling all versions of TensorFlow and Tensorboard. The debugging functionality of the profile plugin is still experimental, and not all views are fully functional. The `trace_viewer` cuts off events after 1M, which can result in all your device traces getting lost if for example, you profile the compilation step by accident.

</Tip>

```bash
python3 train_controlnet_flax.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --validation_steps=1000 \
 --train_batch_size=2 \
 --revision="non-ema" \
 --from_pt \
 --report_to="wandb" \
 --tracker_project_name=$HUB_MODEL_ID \
 --num_train_epochs=11 \
 --push_to_hub \
 --hub_model_id=$HUB_MODEL_ID
```

</hfoption>
</hfoptions>

Once training is complete, you can use your newly trained model for inference!

```py
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch

controlnet = ControlNetModel.from_pretrained("path/to/controlnet", torch_dtype=torch.float16)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "path/to/base/model", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

control_image = load_image("./conditioning_image_1.png")
prompt = "pale golden rod circle with old lace background"

generator = torch.manual_seed(0)
image = pipeline(prompt, num_inference_steps=20, generator=generator, image=control_image).images[0]
image.save("./output.png")
```

## Stable Diffusion XL

Stable Diffusion XL (SDXL) is a powerful text-to-image model that generates high-resolution images, and it adds a second text-encoder to its architecture. Use the [`train_controlnet_sdxl.py`](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet_sdxl.py) script to train a ControlNet adapter for the SDXL model.

The SDXL training script is discussed in more detail in the [SDXL training](sdxl) guide.

## Next steps

Congratulations on training your own ControlNet! To learn more about how to use your new model, the following guides may be helpful:

- Learn how to [use a ControlNet](../using-diffusers/controlnet) for inference on a variety of tasks.
