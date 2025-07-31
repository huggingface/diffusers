# Training Flux Control

This (experimental) example shows how to train Control LoRAs with [Flux](https://huggingface.co/black-forest-labs/FLUX.1-dev) by conditioning it with additional structural controls (like depth maps, poses, etc.). We provide a script for full fine-tuning, too, refer to [this section](#full-fine-tuning). To know more about Flux Control family, refer to the following resources:

* [Docs](https://github.com/black-forest-labs/flux/blob/main/docs/structural-conditioning.md) by Black Forest Labs
* Diffusers docs ([1](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux#canny-control), [2](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux#depth-control))

To incorporate additional condition latents, we expand the input features of Flux.1-Dev from 64 to 128. The first 64 channels correspond to the original input latents to be denoised, while the latter 64 channels correspond to control latents. This expansion happens on the `x_embedder` layer, where the combined latents are projected to the expected feature dimension of rest of the network. Inference is performed using the `FluxControlPipeline`.

> [!NOTE]
> **Gated model**
>
> As the model is gated, before using it with diffusers you first need to go to the [FLUX.1 [dev] Hugging Face page](https://huggingface.co/black-forest-labs/FLUX.1-dev), fill in the form and accept the gate. Once you are in, you need to log in so that your system knows you’ve accepted the gate. Use the command below to log in:

```bash
hf auth login
```

The example command below shows how to launch fine-tuning for pose conditions. The dataset ([`raulc0399/open_pose_controlnet`](https://huggingface.co/datasets/raulc0399/open_pose_controlnet)) being used here already has the pose conditions of the original images, so we don't have to compute them.

```bash
accelerate launch train_control_lora_flux.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --dataset_name="raulc0399/open_pose_controlnet" \
  --output_dir="pose-control-lora" \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --rank=64 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --validation_image="openpose.png" \
  --validation_prompt="A couple, 4k photo, highly detailed" \
  --offload \
  --seed="0" \
  --push_to_hub
```

`openpose.png` comes from [here](https://huggingface.co/Adapter/t2iadapter/resolve/main/openpose.png).

You need to install `diffusers` from the branch of [this PR](https://github.com/huggingface/diffusers/pull/9999). When it's merged, you should install `diffusers` from the `main`.

The training script exposes additional CLI args that might be useful to experiment with:

* `use_lora_bias`: When set, additionally trains the biases of the `lora_B` layer. 
* `train_norm_layers`: When set, additionally trains the normalization scales. Takes care of saving and loading.
* `lora_layers`: Specify the layers you want to apply LoRA to. If you specify "all-linear", all the linear layers will be LoRA-attached.

### Training with DeepSpeed

It's possible to train with [DeepSpeed](https://github.com/microsoft/DeepSpeed), specifically leveraging the Zero2 system optimization. To use it, save the following config to an YAML file (feel free to modify as needed):

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

And then while launching training, pass the config file:

```bash
accelerate launch --config_file=CONFIG_FILE.yaml ...
```

### Inference

The pose images in our dataset were computed using the [`controlnet_aux`](https://github.com/huggingface/controlnet_aux) library. Let's install it first:

```bash
pip install controlnet_aux
```

And then we are ready:

```py
from controlnet_aux import OpenposeDetector
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import torch 

pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("...") # change this.

open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

# prepare pose condition.
url = "https://huggingface.co/Adapter/t2iadapter/resolve/main/people.jpg"
image = load_image(url)
image = open_pose(image, detect_resolution=512, image_resolution=1024)
image = np.array(image)[:, :, ::-1]           
image = Image.fromarray(np.uint8(image))

prompt = "A couple, 4k photo, highly detailed"

gen_images = pipe(
  prompt=prompt,
  control_image=image,
  num_inference_steps=50,
  joint_attention_kwargs={"scale": 0.9},
  guidance_scale=25., 
).images[0]
gen_images.save("output.png")
```

## Full fine-tuning

We provide a non-LoRA version of the training script `train_control_flux.py`. Here is an example command:

```bash
accelerate launch --config_file=accelerate_ds2.yaml train_control_flux.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --dataset_name="raulc0399/open_pose_controlnet" \
  --output_dir="pose-control" \
  --mixed_precision="bf16" \
  --train_batch_size=2 \
  --dataloader_num_workers=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --proportion_empty_prompts=0.2 \
  --learning_rate=5e-5 \
  --adam_weight_decay=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=1000 \
  --checkpointing_steps=1000 \
  --max_train_steps=10000 \
  --validation_steps=200 \
  --validation_image "2_pose_1024.jpg" "3_pose_1024.jpg" \
  --validation_prompt "two friends sitting by each other enjoying a day at the park, full hd, cinematic" "person enjoying a day at the park, full hd, cinematic" \
  --offload \
  --seed="0" \
  --push_to_hub
```

Change the `validation_image` and `validation_prompt` as needed.

For inference, this time, we will run:

```py
from controlnet_aux import OpenposeDetector
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import torch 

transformer = FluxTransformer2DModel.from_pretrained("...") # change this.
pipe = FluxControlPipeline.from_pretrained(
  "black-forest-labs/FLUX.1-dev",  transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

# prepare pose condition.
url = "https://huggingface.co/Adapter/t2iadapter/resolve/main/people.jpg"
image = load_image(url)
image = open_pose(image, detect_resolution=512, image_resolution=1024)
image = np.array(image)[:, :, ::-1]           
image = Image.fromarray(np.uint8(image))

prompt = "A couple, 4k photo, highly detailed"

gen_images = pipe(
  prompt=prompt,
  control_image=image,
  num_inference_steps=50,
  guidance_scale=25., 
).images[0]
gen_images.save("output.png")
```

## Things to note

* The scripts provided in this directory are experimental and educational. This means we may have to tweak things around to get good results on a given condition. We believe this is best done with the community 🤗
* The scripts are not memory-optimized but we offload the VAE and the text encoders to CPU when they are not used if `--offload` is specified. 
* We can extract LoRAs from the fully fine-tuned model. While we currently don't provide any utilities for that, users are welcome to refer to [this script](https://github.com/Stability-AI/stability-ComfyUI-nodes/blob/master/control_lora_create.py) that provides a similar functionality. 