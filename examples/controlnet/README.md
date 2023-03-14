# ControlNet training example

[Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) by Lvmin Zhang and Maneesh Agrawala.

This example is based on the [training example in the original ControlNet repository](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md). It trains a ControlNet to fill circles using a [small synthetic dataset](https://huggingface.co/datasets/fusing/fill50k).

## Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the example folder and run
```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a notebook

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

## Circle filling dataset

The original dataset is hosted in the [ControlNet repo](https://huggingface.co/lllyasviel/ControlNet/blob/main/training/fill50k.zip). We re-uploaded it to be compatible with `datasets` [here](https://huggingface.co/datasets/fusing/fill50k). Note that `datasets` handles dataloading within the training script.

Our training examples use [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) as the original set of ControlNet models were trained from it. However, ControlNet can be trained to augment any Stable Diffusion compatible model (such as [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)) or [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1).

## Training

Our training examples use two test conditioning images. They can be downloaded by running

```sh
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png

wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```


```bash
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path to save model"

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=4
```

This default configuration requires ~38GB VRAM.

By default, the training script logs outputs to tensorboard. Pass `--report_to wandb` to use weights and
biases.

Gradient accumulation with a smaller batch size can be used to reduce training requirements to ~20 GB VRAM.

```bash
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path to save model"

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4
```

## Example results

#### After 300 steps with batch size 8

| |  | 
|-------------------|:-------------------------:|
| | red circle with blue background  | 
![conditioning image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png) | ![red circle with blue background](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/red_circle_with_blue_background_300_steps.png) |
| | cyan circle with brown floral background | 
![conditioning image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png) | ![cyan circle with brown floral background](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/cyan_circle_with_brown_floral_background_300_steps.png) |


#### After 6000 steps with batch size 8:

| |  | 
|-------------------|:-------------------------:|
| | red circle with blue background  | 
![conditioning image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png) | ![red circle with blue background](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/red_circle_with_blue_background_6000_steps.png) |
| | cyan circle with brown floral background | 
![conditioning image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png) | ![cyan circle with brown floral background](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/cyan_circle_with_brown_floral_background_6000_steps.png) |

## Training on a 16 GB GPU

Optimizations:
- Gradient checkpointing
- bitsandbyte's 8-bit optimizer

[bitandbytes install instructions](https://github.com/TimDettmers/bitsandbytes#requirements--installation).

```bash
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path to save model"

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
 --gradient_checkpointing \
 --use_8bit_adam
```

## Training on a 12 GB GPU

Optimizations:
- Gradient checkpointing
- bitsandbyte's 8-bit optimizer
- xformers
- set grads to none

```bash
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path to save model"

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
 --gradient_checkpointing \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none
```

When using `enable_xformers_memory_efficient_attention`, please make sure to install `xformers` by `pip install xformers`. 

## Training on an 8 GB GPU

We have not exhaustively tested DeepSpeed support for ControlNet. While the configuration does
save memory, we have not confirmed the configuration to train successfully. You will very likely
have to make changes to the config to have a successful training run.

Optimizations:
- Gradient checkpointing
- xformers
- set grads to none
- DeepSpeed stage 2 with parameter and optimizer offloading
- fp16 mixed precision

[DeepSpeed](https://www.deepspeed.ai/) can offload tensors from VRAM to either 
CPU or NVME. This requires significantly more RAM (about 25 GB).

Use `accelerate config` to enable DeepSpeed stage 2.

The relevant parts of the resulting accelerate config file are

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 4
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
```

See [documentation](https://huggingface.co/docs/accelerate/usage_guides/deepspeed) for more DeepSpeed configuration options.

Changing the default Adam optimizer to DeepSpeed's Adam
`deepspeed.ops.adam.DeepSpeedCPUAdam` gives a substantial speedup but
it requires CUDA toolchain with the same version as pytorch. 8-bit optimizer
does not seem to be compatible with DeepSpeed at the moment.

```bash
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path to save model"

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none \
 --mixed_precision fp16
```

## Performing inference with the trained ControlNet

The trained model can be run the same as the original ControlNet pipeline with the newly trained ControlNet.
Set `base_model_path` and `controlnet_path` to the values `--pretrained_model_name_or_path` and 
`--output_dir` were respectively set to in the training script.

```py
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch

base_model_path = "path to model"
controlnet_path = "path to controlnet"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

control_image = load_image("./conditioning_image_1.png")
prompt = "pale golden rod circle with old lace background"

# generate image
generator = torch.manual_seed(0)
image = pipe(
     prompt, num_inference_steps=20, generator=generator, image=control_image
).images[0]

image.save("./output.png")
```
