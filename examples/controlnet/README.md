# Controlnet training example

[Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) by Lvmin Zhang and Maneesh Agrawala.

Using the pretrained models we can provide control images (for example, a depth map) to control Stable Diffusion text-to-image generation so that it follows the structure of the depth image and fills in the details. `train_controlnet.py` script shows how to train a controlnet from scratch with simple example data.

This example is based on the [training example in the original Controlnet repository](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md) and it trains a controlnet that can fill circles using a small synthetic dataset. The end results isn't very useful but it should serve as a starting point for controlnet training.

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

Download the fill50k dataset from [huggingface page](https://huggingface.co/lllyasviel/ControlNet) and extract it. You should have a json file with the prompts and two directories, source and target, with training images.

## Initializing the model

Next, you should have a Stable Diffusion checkpoint without an existing controlnet. For example [Stable diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main). You should download the diffusers weights in the folders or convert ckpt or safetensor with `scripts/convert_original_stable_diffusion_to_diffusers.py`.

Empty controlnet can be added to an existing model with
`examples/controlnet/add_controlnet.py` script.

```bash
export MODEL_DIR="path to model"

python examples/controlnet/add_controlnet.py --pretrained_model_path=MODEL_DIR
```

## Training

After having set up the dataset and initializing a controlnet model training can be started.

```bash
export MODEL_DIR="path to model"
export DATASET_DIR="path to extracted fill50k dataset"
export OUTPUT_DIR="path to save model"

accelerate launch examples/controlnet/train_controlnet.py \
 --pretrained_model_name_or_path=MODEL_DIR \
 --instance_data_dir=DATASET_DIR  \
 --output_dir=OUTPUT_DIR \
 --resolution=512 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --learning_rate=1e-5 \
 --lr_warmup_steps=0 \
 --max_train_steps=10000
```

If you run into OOM errors you can try one or more options below:

```bash
--mixed_precision=fp16
--set_grads_to_none
--enable_xformers_memory_efficient_attention
--use_8bit_adam
--gradient_checkpointing
```

Using all the above options it should be possible to be able to train with 8 GB VRAM.

## Testing the trained model

The trained model can be tested with the following code. Change `model` and `control_image_path` to proper values.

```python
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
import torch

model = "path to model"
control_image_path = "path to control image"

control_image = load_image(control_image_path)
prompt = "pale golden rod circle with old lace background"

pipe = StableDiffusionControlNetPipeline.from_pretrained(model, safety_checker=None, torch_dtype=torch.float16).to("cuda")
with torch.inference_mode():
    image = pipe(prompt=prompt, image=control_image).images[0]
image.save("./output.png")
```

If everything worked correctly the "./output.png" image should contain an image of
a filled circle with outline that matches the control image.
