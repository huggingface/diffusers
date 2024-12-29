# ControlNet training example for Stable Diffusion 3/3.5 (SD3/3.5)

The `train_controlnet_sd3.py` script shows how to implement the ControlNet training procedure and adapt it for [Stable Diffusion 3](https://arxiv.org/abs/2403.03206) and [Stable Diffusion 3.5](https://stability.ai/news/introducing-stable-diffusion-3-5).

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

Then cd in the `examples/controlnet` folder and run
```bash
pip install -r requirements_sd3.txt
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

## Circle filling dataset

The original dataset is hosted in the [ControlNet repo](https://huggingface.co/lllyasviel/ControlNet/blob/main/training/fill50k.zip). We re-uploaded it to be compatible with `datasets` [here](https://huggingface.co/datasets/fusing/fill50k). Note that `datasets` handles dataloading within the training script.
Please download the dataset and unzip it in the directory `fill50k` in the `examples/controlnet` folder.

## Training

First download the SD3 model from [Hugging Face Hub](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) or the SD3.5 model from [Hugging Face Hub](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium). We will use it as a base model for the ControlNet training.
> [!NOTE]
> As the model is gated, before using it with diffusers you first need to go to the [Stable Diffusion 3 Medium Hugging Face page](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) or [Stable Diffusion 3.5 Large Hugging Face page](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium), fill in the form and accept the gate. Once you are in, you need to log in so that your system knows you’ve accepted the gate. Use the command below to log in:

```bash
huggingface-cli login
```

This will also allow us to push the trained model parameters to the Hugging Face Hub platform.


Our training examples use two test conditioning images. They can be downloaded by running

```sh
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png

wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

Then run the following commands to train a ControlNet model.

```bash
export MODEL_DIR="stabilityai/stable-diffusion-3-medium-diffusers"
export OUTPUT_DIR="sd3-controlnet-out"

accelerate launch train_controlnet_sd3.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --train_data_dir="fill50k" \
    --resolution=1024 \
    --learning_rate=1e-5 \
    --max_train_steps=15000 \
    --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
    --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
    --validation_steps=100 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4
```

To train a ControlNet model for Stable Diffusion 3.5, replace the `MODEL_DIR` with `stabilityai/stable-diffusion-3.5-medium`.

To better track our training experiments, we're using flags `validation_image`, `validation_prompt`, and `validation_steps` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.

Our experiments were conducted on a single 40GB A100 GPU.

### Inference

Once training is done, we can perform inference like so:

```python
from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from diffusers.utils import load_image
import torch

base_model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
controlnet_path = "DavyMorgan/sd3-controlnet-out"

controlnet = SD3ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet
)
pipe.to("cuda", torch.float16)


control_image = load_image("./conditioning_image_1.png").resize((1024, 1024))
prompt = "pale golden rod circle with old lace background"

# generate image
generator = torch.manual_seed(0)
image = pipe(
    prompt, num_inference_steps=20, generator=generator, control_image=control_image
).images[0]
image.save("./output.png")
```

Similarly, for SD3.5, replace the `base_model_path` with `stabilityai/stable-diffusion-3.5-medium` and controlnet_path `DavyMorgan/sd35-controlnet-out'.

## Notes

### GPU usage

SD3 is a large model and requires a lot of GPU memory. 
We recommend using one GPU with at least 80GB of memory.
Make sure to use the right GPU when configuring the [accelerator](https://huggingface.co/docs/transformers/en/accelerate).


## Example results

### SD3

#### After 500 steps with batch size 8

| |  |
|-------------------|:-------------------------:|
|| pale golden rod circle with old lace background |
 ![conditioning image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png) | ![pale golden rod circle with old lace background](https://huggingface.co/datasets/DavyMorgan/sd3-controlnet-results/resolve/main/step-500.png) |


#### After 6500 steps with batch size 8:

| |  |
|-------------------|:-------------------------:|
|| pale golden rod circle with old lace background |
 ![conditioning image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png) | ![pale golden rod circle with old lace background](https://huggingface.co/datasets/DavyMorgan/sd3-controlnet-results/resolve/main/step-6500.png) |

### SD3.5

#### After 500 steps with batch size 8

| |                                                                                                                                                     |
|-------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------:|
||                                                   pale golden rod circle with old lace background                                                   |
 ![conditioning image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png) | ![pale golden rod circle with old lace background](https://huggingface.co/datasets/DavyMorgan/sd3-controlnet-results/resolve/main/step-500-3.5.png) |


#### After 3000 steps with batch size 8:

| |                                                                                                                                                      |
|-------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------:|
||                                                   pale golden rod circle with old lace background                                                    |
 ![conditioning image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png) | ![pale golden rod circle with old lace background](https://huggingface.co/datasets/DavyMorgan/sd3-controlnet-results/resolve/main/step-3000-3.5.png) |

