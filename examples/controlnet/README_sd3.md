# ControlNet training example for Stable Diffusion 3 (SD3)

The `train_controlnet_sd3.py` script shows how to implement the ControlNet training procedure and adapt it for [Stable Diffusion 3](https://arxiv.org/abs/2403.03206).

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

First download the SD3 model from [Hugging Face Hub](https://huggingface.co/stabilityai/stable-diffusion-3-medium). We will use it as a base model for the ControlNet training.

Our training examples use two test conditioning images. They can be downloaded by running

```sh
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png

wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

Then run `huggingface-cli login` to log into your Hugging Face account. This is needed to be able to push the trained ControlNet parameters to Hugging Face Hub.

```bash
export MODEL_DIR="path to sd3 model"
export OUTPUT_DIR="path to save model"

CUDA_VISIBLE_DEVICES="0,1" accelerate launch train_controlnet_sd3.py \
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

To better track our training experiments, we're using flags `validation_image`, `validation_prompt`, and `validation_steps` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.

Our experiments were conducted on a single 40GB A100 GPU.

### Inference

Once training is done, we can perform inference like so:

```python
from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from diffusers.utils import load_image
import torch

base_model_path = "path to sd3 model"
controlnet_path = "path to controlnet ckpt"

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

## Notes

### GPU usage

SD3 is a large model and requires a lot of GPU memory. 
We recommend using two GPUs with at least 80GB of memory each, as indicated in the training instructions above.
Specifically, one GPU is used for the ControlNet-SD3 training, and the other is used for validation.
In this example, we set `CUDA_VISIBLE_DEVICES="0,1"` to use the first two GPUs on the machine.
Make sure to use GPU 0 when configuring the accelerator, since the validation will use GPU 1.
