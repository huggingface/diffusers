# SVDiff
This is an implementation of [SVDiff: Compact Parameter Space for Diffusion Fine-Tuning](https://arxiv.org/abs/2303.11305) by using d🧨ffusers. 



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

## Single-Subject Generation
"Single-Subject Generation" is a domain-tuning on a single object or concept (using 3-5 images).

### Training
According to the paper, the learning rate for SVDiff needs to be 1000 times larger than the lr used for fine-tuning. 

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_svdiff.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="photo of a sks dog" \
  --class_prompt="photo of a dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-3 \
  --learning_rate_1d=1e-6 \
  --train_text_encoder \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=500
```

### Inference

```python
import torch
from transformers import CLIPTextModel
from diffusers import UNet2DConditionModel, StableDiffusionPipeline, DPMSolverMultistepScheduler

from modeling_svdiff import set_spectral_shifts

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
spectral_shifts_ckpt_dir = "ckpt-dir-path"
device = "cuda" if torch.cuda.is_available() else "cpu"
unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet").to(device)
text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").to(device)
# make sure to move models to device before
unet, _ = set_spectral_shifts(unet, spectral_shifts_ckpt=os.path.join(spectral_shifts_ckpt, "spectral_shifts.safetensors"))
text_encoder, _ = set_spectral_shifts(text_encoder, spectral_shifts_ckpt=os.path.join(spectral_shifts_ckpt, "spectral_shifts_te.safetensors"))
# load pipe
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    unet=unet,
    text_encoder=text_encoder,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(device)
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
```

## Single Image Editing
### Training
In Single Image Editing, your instance prompt should be just the description of your input image **without the identifier**. 

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="dir-path-to-input-image"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_svdiff.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="photo of a pink chair with black legs" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-3 \
  --learning_rate_1d=1e-6 \
  --train_text_encoder \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500
```

### Inference

```python
from PIL import Image
import torch
from transformers import CLIPTextModel
from diffusers import UNet2DConditionModel, DDIMScheduler

from modeling_svdiff import set_spectral_shifts
from pipeline_stable_diffusion_ddim_inversion import StableDiffusionPipelineWithDDIMInversion

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
spectral_shifts_ckpt_dir = "ckpt-dir-path"
image = "path-to-image"
source_prompt = "prompt-for-image"
target_prompt = "prompt-you-want-to-generate"

device = "cuda" if torch.cuda.is_available() else "cpu"
unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet").to(device)
text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").to(device)
# make sure to move models to device before
unet, _ = set_spectral_shifts(unet, spectral_shifts_ckpt=os.path.join(spectral_shifts_ckpt, "spectral_shifts.safetensors"))
text_encoder, _ = set_spectral_shifts(text_encoder, spectral_shifts_ckpt=os.path.join(spectral_shifts_ckpt, "spectral_shifts_te.safetensors"))
# load pipe
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    unet=unet,
    text_encoder=text_encoder,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

# (optional) ddim inversion
# if you don't do it, inv_latents = None
image = Image.open(image).convert("RGB").resize((512, 512))
# in SVDiff, they use guidance scale=1 in ddim inversion
# They use target_prompt in DDIM inversion for better results. See below for comparison between source_prompt and target_prompt.
inv_latents = pipe.invert(target_prompt, image=image, guidance_scale=1.0).latents

# They use a small cfg scale in Single Image Editing 
image = pipe(target_prompt, latents=inv_latents, guidance_scale=3, eta=0.5).images[0]
```

To use slerp to add more stochasticity,
```python
from modeling_svdiff import slerp_tensor

# prev steps omitted
inv_latents = pipe.invert(target_prompt, image=image, guidance_scale=1.0).latents
noise_latents = pipe.prepare_latents(inv_latents.shape[0], inv_latents.shape[1], 512, 512, dtype=inv_latents.dtype, device=pipe.device, generator=torch.Generator("cuda").manual_seed(0))
inv_latents =  slerp_tensor(0.5, inv_latents, noise_latents)
image = pipe(target_prompt, latents=inv_latents).images[0]
```


## Additional Features

### Spectral Shift Scaling

You can adjust the strength of the weights.

![scale](https://github.com/mkshing/svdiff-pytorch/raw/main/assets/scale.png)

```python
from modeling_svdiff import set_spectral_shifts
scale = 1.2

unet, _ = set_spectral_shifts(unet, spectral_shifts_ckpt=os.path.join(ckpt_dir, "spectral_shifts.safetensors"), scale=scale)
text_encoder, _ = set_spectral_shifts(text_encoder, spectral_shifts_ckpt=os.path.join(ckpt_dir, "spectral_shifts_te.safetensors"), scale=scale)
```