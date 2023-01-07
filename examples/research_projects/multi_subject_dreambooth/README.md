# Multi Subject DreamBooth training

[DreamBooth](https://arxiv.org/abs/2208.12242) is a method to personalize text2image models like stable diffusion given just a few(3~5) images of a subject.
This `train_multi_subject_dreambooth.py` script shows how to implement the training procedure for one or more subjects and adapt it for stable diffusion. Note that this code is based off of the `examples/dreambooth/train_dreambooth.py` script as of 01/06/2022.

This script was added by @kopsahlong, and is not actively maintained. However, if you come across anything that could use fixing, feel free to open an issue and tag @kopsahlong.

## Running locally with PyTorch
### Installing the dependencies

Before running the script, make sure to install the library's training dependencies:

To start, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd into the folder `diffusers/examples/research_projects/multi_subject_dreambooth` and run the following:
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

### Multi Subject Training Example
In order to have your model learn multiple concepts at once, we simply add in the additional data directories and prompts to our `instance_data_dir` and `instance_prompt` (as well as `class_data_dir` and `class_prompt` if `--with_prior_preservation` is specified) as one comma separated string.

See an example with 2 subjects below, which learns a model for one dog subject and one human subject:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"

# Subject 1
export INSTANCE_DIR_1="path-to-instance-images-concept-1"
export INSTANCE_PROMPT_1="a photo of a sks dog"
export CLASS_DIR_1="path-to-class-images-dog"
export CLASS_PROMPT_1="a photo of a dog"

# Subject 2
export INSTANCE_DIR_2="path-to-instance-images-concept-2"
export INSTANCE_PROMPT_2="a photo of a t@y person"
export CLASS_DIR_2="path-to-class-images-person"
export CLASS_PROMPT_2="a photo of a person"

accelerate launch train_multi_subject_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir="$INSTANCE_DIR_1,$INSTANCE_DIR_2" \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --instance_prompt="$INSTANCE_PROMPT_1,$INSTANCE_PROMPT_2" \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --class_data_dir="$CLASS_DIR_1,$CLASS_DIR_2" \
  --class_prompt="$CLASS_PROMPT_1,$CLASS_PROMPT_2"\
  --num_class_images=50 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1500
```

This example shows training for 2 subjects, but please note that the model can be trained on any number of new concepts. This can be done by continuing to add in the corresponding directories and prompts to the corresponding comma separated string.

Note also that in this script, `sks` and `t@y` were used as tokens to learn the new subjects ([this thread](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion/issues/71) inspired the use of `t@y` as our second identifier). However, there may be better rare tokens to experiment with, and results also seemed to be good when more intuitive words are used.

### Inference

Once you have trained a model using above command, the inference can be done simply using the `StableDiffusionPipeline`. Make sure to include the `identifier`(e.g. sks in above example) in your prompt.

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "path-to-your-trained-model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of a t@y person petting an sks dog"
image = pipe(prompt, num_inference_steps=200, guidance_scale=7.5).images[0]

image.save("person-petting-dog.png")
```

### Inference from a training checkpoint

You can also perform inference from one of the checkpoints saved during the training process, if you used the `--checkpointing_steps` argument. Please, refer to [the documentation](https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint) to see how to do it.

## Additional Dreambooth documentation
Because the `train_multi_subject_dreambooth.py` script here was forked from an original version of `train_dreambooth.py` in the `examples/dreambooth` folder, I've included the original applicable training documentation for single subject examples below.

This should explain how to play with training variables such as prior preservation, fine tuning the text encoder, etc. which is still applicable to our multi subject training code. Note also that the examples below, which are single subject examples, also work with `train_multi_subject_dreambooth.py`, as this script supports 1 (or more) subjects.

### Single subject dog toy example

Let's get our dataset. Download images from [here](https://drive.google.com/drive/folders/1BO_dyz-p65qhBRRMRA4TbZ8qW4rB99JZ) and save them in a directory. This will be our training data.

And launch the training using

**___Note: Change the `resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.___**

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400
```

### Training with prior-preservation loss

Prior-preservation is used to avoid overfitting and language-drift. Refer to the paper to learn more about it. For prior-preservation we first generate images using the model with a class prompt and then use those during training along with our data.
According to the paper, it's recommended to generate `num_epochs * num_samples` images for prior-preservation. 200-300 works well for most cases. The `num_class_images` flag sets the number of images to generate with the class prompt. You can place existing images in `class_data_dir`, and the training script will generate any additional images so that `num_class_images` are present in `class_data_dir` during training time.

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```


### Training on a 16GB GPU:

With the help of gradient checkpointing and the 8-bit optimizer from bitsandbytes it's possible to run train dreambooth on a 16GB GPU.

To install `bitandbytes` please refer to this [readme](https://github.com/TimDettmers/bitsandbytes#requirements--installation).

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```

### Training on a 8 GB GPU:

By using [DeepSpeed](https://www.deepspeed.ai/) it's possible to offload some
tensors from VRAM to either CPU or NVME allowing to train with less VRAM.

DeepSpeed needs to be enabled with `accelerate config`. During configuration
answer yes to "Do you want to use DeepSpeed?". With DeepSpeed stage 2, fp16
mixed precision and offloading both parameters and optimizer state to cpu it's
possible to train on under 8 GB VRAM with a drawback of requiring significantly
more RAM (about 25 GB). See [documentation](https://huggingface.co/docs/accelerate/usage_guides/deepspeed) for more DeepSpeed configuration options.

Changing the default Adam optimizer to DeepSpeed's special version of Adam
`deepspeed.ops.adam.DeepSpeedCPUAdam` gives a substantial speedup but enabling
it requires CUDA toolchain with the same version as pytorch. 8-bit optimizer
does not seem to be compatible with DeepSpeed at the moment.

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch --mixed_precision="fp16" train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```

### Fine-tune text encoder with the UNet.

The script also allows to fine-tune the `text_encoder` along with the `unet`. It's been observed experimentally that fine-tuning `text_encoder` gives much better results especially on faces. 
Pass the `--train_text_encoder` argument to the script to enable training `text_encoder`.

___Note: Training text encoder requires more memory, with this option the training won't fit on 16GB GPU. It needs at least 24GB VRAM.___

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```

### Using DreamBooth for other pipelines than Stable Diffusion

Altdiffusion also support dreambooth now, the runing comman is basically the same as abouve, all you need to do is replace the `MODEL_NAME` like this:
One can now simply change the `pretrained_model_name_or_path` to another architecture such as [`AltDiffusion`](https://huggingface.co/docs/diffusers/api/pipelines/alt_diffusion).

```
export MODEL_NAME="CompVis/stable-diffusion-v1-4" --> export MODEL_NAME="BAAI/AltDiffusion-m9"
or
export MODEL_NAME="CompVis/stable-diffusion-v1-4" --> export MODEL_NAME="BAAI/AltDiffusion"
```

### Training with xformers:
You can enable memory efficient attention by [installing xFormers](https://github.com/facebookresearch/xformers#installing-xformers) and padding the `--enable_xformers_memory_efficient_attention` argument to the script. This is not available with the Flax/JAX implementation.

You can also use Dreambooth to train the specialized in-painting model. See [the script in the research folder for details](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/dreambooth_inpaint).