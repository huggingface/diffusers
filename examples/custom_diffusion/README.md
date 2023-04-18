# Custom Diffusion training example 
(modified from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README.md)

[Custom Diffusion](https://arxiv.org/abs/2212.04488) is a method to customize text2image models like stable diffusion given just a few(4~5) images of a subject.
The `train.py` script shows how to implement the training procedure and adapt it for stable diffusion.


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
pip install clip-retrieval 
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

### Cat example

Now let's get our dataset. Download dataset from [here](https://www.cs.cmu.edu/~custom-diffusion/assets/data.zip) and unzip it. 

We also collect 200 real images using `clip-retrieval` which are combined with the target images in the training dataset as a regularization. This prevents overfitting to the the given target image. The following flags enable the regularization `with_prior_preservation`, `real_prior` with `prior_loss_weight=1.`. 
The `class_prompt` should be the category name same as target image. The collected real images are with text captions similar to the `class_prompt`. The retrieved image are saved in `class_data_dir`. You can disable `real_prior` to use generated images as regularization. To collect the real images use this command first before training. 

```
pip install clip-retrieval
python retrieve.py --class_prompt cat --class_data_dir real_reg/samples_cat --num_class_images 200
```

**___Note: Change the `resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.___**

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"
export INSTANCE_DIR="./data/cat"
## launch training script (2 GPUs recommended, increase --max_train_steps to 500 if 1 GPU, or increase --train_batch_size=4)

accelerate launch train.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=$INSTANCE_DIR \
          --output_dir=$OUTPUT_DIR \
          --class_data_dir=./real_reg/samples_cat/ \
          --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
          --class_prompt="cat" --num_class_images=200 \
          --instance_prompt="photo of a <new1> cat"  \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=250 \
          --scale_lr --hflip  \
          --modifier_token "<new1>" 
```

**Use `--enable_xformers_memory_efficient_attention` for faster training with lower VRAM requirement (16GB per GPU).**


### Training on multiple concepts

Provide a [json](https://github.com/adobe-research/custom-diffusion/blob/main/assets/concept_list.json) file with the info about each concept, similar to [this](https://github.com/ShivamShrirao/diffusers/blob/main/examples/dreambooth/train_dreambooth.py).

To collect the real images run this command for each concept in the json file. 

```
pip install clip-retrieval
python retrieve.py --class_prompt {} --class_data_dir {} --num_class_images 200
```


```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"

## launch training script (2 GPUs recommended, increase --max_train_steps to 1000 if 1 GPU, or increase --train_batch_size=4)

accelerate launch train.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --output_dir=$OUTPUT_DIR \
          --concepts_list=./concept_list.json \
          --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=500 \
          --num_class_images=200 \
          --scale_lr --hflip  \
          --modifier_token "<new1>+<new2>" 
```

### Training on human faces

For fine-tuning on human faces we found the following configuration to work better: `learning_rate=5e-6`, `max_train_steps=1000 to 2000`, and `freeze_model=crossattn` with atleast 15-20 images. 
To collect the real images use this command first before training. 

```
pip install clip-retrieval
python retrieve.py --class_prompt person --class_data_dir real_reg/samples_person --num_class_images 200
```

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"
export INSTANCE_DIR="path-to-images"

## launch training script (2 GPUs recommended, increase --max_train_steps to 1000 if 1 GPU, or increase --train_batch_size=4)

CUDA_VISIBLE_DEVICES=1 accelerate launch train.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=$INSTANCE_DIR \
          --output_dir=$OUTPUT_DIR \
          --class_data_dir=./real_reg/samples_person/ \
          --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
          --class_prompt="person" --num_class_images=200 \
          --instance_prompt="photo of a <new1> person"  \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=5e-6  \
          --lr_warmup_steps=0 \
          --max_train_steps=1000 \
          --scale_lr --hflip --noaug \
          --freeze_model crossattn \
          --modifier_token "<new1>" \
          --enable_xformers_memory_efficient_attention \
```

### Inference

Once you have trained a model using the above command, you can run inference using the below command. Make sure to include the `modifier token` (e.g. \<new1\> in above example) in your prompt.

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
pipe.unet.load_attn_procs('path-to-save-model', weight_name='pytorch_custom_diffusion_weights.bin')
pipe.load_textual_inversion('path-to-save-model', weight_name='<new1>.bin')
image = pipe("<new1> cat sitting in a bucket", num_inference_steps=100, guidance_scale=6., eta=1.).images[0]

image.save("cat.png")
```

### Inference from a training checkpoint

You can also perform inference from one of the complete checkpoint saved during the training process, if you used the `--checkpointing_steps` argument. 

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained('path-to-the-model/checkpoint-250/', torch_dtype=torch.float16).to("cuda")
image = pipe("<new1> cat sitting in a bucket", num_inference_steps=100, guidance_scale=6., eta=1.).images[0]

image.save("cat.png")
```

### Converting delta.bin to diffusers pipeline

You can also perform inference from one of the complete checkpoint saved during the training process, if you used the `--checkpointing_steps` argument. 

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
pipe.unet.load_attn_procs('path-to-save-model', weight_name='pytorch_custom_diffusion_weights.bin')
pipe.load_textual_inversion('path-to-save-model', weight_name='<new1>.bin')
pipe.save_pretrained('<path-to-your-save-model>')
```

### Set grads to none

To save even more memory, pass the `--set_grads_to_none` argument to the script. This will set grads to None instead of zero. However, be aware that it changes certain behaviors, so if you start experiencing any problems, remove this argument.

More info: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html

### Experimental results
You can refer to [our webpage](https://www.cs.cmu.edu/~custom-diffusion/) that discusses our experiments in detail. 