# Custom Diffusion training example

[Custom Diffusion](https://huggingface.co/papers/2212.04488) is a method to customize text-to-image models like Stable Diffusion given just a few (4~5) images of a subject.
The `train_custom_diffusion.py` script shows how to implement the training procedure and adapt it for stable diffusion.

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
### Cat example 😺

Now let's get our dataset. Download dataset from [here](https://www.cs.cmu.edu/~custom-diffusion/assets/data.zip) and unzip it.

We also collect 200 real images using `clip-retrieval` which are combined with the target images in the training dataset as a regularization. This prevents overfitting to the given target image. The following flags enable the regularization `with_prior_preservation`, `real_prior` with `prior_loss_weight=1.`.
The `class_prompt` should be the category name same as target image. The collected real images are with text captions similar to the `class_prompt`. The retrieved image are saved in `class_data_dir`. You can disable `real_prior` to use generated images as regularization. To collect the real images use this command first before training.

```bash
pip install clip-retrieval
python retrieve.py --class_prompt cat --class_data_dir real_reg/samples_cat --num_class_images 200
```

**___Note: Change the `resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.___**

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"
export INSTANCE_DIR="./data/cat"

accelerate launch train_custom_diffusion.py \
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

**Use `--enable_xformers_memory_efficient_attention` for faster training with lower VRAM requirement (16GB per GPU). Follow [this guide](https://github.com/facebookresearch/xformers) for installation instructions.**

To track your experiments using Weights and Biases (`wandb`) and to save intermediate results (which we HIGHLY recommend), follow these steps:

* Install `wandb`: `pip install wandb`.
* Authorize: `wandb login`.
* Then specify a `validation_prompt` and set `report_to` to `wandb` while launching training. You can also configure the following related arguments:
    * `num_validation_images`
    * `validation_steps`

Here is an example command:

```bash
accelerate launch train_custom_diffusion.py \
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
  --modifier_token "<new1>" \
  --validation_prompt="<new1> cat sitting in a bucket" \
  --report_to="wandb"
```

Here is an example [Weights and Biases page](https://wandb.ai/sayakpaul/custom-diffusion/runs/26ghrcau) where you can check out the intermediate results along with other training details.

If you specify `--push_to_hub`, the learned parameters will be pushed to a repository on the Hugging Face Hub. Here is an [example repository](https://huggingface.co/sayakpaul/custom-diffusion-cat).

### Training on multiple concepts 🐱🪵

Provide a [json](https://github.com/adobe-research/custom-diffusion/blob/main/assets/concept_list.json) file with the info about each concept, similar to [this](https://github.com/ShivamShrirao/diffusers/blob/main/examples/dreambooth/train_dreambooth.py).

To collect the real images run this command for each concept in the json file.

```bash
pip install clip-retrieval
python retrieve.py --class_prompt {} --class_data_dir {} --num_class_images 200
```

And then we're ready to start training!

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_custom_diffusion.py \
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

Here is an example [Weights and Biases page](https://wandb.ai/sayakpaul/custom-diffusion/runs/3990tzkg) where you can check out the intermediate results along with other training details.

### Training on human faces

For fine-tuning on human faces we found the following configuration to work better: `learning_rate=5e-6`, `max_train_steps=1000 to 2000`, and `freeze_model=crossattn` with at least 15-20 images.

To collect the real images use this command first before training.

```bash
pip install clip-retrieval
python retrieve.py --class_prompt person --class_data_dir real_reg/samples_person --num_class_images 200
```

Then start training!

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"
export INSTANCE_DIR="path-to-images"

accelerate launch train_custom_diffusion.py \
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
  --enable_xformers_memory_efficient_attention
```

## Inference

Once you have trained a model using the above command, you can run inference using the below command. Make sure to include the `modifier token` (e.g. \<new1\> in above example) in your prompt.

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to("cuda")
pipe.unet.load_attn_procs(
    "path-to-save-model", weight_name="pytorch_custom_diffusion_weights.bin"
)
pipe.load_textual_inversion("path-to-save-model", weight_name="<new1>.bin")

image = pipe(
    "<new1> cat sitting in a bucket",
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images[0]
image.save("cat.png")
```

It's possible to directly load these parameters from a Hub repository:

```python
import torch
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline

model_id = "sayakpaul/custom-diffusion-cat"
card = RepoCard.load(model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to(
"cuda")
pipe.unet.load_attn_procs(model_id, weight_name="pytorch_custom_diffusion_weights.bin")
pipe.load_textual_inversion(model_id, weight_name="<new1>.bin")

image = pipe(
    "<new1> cat sitting in a bucket",
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images[0]
image.save("cat.png")
```

Here is an example of performing inference with multiple concepts:

```python
import torch
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline

model_id = "sayakpaul/custom-diffusion-cat-wooden-pot"
card = RepoCard.load(model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to(
"cuda")
pipe.unet.load_attn_procs(model_id, weight_name="pytorch_custom_diffusion_weights.bin")
pipe.load_textual_inversion(model_id, weight_name="<new1>.bin")
pipe.load_textual_inversion(model_id, weight_name="<new2>.bin")

image = pipe(
    "the <new1> cat sculpture in the style of a <new2> wooden pot",
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images[0]
image.save("multi-subject.png")
```

Here, `cat` and `wooden pot` refer to the multiple concepts.

### Inference from a training checkpoint

You can also perform inference from one of the complete checkpoint saved during the training process, if you used the `--checkpointing_steps` argument.

TODO.

## Set grads to none
To save even more memory, pass the `--set_grads_to_none` argument to the script. This will set grads to None instead of zero. However, be aware that it changes certain behaviors, so if you start experiencing any problems, remove this argument.

More info: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html

## Experimental results
You can refer to [our webpage](https://www.cs.cmu.edu/~custom-diffusion/) that discusses our experiments in detail. We also released a more extensive dataset of 101 concepts for evaluating model customization methods. For more details please refer to our [dataset webpage](https://www.cs.cmu.edu/~custom-diffusion/dataset.html).