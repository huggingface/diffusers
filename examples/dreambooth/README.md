# DreamBooth training example

[DreamBooth](https://arxiv.org/abs/2208.12242) is a method to personalize text2image models like stable diffusion given just a few(3~5) images of a subject.
The `train_dreambooth.py` script shows how to implement the training procedure and adapt it for stable diffusion.


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

When running `accelerate config`, if we specify torch compile mode to True there can be dramatic speedups. 

### Dog toy example

Now let's get our dataset. For this example we will use some dog images: https://huggingface.co/datasets/diffusers/dog-example.

Let's first download it locally:

```python
from huggingface_hub import snapshot_download

local_dir = "./dog"
snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

And launch the training using:

**___Note: Change the `resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.___**

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="dog"
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
  --max_train_steps=400 \
  --push_to_hub
```

### Training with prior-preservation loss

Prior-preservation is used to avoid overfitting and language-drift. Refer to the paper to learn more about it. For prior-preservation we first generate images using the model with a class prompt and then use those during training along with our data.
According to the paper, it's recommended to generate `num_epochs * num_samples` images for prior-preservation. 200-300 works well for most cases. The `num_class_images` flag sets the number of images to generate with the class prompt. You can place existing images in `class_data_dir`, and the training script will generate any additional images so that `num_class_images` are present in `class_data_dir` during training time.

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="dog"
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
  --max_train_steps=800 \
  --push_to_hub
```


### Training on a 16GB GPU:

With the help of gradient checkpointing and the 8-bit optimizer from bitsandbytes it's possible to run train dreambooth on a 16GB GPU.

To install `bitsandbytes` please refer to this [readme](https://github.com/TimDettmers/bitsandbytes#requirements--installation).

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="dog"
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
  --max_train_steps=800 \
  --push_to_hub
```


### Training on a 12GB GPU:

It is possible to run dreambooth on a 12GB GPU by using the following optimizations:
- [gradient checkpointing and the 8-bit optimizer](#training-on-a-16gb-gpu)
- [xformers](#training-with-xformers)
- [setting grads to none](#set-grads-to-none)

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="dog"
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
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --push_to_hub
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
export INSTANCE_DIR="dog"
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
  --max_train_steps=800 \
  --push_to_hub
```

### Fine-tune text encoder with the UNet.

The script also allows to fine-tune the `text_encoder` along with the `unet`. It's been observed experimentally that fine-tuning `text_encoder` gives much better results especially on faces. 
Pass the `--train_text_encoder` argument to the script to enable training `text_encoder`.

___Note: Training text encoder requires more memory, with this option the training won't fit on 16GB GPU. It needs at least 24GB VRAM.___

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="dog"
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
  --max_train_steps=800 \
  --push_to_hub
```

### Using DreamBooth for pipelines other than Stable Diffusion

The [AltDiffusion pipeline](https://huggingface.co/docs/diffusers/api/pipelines/alt_diffusion) also supports dreambooth fine-tuning. The process is the same as above, all you need to do is replace the `MODEL_NAME` like this:

```
export MODEL_NAME="CompVis/stable-diffusion-v1-4" --> export MODEL_NAME="BAAI/AltDiffusion-m9"
or
export MODEL_NAME="CompVis/stable-diffusion-v1-4" --> export MODEL_NAME="BAAI/AltDiffusion"
```

### Inference

Once you have trained a model using the above command, you can run inference simply using the `StableDiffusionPipeline`. Make sure to include the `identifier` (e.g. sks in above example) in your prompt.

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "path-to-your-trained-model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of sks dog in a bucket"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("dog-bucket.png")
```

### Inference from a training checkpoint

You can also perform inference from one of the checkpoints saved during the training process, if you used the `--checkpointing_steps` argument. Please, refer to [the documentation](https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint) to see how to do it.

## Training with Low-Rank Adaptation of Large Language Models (LoRA)

Low-Rank Adaption of Large Language Models was first introduced by Microsoft in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*

In a nutshell, LoRA allows to adapt pretrained models by adding pairs of rank-decomposition matrices to existing weights and **only** training those newly added weights. This has a couple of advantages:
- Previous pretrained weights are kept frozen so that the model is not prone to [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114)
- Rank-decomposition matrices have significantly fewer parameters than the original model, which means that trained LoRA weights are easily portable.
- LoRA attention layers allow to control to which extent the model is adapted towards new training images via a `scale` parameter.

[cloneofsimo](https://github.com/cloneofsimo) was the first to try out LoRA training for Stable Diffusion in 
the popular [lora](https://github.com/cloneofsimo/lora) GitHub repository.

### Training

Let's get started with a simple example. We will re-use the dog example of the [previous section](#dog-toy-example).

First, you need to set-up your dreambooth training example as is explained in the [installation section](#Installing-the-dependencies).
Next, let's download the dog dataset. Download images from [here](https://drive.google.com/drive/folders/1BO_dyz-p65qhBRRMRA4TbZ8qW4rB99JZ) and save them in a directory. Make sure to set `INSTANCE_DIR` to the name of your directory further below. This will be our training data.

Now, you can launch the training. Here we will use [Stable Diffusion 1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5).

**___Note: Change the `resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.___**

**___Note: It is quite useful to monitor the training progress by regularly generating sample images during training. [wandb](https://docs.wandb.ai/quickstart) is a nice solution to easily see generating images during training. All you need to do is to run `pip install wandb` before training and pass `--report_to="wandb"` to automatically log images.___**


```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="path-to-save-model"
```

For this example we want to directly store the trained LoRA embeddings on the Hub, so 
we need to be logged in and add the `--push_to_hub` flag.

```bash
huggingface-cli login
```

Now we can start training!

```bash
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=50 \
  --seed="0" \
  --push_to_hub
```

**___Note: When using LoRA we can use a much higher learning rate compared to vanilla dreambooth. Here we 
use *1e-4* instead of the usual *2e-6*.___**

The final LoRA embedding weights have been uploaded to [patrickvonplaten/lora_dreambooth_dog_example](https://huggingface.co/patrickvonplaten/lora_dreambooth_dog_example). **___Note: [The final weights](https://huggingface.co/patrickvonplaten/lora/blob/main/pytorch_attn_procs.bin) are only 3 MB in size which is orders of magnitudes smaller than the original model.**

The training results are summarized [here](https://api.wandb.ai/report/patrickvonplaten/xm6cd5q5).
You can use the `Step` slider to see how the model learned the features of our subject while the model trained.

Optionally, we can also train additional LoRA layers for the text encoder. Specify the `--train_text_encoder` argument above for that. If you're interested to know more about how we
enable this support, check out this [PR](https://github.com/huggingface/diffusers/pull/2918). 

With the default hyperparameters from the above, the training seems to go in a positive direction. Check out [this panel](https://wandb.ai/sayakpaul/dreambooth-lora/reports/test-23-04-17-17-00-13---Vmlldzo0MDkwNjMy). The trained LoRA layers are available [here](https://huggingface.co/sayakpaul/dreambooth).


### Inference

After training, LoRA weights can be loaded very easily into the original pipeline. First, you need to 
load the original pipeline:

```python
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
```

Next, we can load the adapter layers into the UNet with the [`load_attn_procs` function](https://huggingface.co/docs/diffusers/api/loaders#diffusers.loaders.UNet2DConditionLoadersMixin.load_attn_procs).

```python
pipe.unet.load_attn_procs("patrickvonplaten/lora_dreambooth_dog_example")
```

Finally, we can run the model in inference.

```python
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
```

If you are loading the LoRA parameters from the Hub and if the Hub repository has
a `base_model` tag (such as [this](https://huggingface.co/patrickvonplaten/lora_dreambooth_dog_example/blob/main/README.md?code=true#L4)), then
you can do: 

```py 
from huggingface_hub.repocard import RepoCard

lora_model_id = "patrickvonplaten/lora_dreambooth_dog_example"
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
...
```

If you used `--train_text_encoder` during training, then use `pipe.load_lora_weights()` to load the LoRA
weights. For example:

```python
from huggingface_hub.repocard import RepoCard
from diffusers import StableDiffusionPipeline
import torch 

lora_model_id = "sayakpaul/dreambooth-text-encoder-test"
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_model_id)
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
```

Note that the use of [`LoraLoaderMixin.load_lora_weights`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.load_lora_weights) is preferred to [`UNet2DConditionLoadersMixin.load_attn_procs`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.UNet2DConditionLoadersMixin.load_attn_procs) for loading LoRA parameters. This is because
`LoraLoaderMixin.load_lora_weights` can handle the following situations:

* LoRA parameters that don't have separate identifiers for the UNet and the text encoder (such as [`"patrickvonplaten/lora_dreambooth_dog_example"`](https://huggingface.co/patrickvonplaten/lora_dreambooth_dog_example)). So, you can just do:

  ```py 
  pipe.load_lora_weights(lora_model_path)
  ```

* LoRA parameters that have separate identifiers for the UNet and the text encoder such as: [`"sayakpaul/dreambooth"`](https://huggingface.co/sayakpaul/dreambooth).

## Training with Flax/JAX

For faster training on TPUs and GPUs you can leverage the flax training example. Follow the instructions above to get the model and dataset before running the script.

____Note: The flax example don't yet support features like gradient checkpoint, gradient accumulation etc, so to use flax for faster training we will need >30GB cards.___


Before running the scripts, make sure to install the library's training dependencies:

```bash
pip install -U -r requirements_flax.txt
```


### Training without prior preservation loss

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --max_train_steps=400
```


### Training with prior preservation loss

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --num_class_images=200 \
  --max_train_steps=800
```


### Fine-tune text encoder with the UNet.

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py \
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
  --learning_rate=2e-6 \
  --num_class_images=200 \
  --max_train_steps=800
```

### Training with xformers:
You can enable memory efficient attention by [installing xFormers](https://github.com/facebookresearch/xformers#installing-xformers) and padding the `--enable_xformers_memory_efficient_attention` argument to the script. This is not available with the Flax/JAX implementation.

You can also use Dreambooth to train the specialized in-painting model. See [the script in the research folder for details](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/dreambooth_inpaint).

### Set grads to none

To save even more memory, pass the `--set_grads_to_none` argument to the script. This will set grads to None instead of zero. However, be aware that it changes certain behaviors, so if you start experiencing any problems, remove this argument.

More info: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html

### Experimental results
You can refer to [this blog post](https://huggingface.co/blog/dreambooth) that discusses some of DreamBooth experiments in detail. Specifically, it recommends a set of DreamBooth-specific tips and tricks that we have found to work well for a variety of subjects. 

## IF

You can use the lora and full dreambooth scripts to train the text to image [IF model](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0) and the stage II upscaler 
[IF model](https://huggingface.co/DeepFloyd/IF-II-L-v1.0).

Note that IF has a predicted variance, and our finetuning scripts only train the models predicted error, so for finetuned IF models we switch to a fixed
variance schedule. The full finetuning scripts will update the scheduler config for the full saved model. However, when loading saved LoRA weights, you
must also update the pipeline's scheduler config.

```py
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0")

pipe.load_lora_weights("<lora weights path>")

# Update scheduler config to fixed variance schedule
pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, variance_type="fixed_small")
```

Additionally, a few alternative cli flags are needed for IF.

`--resolution=64`: IF is a pixel space diffusion model. In order to operate on un-compressed pixels, the input images are of a much smaller resolution. 

`--pre_compute_text_embeddings`: IF uses [T5](https://huggingface.co/docs/transformers/model_doc/t5) for its text encoder. In order to save GPU memory, we pre compute all text embeddings and then de-allocate
T5.

`--tokenizer_max_length=77`: T5 has a longer default text length, but the default IF encoding procedure uses a smaller number.

`--text_encoder_use_attention_mask`: T5 passes the attention mask to the text encoder.

### Tips and Tricks
We find LoRA to be sufficient for finetuning the stage I model as the low resolution of the model makes representing finegrained detail hard regardless.

For common and/or not-visually complex object concepts, you can get away with not-finetuning the upscaler. Just be sure to adjust the prompt passed to the
upscaler to remove the new token from the instance prompt. I.e. if your stage I prompt is "a sks dog", use "a dog" for your stage II prompt.

For finegrained detail like faces that aren't present in the original training set, we find that full finetuning of the stage II upscaler is better than 
LoRA finetuning stage II.

For finegrained detail like faces, we find that lower learning rates along with larger batch sizes work best.

For stage II, we find that lower learning rates are also needed.

We found experimentally that the DDPM scheduler with the default larger number of denoising steps to sometimes work better than the DPM Solver scheduler
used in the training scripts.

### Stage II additional validation images

The stage II validation requires images to upscale, we can download a downsized version of the training set:

```py
from huggingface_hub import snapshot_download

local_dir = "./dog_downsized"
snapshot_download(
    "diffusers/dog-example-downsized",
    local_dir=local_dir,
    repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

### IF stage I LoRA Dreambooth
This training configuration requires ~28 GB VRAM.

```sh
export MODEL_NAME="DeepFloyd/IF-I-XL-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_dog_lora"

accelerate launch train_dreambooth_lora.py \
  --report_to wandb \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks dog" \
  --resolution=64 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --scale_lr \
  --max_train_steps=1200 \
  --validation_prompt="a sks dog" \
  --validation_epochs=25 \
  --checkpointing_steps=100 \
  --pre_compute_text_embeddings \
  --tokenizer_max_length=77 \
  --text_encoder_use_attention_mask
```

### IF stage II LoRA Dreambooth

`--validation_images`: These images are upscaled during validation steps.

`--class_labels_conditioning=timesteps`: Pass additional conditioning to the UNet needed for stage II.

`--learning_rate=1e-6`: Lower learning rate than stage I.

`--resolution=256`: The upscaler expects higher resolution inputs

```sh
export MODEL_NAME="DeepFloyd/IF-II-L-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_dog_upscale"
export VALIDATION_IMAGES="dog_downsized/image_1.png dog_downsized/image_2.png dog_downsized/image_3.png dog_downsized/image_4.png"

python train_dreambooth_lora.py \
    --report_to wandb \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --instance_prompt="a sks dog" \
    --resolution=256 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-6 \ 
    --max_train_steps=2000 \
    --validation_prompt="a sks dog" \
    --validation_epochs=100 \
    --checkpointing_steps=500 \
    --pre_compute_text_embeddings \
    --tokenizer_max_length=77 \
    --text_encoder_use_attention_mask \
    --validation_images $VALIDATION_IMAGES \
    --class_labels_conditioning=timesteps
```

### IF Stage I Full Dreambooth
`--skip_save_text_encoder`: When training the full model, this will skip saving the entire T5 with the finetuned model. You can still load the pipeline
with a T5 loaded from the original model.

`use_8bit_adam`: Due to the size of the optimizer states, we recommend training the full XL IF model with 8bit adam. 

`--learning_rate=1e-7`: For full dreambooth, IF requires very low learning rates. With higher learning rates model quality will degrade. Note that it is 
likely the learning rate can be increased with larger batch sizes.

Using 8bit adam and a batch size of 4, the model can be trained in ~48 GB VRAM.

```sh
export MODEL_NAME="DeepFloyd/IF-I-XL-v1.0"

export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_if"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=64 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-7 \
  --max_train_steps=150 \
  --validation_prompt "a photo of sks dog" \
  --validation_steps 25 \
  --text_encoder_use_attention_mask \
  --tokenizer_max_length 77 \
  --pre_compute_text_embeddings \
  --use_8bit_adam \
  --set_grads_to_none \
  --skip_save_text_encoder \
  --push_to_hub
```

### IF Stage II Full Dreambooth

`--learning_rate=5e-6`: With a smaller effective batch size of 4, we found that we required learning rates as low as
1e-8.

`--resolution=256`: The upscaler expects higher resolution inputs

`--train_batch_size=2` and `--gradient_accumulation_steps=6`: We found that full training of stage II particularly with
faces required large effective batch sizes.

```sh
export MODEL_NAME="DeepFloyd/IF-II-L-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_dog_upscale"
export VALIDATION_IMAGES="dog_downsized/image_1.png dog_downsized/image_2.png dog_downsized/image_3.png dog_downsized/image_4.png"

accelerate launch train_dreambooth.py \
  --report_to wandb \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks dog" \
  --resolution=256 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=6 \
  --learning_rate=5e-6 \
  --max_train_steps=2000 \
  --validation_prompt="a sks dog" \
  --validation_steps=150 \
  --checkpointing_steps=500 \
  --pre_compute_text_embeddings \
  --tokenizer_max_length=77 \
  --text_encoder_use_attention_mask \
  --validation_images $VALIDATION_IMAGES \
  --class_labels_conditioning timesteps \
  --push_to_hub
```

## Stable Diffusion XL

We support fine-tuning of the UNet shipped in [Stable Diffusion XL](https://huggingface.co/papers/2307.01952) with DreamBooth and LoRA via the `train_dreambooth_lora_sdxl.py` script. Please refer to the docs [here](./README_sdxl.md). 
