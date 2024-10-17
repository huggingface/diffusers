# Kandinsky2.2 text-to-image fine-tuning

Kandinsky 2.2 includes a prior pipeline that generates image embeddings from text prompts, and a decoder pipeline that generates the output image based on the image embeddings. We provide `train_text_to_image_prior.py` and `train_text_to_image_decoder.py` scripts to show you how to fine-tune the Kandinsky prior and decoder models separately based on your own dataset. To achieve the best results, you should fine-tune **_both_** your prior and decoder models.

___Note___:

___This script is experimental. The script fine-tunes the whole model and often times the model overfits and runs into issues like catastrophic forgetting. It's recommended to try different hyperparameters to get the best result on your dataset.___


## Running locally with PyTorch

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then cd in the example folder  and run
```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```
For this example we want to directly store the trained LoRA embeddings on the Hub, so we need to be logged in and add the --push_to_hub flag.

___

### Naruto example

For all our examples, we will directly store the trained weights on the Hub, so we need to be logged in and add the `--push_to_hub` flag. In order to do that, you have to be a registered user on the 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work. For more information on access tokens, please refer to the [User Access Tokens](https://huggingface.co/docs/hub/security-tokens) guide.

Run the following command to authenticate your token

```bash
huggingface-cli login
```

We also use [Weights and Biases](https://docs.wandb.ai/quickstart) logging by default, because it is really useful to monitor the training progress by regularly generating sample images during training. To install wandb, run

```bash
pip install wandb
```

To disable wandb logging, remove the `--report_to=="wandb"` and `--validation_prompts="A robot naruto, 4k photo"` flags from below examples

#### Fine-tune decoder
<br>

<!-- accelerate_snippet_start -->
```bash
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_decoder.py \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --validation_prompts="A robot naruto, 4k photo" \
  --report_to="wandb" \
  --push_to_hub \
  --output_dir="kandi2-decoder-naruto-model"
```
<!-- accelerate_snippet_end -->


To train on your own training files, prepare the dataset according to the format required by `datasets`. You can find the instructions for how to do that in the [ImageFolder with metadata](https://huggingface.co/docs/datasets/en/image_load#imagefolder-with-metadata) guide.
If you wish to use custom loading logic, you should modify the script and we have left pointers for that in the training script.

```bash
export TRAIN_DIR="path_to_your_dataset"

accelerate launch --mixed_precision="fp16" train_text_to_image_decoder.py \
  --train_data_dir=$TRAIN_DIR \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --validation_prompts="A robot naruto, 4k photo" \
  --report_to="wandb" \
  --push_to_hub \
  --output_dir="kandi22-decoder-naruto-model"
```


Once the training is finished the model will be saved in the `output_dir` specified in the command. In this example it's `kandi22-decoder-naruto-model`. To load the fine-tuned model for inference just pass that path to `AutoPipelineForText2Image`

```python
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained(output_dir, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

prompt='A robot naruto, 4k photo'
images = pipe(prompt=prompt).images
images[0].save("robot-naruto.png")
```

Checkpoints only save the unet, so to run inference from a checkpoint, just load the unet
```python
from diffusers import AutoPipelineForText2Image, UNet2DConditionModel

model_path = "path_to_saved_model"

unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-<N>/unet")

pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", unet=unet, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

image = pipe(prompt="A robot naruto, 4k photo").images[0]
image.save("robot-naruto.png")
```

#### Fine-tune prior

You can fine-tune the Kandinsky prior model with `train_text_to_image_prior.py` script. Note that we currently do not support `--gradient_checkpointing` for prior model fine-tuning.

<br>

<!-- accelerate_snippet_start -->
```bash
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_prior.py \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --validation_prompts="A robot naruto, 4k photo" \
  --report_to="wandb" \
  --push_to_hub \
  --output_dir="kandi2-prior-naruto-model"
```
<!-- accelerate_snippet_end -->


To perform inference with the fine-tuned prior model, you will need to first create a prior pipeline by passing the `output_dir` to `DiffusionPipeline`. Then create a `KandinskyV22CombinedPipeline` from a pretrained or fine-tuned decoder checkpoint along with all the modules of the prior pipeline you just created.

```python
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
import torch

pipe_prior = DiffusionPipeline.from_pretrained(output_dir, torch_dtype=torch.float16)
prior_components = {"prior_" + k: v for k,v in pipe_prior.components.items()}
pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", **prior_components, torch_dtype=torch.float16)

pipe.enable_model_cpu_offload()
prompt='A robot naruto, 4k photo'
images = pipe(prompt=prompt, negative_prompt=negative_prompt).images
images[0]
```

If you want to use a fine-tuned decoder checkpoint along with your fine-tuned prior checkpoint, you can simply replace the "kandinsky-community/kandinsky-2-2-decoder" in above code with your custom model repo name. Note that in order to be able to create a `KandinskyV22CombinedPipeline`, your model repository need to have a prior tag. If you have created your model repo using our training script, the prior tag is automatically included.

#### Training with multiple GPUs

`accelerate` allows for seamless multi-GPU training. Follow the instructions [here](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
for running distributed training with `accelerate`. Here is an example command:

```bash
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16" --multi_gpu  train_text_to_image_decoder.py \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --validation_prompts="A robot naruto, 4k photo" \
  --report_to="wandb" \
  --push_to_hub \
  --output_dir="kandi2-decoder-naruto-model"
```


#### Training with Min-SNR weighting

We support training with the Min-SNR weighting strategy proposed in [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556) which helps achieve faster convergence
by rebalancing the loss. Enable the `--snr_gamma` argument and set it to the recommended
value of 5.0.


## Training with LoRA

Low-Rank Adaption of Large Language Models was first introduced by Microsoft in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*.

In a nutshell, LoRA allows adapting pretrained models by adding pairs of rank-decomposition matrices to existing weights and **only** training those newly added weights. This has a couple of advantages:

- Previous pretrained weights are kept frozen so that model is not prone to [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114).
- Rank-decomposition matrices have significantly fewer parameters than original model, which means that trained LoRA weights are easily portable.
- LoRA attention layers allow to control to which extent the model is adapted toward new training images via a `scale` parameter.

[cloneofsimo](https://github.com/cloneofsimo) was the first to try out LoRA training for Stable Diffusion in the popular [lora](https://github.com/cloneofsimo/lora) GitHub repository.

With LoRA, it's possible to fine-tune Kandinsky 2.2 on a custom image-caption pair dataset
on consumer GPUs like Tesla T4, Tesla V100.

### Training

First, you need to set up your development environment as explained in the [installation](#installing-the-dependencies). Make sure to set the `MODEL_NAME` and `DATASET_NAME` environment variables. Here, we will use [Kandinsky 2.2](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) and the [Narutos dataset](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions).


#### Train decoder

```bash
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16" train_text_to_image_decoder_lora.py \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=768 \
  --train_batch_size=1 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --rank=4 \
  --gradient_checkpointing \
  --output_dir="kandi22-decoder-naruto-lora" \
  --validation_prompt="cute dragon creature" --report_to="wandb" \
  --push_to_hub \
```

#### Train prior

```bash
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16" train_text_to_image_prior_lora.py \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=768 \
  --train_batch_size=1 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --rank=4 \
  --output_dir="kandi22-prior-naruto-lora" \
  --validation_prompt="cute dragon creature" --report_to="wandb" \
  --push_to_hub \
```

**___Note: When using LoRA we can use a much higher learning rate compared to non-LoRA fine-tuning. Here we use *1e-4* instead of the usual *1e-5*. Also, by using LoRA, it's possible to run above scripts in consumer GPUs like T4 or V100.___**


### Inference

#### Inference using fine-tuned LoRA checkpoint for decoder

Once you have trained a Kandinsky decoder model using the above command, inference can be done with the `AutoPipelineForText2Image` after loading the trained LoRA weights.  You need to pass the `output_dir` for loading the LoRA weights, which in this case is `kandi22-decoder-naruto-lora`.


```python
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(output_dir)
pipe.enable_model_cpu_offload()

prompt='A robot naruto, 4k photo'
image = pipe(prompt=prompt).images[0]
image.save("robot_naruto.png")
```

#### Inference using fine-tuned LoRA checkpoint for prior

```python
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16)
pipe.prior_prior.load_attn_procs(output_dir)
pipe.enable_model_cpu_offload()

prompt='A robot naruto, 4k photo'
image = pipe(prompt=prompt).images[0]
image.save("robot_naruto.png")
image
```

### Training with xFormers:

You can enable memory efficient attention by [installing xFormers](https://huggingface.co/docs/diffusers/main/en/optimization/xformers) and passing the `--enable_xformers_memory_efficient_attention` argument to the script.

xFormers training is not available for fine-tuning the prior model.

**Note**:

According to [this issue](https://github.com/huggingface/diffusers/issues/2234#issuecomment-1416931212), xFormers `v0.0.16` cannot be used for training in some GPUs. If you observe that problem, please install a development version as indicated in that comment.