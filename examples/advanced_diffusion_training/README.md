# Advanced diffusion training examples

## Train Dreambooth LoRA with Stable Diffusion XL
> [!TIP]
> 💡 This example follows the techniques and recommended practices covered in the blog post: [LoRA training scripts of the world, unite!](https://huggingface.co/blog/sdxl_lora_advanced_script). Make sure to check it out before starting 🤗

[DreamBooth](https://huggingface.co/papers/2208.12242) is a method to personalize text2image models like stable diffusion given just a few(3~5) images of a subject.

LoRA - Low-Rank Adaption of Large Language Models, was first introduced by Microsoft in [LoRA: Low-Rank Adaptation of Large Language Models](https://huggingface.co/papers/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*
In a nutshell, LoRA allows to adapt pretrained models by adding pairs of rank-decomposition matrices to existing weights and **only** training those newly added weights. This has a couple of advantages:
- Previous pretrained weights are kept frozen so that the model is not prone to [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114)
- Rank-decomposition matrices have significantly fewer parameters than the original model, which means that trained LoRA weights are easily portable.
- LoRA attention layers allow to control to which extent the model is adapted towards new training images via a `scale` parameter.
[cloneofsimo](https://github.com/cloneofsimo) was the first to try out LoRA training for Stable Diffusion in
the popular [lora](https://github.com/cloneofsimo/lora) GitHub repository.

The `train_dreambooth_lora_sdxl_advanced.py` script shows how to implement dreambooth-LoRA, combining the training process shown in `train_dreambooth_lora_sdxl.py`, with
advanced features and techniques, inspired and built upon contributions by [Nataniel Ruiz](https://twitter.com/natanielruizg): [Dreambooth](https://dreambooth.github.io), [Rinon Gal](https://twitter.com/RinonGal): [Textual Inversion](https://textual-inversion.github.io), [Ron Mokady](https://twitter.com/MokadyRon): [Pivotal Tuning](https://huggingface.co/papers/2106.05744), [Simo Ryu](https://twitter.com/cloneofsimo): [cog-sdxl](https://github.com/replicate/cog-sdxl),
[Kohya](https://twitter.com/kohya_tech/): [sd-scripts](https://github.com/kohya-ss/sd-scripts), [The Last Ben](https://twitter.com/__TheBen): [fast-stable-diffusion](https://github.com/TheLastBen/fast-stable-diffusion) ❤️

> [!NOTE]
> 💡If this is your first time training a Dreambooth LoRA, congrats!🥳
> You might want to familiarize yourself more with the techniques: [Dreambooth blog](https://huggingface.co/blog/dreambooth), [Using LoRA for Efficient Stable Diffusion Fine-Tuning blog](https://huggingface.co/blog/lora)

📚 Read more about the advanced features and best practices in this community derived blog post: [LoRA training scripts of the world, unite!](https://huggingface.co/blog/sdxl_lora_advanced_script)


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

Then cd in the `examples/advanced_diffusion_training` folder and run
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
Note also that we use PEFT library as backend for LoRA training, make sure to have `peft>=0.6.0` installed in your environment.

Lastly, we recommend logging into your HF account so that your trained LoRA is automatically uploaded to the hub:
```bash
hf auth login
```
This command will prompt you for a token. Copy-paste yours from your [settings/tokens](https://huggingface.co/settings/tokens),and press Enter.

> [!NOTE]
> In the examples below we use `wandb` to document the training runs. To do the same, make sure to install `wandb`:
> `pip install wandb`
> Alternatively, you can use other tools / train without reporting by modifying the flag  `--report_to="wandb"`.

### Pivotal Tuning
**Training with text encoder(s)**

Alongside the UNet, LoRA fine-tuning of the text encoders is also supported. In addition to the text encoder optimization
available with `train_dreambooth_lora_sdxl_advanced.py`, in the advanced script **pivotal tuning** is also supported.
[pivotal tuning](https://huggingface.co/blog/sdxl_lora_advanced_script#pivotal-tuning) combines Textual Inversion with regular diffusion fine-tuning -
we insert new tokens into the text encoders of the model, instead of reusing existing ones.
We then optimize the newly-inserted token embeddings to represent the new concept.

To do so, just specify `--train_text_encoder_ti` while launching training (for regular text encoder optimizations, use `--train_text_encoder`).
Please keep the following points in mind:

* SDXL has two text encoders. So, we fine-tune both using LoRA.
* When not fine-tuning the text encoders, we ALWAYS precompute the text embeddings to save memory.

### 3D icon example

Now let's get our dataset. For this example we will use some cool images of 3d rendered icons: https://huggingface.co/datasets/linoyts/3d_icon.

Let's first download it locally:

```python
from huggingface_hub import snapshot_download

local_dir = "./3d_icon"
snapshot_download(
    "LinoyTsaban/3d_icon",
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

Let's review some of the advanced features we're going to be using for this example:
- **custom captions**:
To use custom captioning, first ensure that you have the datasets library installed, otherwise you can install it by
```bash
pip install datasets
```

Now we'll simply specify the name of the dataset and caption column (in this case it's "prompt")

```
--dataset_name=./3d_icon
--caption_column=prompt
```

You can also load a dataset straight from by specifying it's name in `dataset_name`.
Look [here](https://huggingface.co/blog/sdxl_lora_advanced_script#custom-captioning) for more info on creating/loading your own caption dataset.

- **optimizer**: for this example, we'll use [prodigy](https://huggingface.co/blog/sdxl_lora_advanced_script#adaptive-optimizers) - an adaptive optimizer
  - To use Prodigy, please make sure to install the prodigyopt library: `pip install prodigyopt`
- **pivotal tuning**
- **min SNR gamma**

**Now, we can launch training:**

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATASET_NAME="./3d_icon"
export OUTPUT_DIR="3d-icon-SDXL-LoRA"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

accelerate launch train_dreambooth_lora_sdxl_advanced.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --dataset_name=$DATASET_NAME \
  --instance_prompt="3d icon in the style of TOK" \
  --validation_prompt="a TOK icon of an astronaut riding a horse, in the style of TOK" \
  --output_dir=$OUTPUT_DIR \
  --caption_column="prompt" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=3 \
  --repeats=1 \
  --report_to="wandb"\
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1.0 \
  --text_encoder_lr=1.0 \
  --optimizer="prodigy"\
  --train_text_encoder_ti\
  --train_text_encoder_ti_frac=0.5\
  --snr_gamma=5.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank=8 \
  --max_train_steps=1000 \
  --checkpointing_steps=2000 \
  --seed="0" \
  --push_to_hub
```

To better track our training experiments, we're using the following flags in the command above:

* `report_to="wandb` will ensure the training runs are tracked on Weights and Biases. To use it, be sure to install `wandb` with `pip install wandb`.
* `validation_prompt` and `validation_epochs` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.

Our experiments were conducted on a single 40GB A100 GPU.


### Inference

Once training is done, we can perform inference like so:
1. starting with loading the unet lora weights
```python
import torch
from huggingface_hub import hf_hub_download, upload_file
from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL
from safetensors.torch import load_file

username = "linoyts"
repo_id = f"{username}/3d-icon-SDXL-LoRA"

pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
).to("cuda")


pipe.load_lora_weights(repo_id, weight_name="pytorch_lora_weights.safetensors")
```
2. now we load the pivotal tuning embeddings

```python
text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

embedding_path = hf_hub_download(repo_id=repo_id, filename="3d-icon-SDXL-LoRA_emb.safetensors", repo_type="model")

state_dict = load_file(embedding_path)
# load embeddings of text_encoder 1 (CLIP ViT-L/14)
pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
# load embeddings of text_encoder 2 (CLIP ViT-G/14)
pipe.load_textual_inversion(state_dict["clip_g"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
```

3. let's generate images

```python
instance_token = "<s0><s1>"
prompt = f"a {instance_token} icon of an orange llama eating ramen, in the style of {instance_token}"

image = pipe(prompt=prompt, num_inference_steps=25, cross_attention_kwargs={"scale": 1.0}).images[0]
image.save("llama.png")
```

### Comfy UI / AUTOMATIC1111 Inference
The new script fully supports textual inversion loading with Comfy UI and AUTOMATIC1111 formats!

**AUTOMATIC1111 / SD.Next** \
In AUTOMATIC1111/SD.Next we will load a LoRA and a textual embedding at the same time.
- *LoRA*: Besides the diffusers format, the script will also train a WebUI compatible LoRA. It is generated as `{your_lora_name}.safetensors`. You can then include it in your `models/Lora` directory.
- *Embedding*: the embedding is the same for diffusers and WebUI. You can download your `{lora_name}_emb.safetensors` file from a trained model, and include it in your `embeddings` directory.

You can then run inference by prompting `a y2k_emb webpage about the movie Mean Girls <lora:y2k:0.9>`. You can use the `y2k_emb` token normally, including increasing its weight by doing `(y2k_emb:1.2)`.

**ComfyUI** \
In ComfyUI we will load a LoRA and a textual embedding at the same time.
- *LoRA*: Besides the diffusers format, the script will also train a ComfyUI compatible LoRA. It is generated as `{your_lora_name}.safetensors`. You can then include it in your `models/Lora` directory. Then you will load the LoRALoader node and hook that up with your model and CLIP. [Official guide for loading LoRAs](https://comfyanonymous.github.io/ComfyUI_examples/lora/)
- *Embedding*: the embedding is the same for diffusers and WebUI. You can download your `{lora_name}_emb.safetensors` file from a trained model, and include it in your `models/embeddings` directory and use it in your prompts like `embedding:y2k_emb`. [Official guide for loading embeddings](https://comfyanonymous.github.io/ComfyUI_examples/textual_inversion_embeddings/).
-
### Specifying a better VAE

SDXL's VAE is known to suffer from numerical instability issues. This is why we also expose a CLI argument namely `--pretrained_vae_model_name_or_path` that lets you specify the location of a better VAE (such as [this one](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)).

### DoRA training
The advanced script supports DoRA training too!
> Proposed in [DoRA: Weight-Decomposed Low-Rank Adaptation](https://huggingface.co/papers/2402.09353),
**DoRA** is very similar to LoRA, except it decomposes the pre-trained weight into two components, **magnitude** and **direction** and employs LoRA for _directional_ updates to efficiently minimize the number of trainable parameters.
The authors found that by using DoRA, both the learning capacity and training stability of LoRA are enhanced without any additional overhead during inference.

> [!NOTE]
> 💡DoRA training is still _experimental_
> and is likely to require different hyperparameter values to perform best compared to a LoRA.
> Specifically, we've noticed 2 differences to take into account your training:
> 1. **LoRA seem to converge faster than DoRA** (so a set of parameters that may lead to overfitting when training a LoRA may be working well for a DoRA)
> 2. **DoRA quality superior to LoRA especially in lower ranks** the difference in quality of DoRA of rank 8 and LoRA of rank 8 appears to be more significant than when training ranks of 32 or 64 for example.
> This is also aligned with some of the quantitative analysis shown in the paper.

**Usage**
1. To use DoRA you need to install `peft` from main:
```bash
pip install git+https://github.com/huggingface/peft.git
```
2. Enable DoRA training by adding this flag
```bash
--use_dora
```
**Inference**
The inference is the same as if you train a regular LoRA 🤗

## Conducting EDM-style training

It's now possible to perform EDM-style training as proposed in [Elucidating the Design Space of Diffusion-Based Generative Models](https://huggingface.co/papers/2206.00364).

simply set:

```diff
+  --do_edm_style_training \
```

Other SDXL-like models that use the EDM formulation, such as [playgroundai/playground-v2.5-1024px-aesthetic](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic), can also be DreamBooth'd with the script. Below is an example command:

```bash
accelerate launch train_dreambooth_lora_sdxl_advanced.py \
  --pretrained_model_name_or_path="playgroundai/playground-v2.5-1024px-aesthetic"  \
  --dataset_name="linoyts/3d_icon" \
  --instance_prompt="3d icon in the style of TOK" \
  --validation_prompt="a TOK icon of an astronaut riding a horse, in the style of TOK" \
  --output_dir="3d-icon-SDXL-LoRA" \
  --do_edm_style_training \
  --caption_column="prompt" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=3 \
  --repeats=1 \
  --report_to="wandb"\
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1.0 \
  --text_encoder_lr=1.0 \
  --optimizer="prodigy"\
  --train_text_encoder_ti\
  --train_text_encoder_ti_frac=0.5\
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank=8 \
  --max_train_steps=1000 \
  --checkpointing_steps=2000 \
  --seed="0" \
  --push_to_hub
```

> [!CAUTION]
> Min-SNR gamma is not supported with the EDM-style training yet. When training with the PlaygroundAI model, it's recommended to not pass any "variant".

### B-LoRA training
The advanced script now supports B-LoRA training too!
> Proposed in [Implicit Style-Content Separation using B-LoRA](https://huggingface.co/papers/2403.14572),
B-LoRA is a method that leverages LoRA to implicitly separate the style and content components of a **single** image.
It was shown that learning the LoRA weights of two specific blocks (referred to as B-LoRAs)
achieves style-content separation that cannot be achieved by training each B-LoRA independently.
Once trained, the two B-LoRAs can be used as independent components to allow various image stylization tasks

**Usage**
Enable B-LoRA training by adding this flag
```bash
--use_blora
```
You can train a B-LoRA with as little as 1 image, and 1000 steps. Try this default configuration as a start:
```bash
!accelerate launch train_dreambooth_b-lora_sdxl.py \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
 --instance_data_dir="linoyts/B-LoRA_teddy_bear" \
 --output_dir="B-LoRA_teddy_bear" \
 --instance_prompt="a [v18]" \
 --resolution=1024 \
 --rank=64 \
 --train_batch_size=1 \
 --learning_rate=5e-5 \
 --lr_scheduler="constant" \
 --lr_warmup_steps=0 \
 --max_train_steps=1000 \
 --checkpointing_steps=2000 \
 --seed="0" \
 --gradient_checkpointing \
 --mixed_precision="fp16"
```
**Inference**
The inference is a bit different:
1. we need load *specific* unet layers (as opposed to a regular LoRA/DoRA)
2. the trained layers we load, changes based on our objective (e.g. style/content)

```python
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

# taken & modified from B-LoRA repo - https://github.com/yardenfren1996/B-LoRA/blob/main/blora_utils.py
def is_belong_to_blocks(key, blocks):
    try:
        for g in blocks:
            if g in key:
                return True
        return False
    except Exception as e:
        raise type(e)(f'failed to is_belong_to_block, due to: {e}')

def lora_lora_unet_blocks(lora_path, alpha, target_blocks):
  state_dict, _ = pipeline.lora_state_dict(lora_path)
  filtered_state_dict = {k: v * alpha for k, v in state_dict.items() if is_belong_to_blocks(k, target_blocks)}
  return filtered_state_dict

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda")

# pick a blora for content/style (you can also set one to None)
content_B_lora_path  = "lora-library/B-LoRA-teddybear"
style_B_lora_path= "lora-library/B-LoRA-pen_sketch"


content_B_LoRA = lora_lora_unet_blocks(content_B_lora_path,alpha=1,target_blocks=["unet.up_blocks.0.attentions.0"])
style_B_LoRA = lora_lora_unet_blocks(style_B_lora_path,alpha=1.1,target_blocks=["unet.up_blocks.0.attentions.1"])
combined_lora = {**content_B_LoRA, **style_B_LoRA}

# Load both loras
pipeline.load_lora_into_unet(combined_lora, None, pipeline.unet)

#generate
prompt = "a [v18] in [v30] style"
pipeline(prompt, num_images_per_prompt=4).images
```
### LoRA training of Targeted U-net Blocks
The advanced script now supports custom choice of U-net blocks to train during Dreambooth LoRA tuning.
> [!NOTE]
> This feature is still experimental

> Recently, works like B-LoRA showed the potential advantages of learning the LoRA weights of specific U-net blocks, not only in speed & memory,
> but also in reducing the amount of needed data, improving style manipulation and overcoming overfitting issues.
> In light of this, we're introducing a new feature to the advanced script to allow for configurable U-net learned blocks.

**Usage**
Configure LoRA learned U-net blocks adding a `lora_unet_blocks` flag, with a comma separated string specifying the targeted blocks.
e.g:
```bash
--lora_unet_blocks="unet.up_blocks.0.attentions.0,unet.up_blocks.0.attentions.1"
```

> [!NOTE]
> if you specify both `--use_blora` and `--lora_unet_blocks`, values given in --lora_unet_blocks will be ignored.
> When enabling --use_blora, targeted U-net blocks are automatically set to be "unet.up_blocks.0.attentions.0,unet.up_blocks.0.attentions.1" as discussed in the paper.
> If you wish to experiment with different blocks, specify `--lora_unet_blocks` only.

**Inference**
Inference is the same as for B-LoRAs, except the input targeted blocks should be modified based on your training configuration.
```python
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

# taken & modified from B-LoRA repo - https://github.com/yardenfren1996/B-LoRA/blob/main/blora_utils.py
def is_belong_to_blocks(key, blocks):
    try:
        for g in blocks:
            if g in key:
                return True
        return False
    except Exception as e:
        raise type(e)(f'failed to is_belong_to_block, due to: {e}')

def lora_lora_unet_blocks(lora_path, alpha, target_blocks):
  state_dict, _ = pipeline.lora_state_dict(lora_path)
  filtered_state_dict = {k: v * alpha for k, v in state_dict.items() if is_belong_to_blocks(k, target_blocks)}
  return filtered_state_dict

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda")

lora_path  = "lora-library/B-LoRA-pen_sketch"

state_dict = lora_lora_unet_blocks(content_B_lora_path,alpha=1,target_blocks=["unet.up_blocks.0.attentions.0"])

# Load trained lora layers into the unet
pipeline.load_lora_into_unet(state_dict, None, pipeline.unet)

#generate
prompt = "a dog in [v30] style"
pipeline(prompt, num_images_per_prompt=4).images
```


### Tips and Tricks
Check out [these recommended practices](https://huggingface.co/blog/sdxl_lora_advanced_script#additional-good-practices)

## Running on Colab Notebook
Check out [this notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_Dreambooth_LoRA_advanced_example.ipynb).
to train using the advanced features (including pivotal tuning), and [this notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_DreamBooth_LoRA_.ipynb) to train on a free colab, using some of the advanced features (excluding pivotal tuning)

