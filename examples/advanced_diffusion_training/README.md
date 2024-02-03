# Advanced diffusion training examples

## Train Dreambooth LoRA with Stable Diffusion XL
> [!TIP]
> 💡 This example follows the techniques and recommended practices covered in the blog post: [LoRA training scripts of the world, unite!](https://huggingface.co/blog/sdxl_lora_advanced_script). Make sure to check it out before starting 🤗

[DreamBooth](https://arxiv.org/abs/2208.12242) is a method to personalize text2image models like stable diffusion given just a few(3~5) images of a subject.

LoRA - Low-Rank Adaption of Large Language Models, was first introduced by Microsoft in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*
In a nutshell, LoRA allows to adapt pretrained models by adding pairs of rank-decomposition matrices to existing weights and **only** training those newly added weights. This has a couple of advantages:
- Previous pretrained weights are kept frozen so that the model is not prone to [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114)
- Rank-decomposition matrices have significantly fewer parameters than the original model, which means that trained LoRA weights are easily portable.
- LoRA attention layers allow to control to which extent the model is adapted towards new training images via a `scale` parameter.
[cloneofsimo](https://github.com/cloneofsimo) was the first to try out LoRA training for Stable Diffusion in 
the popular [lora](https://github.com/cloneofsimo/lora) GitHub repository.

The `train_dreambooth_lora_sdxl_advanced.py` script shows how to implement dreambooth-LoRA, combining the training process shown in `train_dreambooth_lora_sdxl.py`, with 
advanced features and techniques, inspired and built upon contributions by [Nataniel Ruiz](https://twitter.com/natanielruizg): [Dreambooth](https://dreambooth.github.io), [Rinon Gal](https://twitter.com/RinonGal): [Textual Inversion](https://textual-inversion.github.io), [Ron Mokady](https://twitter.com/MokadyRon): [Pivotal Tuning](https://arxiv.org/abs/2106.05744), [Simo Ryu](https://twitter.com/cloneofsimo): [cog-sdxl](https://github.com/replicate/cog-sdxl), 
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
* When not fine-tuning the text encoders, we ALWAYS precompute the text embeddings to save memoםהקרry.


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
Look [here](https://huggingface.co/blog/sdxl_lora_advanced_script#custom-captioning) for more info on creating/loadin your own caption dataset.

- **optimizer**: for this example, we'll use [prodigy](https://huggingface.co/blog/sdxl_lora_advanced_script#adaptive-optimizers) - an adaptive optimizer
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


### Tips and Tricks
Check out [these recommended practices](https://huggingface.co/blog/sdxl_lora_advanced_script#additional-good-practices)

## Running on Colab Notebook
Check out [this notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_DreamBooth_LoRA_advanced_example.ipynb). 
to train using the advanced features (including pivotal tuning), and [this notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_DreamBooth_LoRA_.ipynb) to train on a free colab, using some of the advanced features (excluding pivotal tuning)

