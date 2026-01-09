# Advanced diffusion training examples

## Train Dreambooth LoRA with Flux.1 Dev
> [!TIP]
> 💡 This example follows some of the techniques and recommended practices covered in the community derived guide we made for SDXL training: [LoRA training scripts of the world, unite!](https://huggingface.co/blog/sdxl_lora_advanced_script). 
> As many of these are architecture agnostic & generally relevant to fine-tuning of diffusion models we suggest to take a look 🤗

[DreamBooth](https://huggingface.co/papers/2208.12242) is a method to personalize text-to-image models like flux, stable diffusion given just a few(3~5) images of a subject.

LoRA - Low-Rank Adaption of Large Language Models, was first introduced by Microsoft in [LoRA: Low-Rank Adaptation of Large Language Models](https://huggingface.co/papers/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*
In a nutshell, LoRA allows to adapt pretrained models by adding pairs of rank-decomposition matrices to existing weights and **only** training those newly added weights. This has a couple of advantages:
- Previous pretrained weights are kept frozen so that the model is not prone to [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114)
- Rank-decomposition matrices have significantly fewer parameters than the original model, which means that trained LoRA weights are easily portable.
- LoRA attention layers allow to control to which extent the model is adapted towards new training images via a `scale` parameter.
[cloneofsimo](https://github.com/cloneofsimo) was the first to try out LoRA training for Stable Diffusion in
the popular [lora](https://github.com/cloneofsimo/lora) GitHub repository.

The `train_dreambooth_lora_flux_advanced.py` script shows how to implement dreambooth-LoRA, combining the training process shown in `train_dreambooth_lora_flux.py`, with
advanced features and techniques, inspired and built upon contributions by [Nataniel Ruiz](https://twitter.com/natanielruizg): [Dreambooth](https://dreambooth.github.io), [Rinon Gal](https://twitter.com/RinonGal): [Textual Inversion](https://textual-inversion.github.io), [Ron Mokady](https://twitter.com/MokadyRon): [Pivotal Tuning](https://huggingface.co/papers/2106.05744), [Simo Ryu](https://twitter.com/cloneofsimo): [cog-sdxl](https://github.com/replicate/cog-sdxl),
[ostris](https://x.com/ostrisai):[ai-toolkit](https://github.com/ostris/ai-toolkit), [bghira](https://github.com/bghira):[SimpleTuner](https://github.com/bghira/SimpleTuner), [Kohya](https://twitter.com/kohya_tech/): [sd-scripts](https://github.com/kohya-ss/sd-scripts), [The Last Ben](https://twitter.com/__TheBen): [fast-stable-diffusion](https://github.com/TheLastBen/fast-stable-diffusion) ❤️

> [!NOTE]
> 💡If this is your first time training a Dreambooth LoRA, congrats!🥳
> You might want to familiarize yourself more with the techniques: [Dreambooth blog](https://huggingface.co/blog/dreambooth), [Using LoRA for Efficient Stable Diffusion Fine-Tuning blog](https://huggingface.co/blog/lora)

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

### LoRA Rank and Alpha
Two key LoRA hyperparameters are LoRA rank and LoRA alpha. 
- `--rank`: Defines the dimension of the trainable LoRA matrices. A higher rank means more expressiveness and capacity to learn (and more parameters).
- `--lora_alpha`: A scaling factor for the LoRA's output. The LoRA update is scaled by lora_alpha / lora_rank.
- lora_alpha vs. rank:
This ratio dictates the LoRA's effective strength:
lora_alpha == rank: Scaling factor is 1. The LoRA is applied with its learned strength. (e.g., alpha=16, rank=16)
lora_alpha < rank: Scaling factor < 1. Reduces the LoRA's impact. Useful for subtle changes or to prevent overpowering the base model. (e.g., alpha=8, rank=16)
lora_alpha > rank: Scaling factor > 1. Amplifies the LoRA's impact. Allows a lower rank LoRA to have a stronger effect. (e.g., alpha=32, rank=16)

> [!TIP]
> A common starting point is to set `lora_alpha` equal to `rank`. 
> Some also set `lora_alpha` to be twice the `rank` (e.g., lora_alpha=32 for lora_rank=16) 
> to give the LoRA updates more influence without increasing parameter count. 
> If you find your LoRA is "overcooking" or learning too aggressively, consider setting `lora_alpha` to half of `rank` 
> (e.g., lora_alpha=8 for rank=16). Experimentation is often key to finding the optimal balance for your use case.


### Target Modules
When LoRA was first adapted from language models to diffusion models, it was applied to the cross-attention layers in the Unet that relate the image representations with the prompts that describe them. 
More recently, SOTA text-to-image diffusion models replaced the Unet with a diffusion Transformer(DiT). With this change, we may also want to explore 
applying LoRA training onto different types of layers and blocks. To allow more flexibility and control over the targeted modules we added `--lora_layers`- in which you can specify in a comma separated string
the exact modules for LoRA training. Here are some examples of target modules you can provide: 
- for attention only layers: `--lora_layers="attn.to_k,attn.to_q,attn.to_v,attn.to_out.0"`
- to train the same modules as in the fal trainer: `--lora_layers="attn.to_k,attn.to_q,attn.to_v,attn.to_out.0,attn.add_k_proj,attn.add_q_proj,attn.add_v_proj,attn.to_add_out,ff.net.0.proj,ff.net.2,ff_context.net.0.proj,ff_context.net.2"`
- to train the same modules as in ostris ai-toolkit / replicate trainer: `--lora_blocks="attn.to_k,attn.to_q,attn.to_v,attn.to_out.0,attn.add_k_proj,attn.add_q_proj,attn.add_v_proj,attn.to_add_out,ff.net.0.proj,ff.net.2,ff_context.net.0.proj,ff_context.net.2,norm1_context.linear, norm1.linear,norm.linear,proj_mlp,proj_out"`
> [!NOTE]
> `--lora_layers` can also be used to specify which **blocks** to apply LoRA training to. To do so, simply add a block prefix to each layer in the comma separated string:
> **single DiT blocks**: to target the ith single transformer block, add the prefix `single_transformer_blocks.i`, e.g. - `single_transformer_blocks.i.attn.to_k`
> **MMDiT blocks**: to target the ith MMDiT block, add the prefix `transformer_blocks.i`, e.g. - `transformer_blocks.i.attn.to_k` 
> [!NOTE]
> keep in mind that while training more layers can improve quality and expressiveness, it also increases the size of the output LoRA weights.

### Pivotal Tuning (and more)
**Training with text encoder(s)**

Alongside the Transformer, LoRA fine-tuning of the text encoders is also supported. In addition to the text encoder optimization
available with `train_dreambooth_lora_flux_advanced.py`, in the advanced script **pivotal tuning** is also supported.
[pivotal tuning](https://huggingface.co/blog/sdxl_lora_advanced_script#pivotal-tuning) combines Textual Inversion with regular diffusion fine-tuning -
we insert new tokens into the text encoders of the model, instead of reusing existing ones.
We then optimize the newly-inserted token embeddings to represent the new concept.

To do so, just specify `--train_text_encoder_ti` while launching training (for regular text encoder optimizations, use `--train_text_encoder`).
Please keep the following points in mind:

* Flux uses two text encoders - [CLIP](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux#diffusers.FluxPipeline.text_encoder) & [T5](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux#diffusers.FluxPipeline.text_encoder_2) , by default `--train_text_encoder_ti` performs pivotal tuning for the **CLIP** encoder only.
To activate pivotal tuning for both encoders, add the flag `--enable_t5_ti`. 
* When not fine-tuning the text encoders, we ALWAYS precompute the text embeddings to save memory.
* **pure textual inversion** - to support the full range from pivotal tuning to textual inversion we introduce `--train_transformer_frac` which controls the amount of epochs the transformer LoRA layers are trained. By default, `--train_transformer_frac==1`, to trigger a textual inversion run set `--train_transformer_frac==0`. Values between 0 and 1 are supported as well, and we welcome the community to experiment w/ different settings and share the results!
* **token initializer** - similar to the original textual inversion work, you can specify a concept of your choosing as the starting point for training. By default, when enabling `--train_text_encoder_ti`, the new inserted tokens are initialized randomly. You can specify a token in `--initializer_concept` such that the starting point for the trained embeddings will be the embeddings associated with your chosen `--initializer_concept`.

## Training examples

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

### Example #1: Pivotal tuning
**Now, we can launch training:**

```bash
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export DATASET_NAME="./3d_icon"
export OUTPUT_DIR="3d-icon-Flux-LoRA"

accelerate launch train_dreambooth_lora_flux_advanced.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --instance_prompt="3d icon in the style of TOK" \
  --output_dir=$OUTPUT_DIR \
  --caption_column="prompt" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
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
  --max_train_steps=700 \
  --checkpointing_steps=2000 \
  --seed="0" \
  --push_to_hub
```

To better track our training experiments, we're using the following flags in the command above:

* `report_to="wandb` will ensure the training runs are tracked on Weights and Biases. To use it, be sure to install `wandb` with `pip install wandb`.
* `validation_prompt` and `validation_epochs` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.

Our experiments were conducted on a single 40GB A100 GPU.

### Example #2: Pivotal tuning with T5
Now let's try that with T5 as well, so instead of only optimizing the CLIP embeddings associated with newly inserted tokens, we'll optimize
the T5 embeddings as well. We can do this by simply adding `--enable_t5_ti` to the previous configuration:
```bash
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export DATASET_NAME="./3d_icon"
export OUTPUT_DIR="3d-icon-Flux-LoRA"

accelerate launch train_dreambooth_lora_flux_advanced.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --instance_prompt="3d icon in the style of TOK" \
  --output_dir=$OUTPUT_DIR \
  --caption_column="prompt" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --repeats=1 \
  --report_to="wandb"\
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1.0 \
  --text_encoder_lr=1.0 \
  --optimizer="prodigy"\
  --train_text_encoder_ti\
  --enable_t5_ti\
  --train_text_encoder_ti_frac=0.5\
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank=8 \
  --max_train_steps=700 \
  --checkpointing_steps=2000 \
  --seed="0" \
  --push_to_hub
```

### Example #3: Textual Inversion
To explore a pure textual inversion - i.e. only optimizing the text embeddings w/o training transformer LoRA layers, we 
can set the value for `--train_transformer_frac` - which is responsible for the percent of epochs in which the transformer is 
trained. By setting `--train_transformer_frac == 0` and enabling `--train_text_encoder_ti` we trigger a textual inversion train 
run.
```bash
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export DATASET_NAME="./3d_icon"
export OUTPUT_DIR="3d-icon-Flux-LoRA"

accelerate launch train_dreambooth_lora_flux_advanced.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --instance_prompt="3d icon in the style of TOK" \
  --output_dir=$OUTPUT_DIR \
  --caption_column="prompt" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --repeats=1 \
  --report_to="wandb"\
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1.0 \
  --text_encoder_lr=1.0 \
  --optimizer="prodigy"\
  --train_text_encoder_ti\
  --enable_t5_ti\
  --train_text_encoder_ti_frac=0.5\
  --train_transformer_frac=0\
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank=8 \
  --max_train_steps=700 \
  --checkpointing_steps=2000 \
  --seed="0" \
  --push_to_hub
```
### Inference - pivotal tuning

Once training is done, we can perform inference like so:
1. starting with loading the transformer lora weights
```python
import torch
from huggingface_hub import hf_hub_download, upload_file
from diffusers import AutoPipelineForText2Image
from safetensors.torch import load_file

username = "linoyts"
repo_id = f"{username}/3d-icon-Flux-LoRA"

pipe = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')


pipe.load_lora_weights(repo_id, weight_name="pytorch_lora_weights.safetensors")
```
2. now we load the pivotal tuning embeddings 
> [!NOTE] #1 if `--enable_t5_ti` wasn't passed, we only load the embeddings to the CLIP encoder.

> [!NOTE] #2 the number of tokens (i.e. <s0>,...,<si>) is either determined by `--num_new_tokens_per_abstraction` or by `--initializer_concept`. Make sure to update inference code accordingly :)
```python
text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

embedding_path = hf_hub_download(repo_id=repo_id, filename="3d-icon-Flux-LoRA_emb.safetensors", repo_type="model")

state_dict = load_file(embedding_path)
# load embeddings of text_encoder 1 (CLIP ViT-L/14)
pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
# load embeddings of text_encoder 2 (T5 XXL) - ignore this line if you didn't enable `--enable_t5_ti`
pipe.load_textual_inversion(state_dict["t5"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
```

3. let's generate images

```python
instance_token = "<s0><s1>"
prompt = f"a {instance_token} icon of an orange llama eating ramen, in the style of {instance_token}"

image = pipe(prompt=prompt, num_inference_steps=25, cross_attention_kwargs={"scale": 1.0}).images[0]
image.save("llama.png")
```

### Inference - pure textual inversion
In this case, we don't load transformer layers as before, since we only optimize the text embeddings. The output of a textual inversion train run is a
`.safetensors` file containing the trained embeddings for the new tokens either for the CLIP encoder, or for both encoders (CLIP and T5) 

1. starting with loading the embeddings.
💡note that here too, if you didn't enable `--enable_t5_ti`, you only load the embeddings to the CLIP encoder

```python
import torch
from huggingface_hub import hf_hub_download, upload_file
from diffusers import AutoPipelineForText2Image
from safetensors.torch import load_file

username = "linoyts"
repo_id = f"{username}/3d-icon-Flux-LoRA"

pipe = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')

text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

embedding_path = hf_hub_download(repo_id=repo_id, filename="3d-icon-Flux-LoRA_emb.safetensors", repo_type="model")

state_dict = load_file(embedding_path)
# load embeddings of text_encoder 1 (CLIP ViT-L/14)
pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
# load embeddings of text_encoder 2 (T5 XXL) - ignore this line if you didn't enable `--enable_t5_ti`
pipe.load_textual_inversion(state_dict["t5"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
```
2. let's generate images

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
