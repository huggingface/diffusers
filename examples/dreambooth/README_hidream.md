# DreamBooth training example for HiDream Image

[DreamBooth](https://huggingface.co/papers/2208.12242) is a method to personalize text2image models like stable diffusion given just a few (3~5) images of a subject.

The `train_dreambooth_lora_hidream.py` script shows how to implement the training procedure with [LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) and adapt it for [HiDream Image](https://huggingface.co/docs/diffusers/main/en/api/pipelines/). 


This will also allow us to push the trained model parameters to the Hugging Face Hub platform.

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

Then cd in the `examples/dreambooth` folder and run
```bash
pip install -r requirements_hidream.txt
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
Note also that we use PEFT library as backend for LoRA training, make sure to have `peft>=0.14.0` installed in your environment.


### 3d icon example

For this example we will use some 3d icon images: https://huggingface.co/datasets/linoyts/3d_icon.

This will also allow us to push the trained LoRA parameters to the Hugging Face Hub platform.

Now, we can launch training using:
> [!NOTE]
> The following training configuration prioritizes lower memory consumption by using gradient checkpointing, 
> 8-bit Adam optimizer, latent caching, offloading, no validation.
> all text embeddings are pre-computed to save memory.
```bash
export MODEL_NAME="HiDream-ai/HiDream-I1-Dev"
export INSTANCE_DIR="linoyts/3d_icon"
export OUTPUT_DIR="trained-hidream-lora"

accelerate launch train_dreambooth_lora_hidream.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --dataset_name=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="3d icon" \
  --caption_column="prompt"\
  --validation_prompt="a 3dicon, a llama eating ramen" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --use_8bit_adam \
  --rank=8 \
  --learning_rate=2e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --max_train_steps=1000 \
  --cache_latents\
  --gradient_checkpointing \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub
```

For using `push_to_hub`, make you're logged into your Hugging Face account:

```bash
hf auth login
```

To better track our training experiments, we're using the following flags in the command above:

* `report_to="wandb` will ensure the training runs are tracked on [Weights and Biases](https://wandb.ai/site). To use it, be sure to install `wandb` with `pip install wandb`. Don't forget to call `wandb login <your_api_key>` before training if you haven't done it before.
* `validation_prompt` and `validation_epochs` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.

## Notes

Additionally, we welcome you to explore the following CLI arguments:

* `--lora_layers`: The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v" will result in lora training of attention layers only.
* `--rank`: The rank of the LoRA layers. The higher the rank, the more parameters are trained. The default is 16.

We provide several options for optimizing memory optimization:

* `--offload`: When enabled, we will offload the text encoder and VAE to CPU, when they are not used.
* `cache_latents`: When enabled, we will pre-compute the latents from the input images with the VAE and remove the VAE from memory once done.
* `--use_8bit_adam`: When enabled, we will use the 8bit version of AdamW provided by the `bitsandbytes` library.

Refer to the [official documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/) of the `HiDreamImagePipeline` to know more about the model.

## Using quantization

You can quantize the base model with [`bitsandbytes`](https://huggingface.co/docs/bitsandbytes/index) to reduce memory usage. To do so, pass a JSON file path to `--bnb_quantization_config_path`. This file should hold the configuration to initialize `BitsAndBytesConfig`. Below is an example JSON file:

```json
{
    "load_in_4bit": true,
    "bnb_4bit_quant_type": "nf4"
}
```

Below, we provide some numbers with and without the use of NF4 quantization when training:

```
(with quantization)
Memory (before device placement): 9.085089683532715 GB.
Memory (after device placement): 34.59585428237915 GB.
Memory (after backward): 36.90267467498779 GB.

(without quantization)
Memory (before device placement): 0.0 GB.
Memory (after device placement): 57.6400408744812 GB.
Memory (after backward): 59.932212829589844 GB.
```

The reason why we see some memory before device placement in the case of quantization is because, by default bnb quantized models are placed on the GPU first.