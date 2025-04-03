## LoRA fine-tuning Flux.1 Dev with quantization

> [!NOTE]  
> This example is educational in nature and fixes some arguments to keep things simple. It should act as a reference to build things further.

This example shows how to fine-tune [Flux.1 Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) with LoRA and quantization. We show this by using the [`Norod78/Yarn-art-style`](https://huggingface.co/datasets/Norod78/Yarn-art-style) dataset. Steps below summarize the workflow:

* We precompute the text embeddings in `compute_embeddings.py` and serialize them into a parquet file.
  * Even though optional, we load the T5-xxl in NF4 to further reduce the memory foot-print. 
* `train_dreambooth_lora_flux_miniature.py` takes care of training:
  * Since we already precomputed the text embeddings, we don't load the text encoders.
  * We load the VAE and use it to precompute the image latents and we then delete it. 
  * Load the Flux transformer, quantize it with the [NF4 datatype](https://arxiv.org/abs/2305.14314) through `bitsandbytes`, prepare it for 4bit training. 
  * Add LoRA adapter layers to it and then ensure they are kept in FP32 precision.
  * Train!

To run training in a memory-optimized manner, we additionally use:

* 8Bit Adam
* Gradient checkpointing 

We have tested the scripts on a 24GB 4090. It works on a free-tier Colab Notebook, too, but it's extremely slow. 

## Training

Ensure you have installed the required libraries:

```bash
pip install -U transformers accelerate bitsandbytes peft datasets 
pip install git+https://github.com/huggingface/diffusers -U
```

Now, compute the text embeddings:

```bash
python compute_embeddings.py
```

It should create a file named `embeddings.parquet`. We're then ready to launch training. First, authenticate so that you can access the Flux.1 Dev model: 

```bash
huggingface-cli
```

Then launch:

```bash
accelerate launch --config_file=accelerate.yaml \
  train_dreambooth_lora_flux_miniature.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --data_df_path="embeddings.parquet" \
  --output_dir="yarn_art_lora_flux_nf4" \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --weighting_scheme="none" \
  --resolution=1024 \
  --train_batch_size=1 \
  --repeats=1 \
  --learning_rate=1e-4 \
  --guidance_scale=1 \
  --report_to="wandb" \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --cache_latents \
  --rank=4 \
  --max_train_steps=700 \
  --seed="0"
```

We can direcly pass a quantized checkpoint path, too:

```diff
+ --quantized_model_path="hf-internal-testing/flux.1-dev-nf4-pkg"
```

Depending on the machine, training time will vary but for our case, it was 1.5 hours. It maybe possible to speed this up by using `torch.bfloat16`. 

We support training with the DeepSpeed Zero2 optimizer, too. To use it, first install DeepSpeed:

```bash
pip install -Uq deepspeed
```

And then launch:

```bash
accelerate launch --config_file=ds2.yaml \
  train_dreambooth_lora_flux_miniature.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --data_df_path="embeddings.parquet" \
  --output_dir="yarn_art_lora_flux_nf4" \
  --mixed_precision="no" \
  --use_8bit_adam \
  --weighting_scheme="none" \
  --resolution=1024 \
  --train_batch_size=1 \
  --repeats=1 \
  --learning_rate=1e-4 \
  --guidance_scale=1 \
  --report_to="wandb" \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --cache_latents \
  --rank=4 \
  --max_train_steps=700 \
  --seed="0"
```

## Inference

When loading the LoRA params (that were obtained on a quantized base model) and merging them into the base model, it is recommended to first dequantize the base model, merge the LoRA params into it, and then quantize the model again. This is because merging into 4bit quantized models can lead to some rounding errors. Below, we provide an end-to-end example:

1. First, load the original model and merge the LoRA params into it:

```py
from diffusers import FluxPipeline 
import torch 

ckpt_id = "black-forest-labs/FLUX.1-dev"
pipeline = FluxPipeline.from_pretrained(
    ckpt_id, text_encoder=None, text_encoder_2=None, torch_dtype=torch.float16
)
pipeline.load_lora_weights("yarn_art_lora_flux_nf4", weight_name="pytorch_lora_weights.safetensors")
pipeline.fuse_lora()
pipeline.unload_lora_weights()

pipeline.transformer.save_pretrained("fused_transformer")
```

2. Quantize the model and run inference

```py
from diffusers import AutoPipelineForText2Image, FluxTransformer2DModel, BitsAndBytesConfig
import torch

ckpt_id = "black-forest-labs/FLUX.1-dev"
bnb_4bit_compute_dtype = torch.float16
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
)
transformer = FluxTransformer2DModel.from_pretrained(
    "fused_transformer",
    quantization_config=nf4_config,
    torch_dtype=bnb_4bit_compute_dtype,
)
pipeline = AutoPipelineForText2Image.from_pretrained(
    ckpt_id, transformer=transformer, torch_dtype=bnb_4bit_compute_dtype
)
pipeline.enable_model_cpu_offload()

image = pipeline(
    "a puppy in a pond, yarn art style", num_inference_steps=28, guidance_scale=3.5, height=768
).images[0]
image.save("yarn_merged.png")
```

|   Dequantize, merge, quantize   |   Merging directly into quantized model   |
|-------|-------|
| ![Image A](https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/quantized_flux_training/merged.png) | ![Image B](https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/quantized_flux_training/unmerged.png) |

As we can notice the first column result follows the style more closely.
