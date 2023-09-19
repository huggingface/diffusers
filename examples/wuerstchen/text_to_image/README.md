# Würstchen text-to-image fine-tuning

## Running locally with PyTorch

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then cd into the example folder and run
```bash
cd examples/wuerstchen/text_to_image
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```
For this example we want to directly store the trained LoRA embeddings on the Hub, so we need to be logged in and add the `--push_to_hub` flag.

## Prior training

You can fine-tune the Würstchen prior model with `train_text_to_image_prior.py` script. Note that we currently do not support `--gradient_checkpointing` for prior model fine-tuning.

<br>

<!-- accelerate_snippet_start -->
```bash
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

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
  --validation_prompts="A robot pokemon, 4k photo" \
  --report_to="wandb" \
  --push_to_hub \
  --output_dir="wuerstchen-prior-pokemon-model" 
```
<!-- accelerate_snippet_end -->
