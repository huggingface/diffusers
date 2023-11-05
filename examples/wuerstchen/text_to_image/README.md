# Würstchen text-to-image fine-tuning

## Running locally with PyTorch

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date. To do this, execute the following steps in a new virtual environment:
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
For this example we want to directly store the trained LoRA embeddings on the Hub, so we need to be logged in and add the `--push_to_hub` flag to the training script. To log in, run:
```bash
huggingface-cli login
```

## Prior training

You can fine-tune the Würstchen prior model with the `train_text_to_image_prior.py` script. Note that we currently support `--gradient_checkpointing` for prior model fine-tuning so you can use it for more GPU memory constrained setups.

<br>

<!-- accelerate_snippet_start -->
```bash
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch  train_text_to_image_prior.py \
  --mixed_precision="fp16" \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --dataloader_num_workers=4 \
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

## Training with LoRA

Low-Rank Adaption of Large Language Models (or LoRA) was first introduced by Microsoft in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*.

In a nutshell, LoRA allows adapting pretrained models by adding pairs of rank-decomposition matrices to existing weights and **only** training those newly added weights. This has a couple of advantages:

- Previous pretrained weights are kept frozen so that the model is not prone to [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114).
- Rank-decomposition matrices have significantly fewer parameters than original model, which means that trained LoRA weights are easily portable.
- LoRA attention layers allow to control to which extent the model is adapted toward new training images via a `scale` parameter.


### Prior Training

First, you need to set up your development environment as explained in the [installation](#Running-locally-with-PyTorch) section. Make sure to set the `DATASET_NAME` environment variable. Here, we will use the [Pokemon captions dataset](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions).  

```bash
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch train_text_to_image_prior_lora.py \
  --mixed_precision="fp16" \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=768 \
  --train_batch_size=8 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --rank=4 \
  --validation_prompt="cute dragon creature" \
  --report_to="wandb" \
  --push_to_hub \
  --output_dir="wuerstchen-prior-pokemon-lora"
```
