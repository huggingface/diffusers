# Würstchen Dreambooth fine-tuning

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
cd examples/wuerstchen/dreambooth
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

## Dreambooth

[DreamBooth](https://arxiv.org/abs/2208.12242) is a method to personalize text2image models like stable diffusion given just a few(3~5) images of a subject.
The `train_dreambooth_lora.py` script shows how to implement the training procedure and adapt it for the Würstchen model.


For this example we will use some dog images: https://huggingface.co/datasets/diffusers/dog-example together with LoRA. In a nutshell, LoRA allows to adapt pretrained models by adding pairs of rank-decomposition matrices to existing weights and **only** training those newly added weights.

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

```bash
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth-model"

accelerate launch train_dreambooth_lora.py \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --push_to_hub
```

**___Note: When using LoRA we can use a much higher learning rate compared to vanilla dreambooth. Here we use *1e-4*.

### Training with prior-preservation loss

Prior-preservation is used to avoid overfitting and language-drift. Refer to the paper to learn more about it. For prior-preservation we first generate images using the model with a class prompt and then use those during training along with our data.
According to the paper, it's recommended to generate `num_epochs * num_samples` images for prior-preservation. 200-300 works well for most cases. The `num_class_images` flag sets the number of images to generate with the class prompt. You can place existing images in `class_data_dir`, and the training script will generate any additional images so that `num_class_images` are present in `class_data_dir` during training time.

```bash
export INSTANCE_DIR="dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth_lora.py \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --push_to_hub
```

### Fine-tune text encoder with the UNet.

The script also allows to fine-tune the `text_encoder` along with the `prior`. It's been observed experimentally that fine-tuning `text_encoder` gives much better results especially on faces.  Pass the `--train_text_encoder` argument to the script to enable training `text_encoder`.

```bash
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth-model"

accelerate launch train_dreambooth_lora.py \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=4 \
  --use_8bit_adam \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --push_to_hub
```
