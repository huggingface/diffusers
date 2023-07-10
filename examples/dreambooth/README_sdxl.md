# DreamBooth training example for Stable Diffusion XL (SDXL)

[DreamBooth](https://arxiv.org/abs/2208.12242) is a method to personalize text2image models like stable diffusion given just a few(3~5) images of a subject.

The `train_dreambooth_lora_sdxl.py` script shows how to implement the training procedure and adapt it for [Stable Diffusion XL](https://github.com/Stability-AI/generative-models/blob/main/assets/sdxl_report.pdf).

> 💡 **Note**: For now, we only allow DreamBooth fine-tuning of the SDXL UNet via LoRA. LoRA is a parameter-efficient fine-tuning technique introduced in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*. 

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

Since SDXL 0.9 weights are gated, we need to be authenticated to be able to use them. So, let's run:

```bash
huggingface-cli login
```

This will also allow us to push the trained LoRA parameters to the Hugging Face Hub platform. 

Now, we can launch training using:

```bash
export MODEL_NAME="diffusers/stable-diffusion-xl-base-0.9"
export INSTANCE_DIR="dog"
export CLASS_DIR="dog-class"
export OUTPUT_DIR="lora-trained-xl"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=50 \
  --seed="0" \
  --push_to_hub
```

To better track our training experiments, we're using the following flags in the command above:

* `report_to="wandb` will ensure the training runs are tracked on Weights and Biases. To use it, be sure to install `wandb` with `pip install wandb`.
* `validation_prompt` and `validation_epochs` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected. 

Our experiments were conducted on a single 40GB A100 GPU.

## Notes

In our experiments, we have found out that SDXL, with the default settings of the script, already yields better initial results. This is why, we didn't explore further hyperparameter tuning experiments. However, we encourage the community to explore this avenue further and share their results with us 🤗

## Results

You can explore the results from a couple of our internal experiments by checking out this link: [https://wandb.ai/sayakpaul/dreambooth-lora-sd-xl](https://wandb.ai/sayakpaul/dreambooth-lora-sd-xl). Specifically, we used the same script with the exact the same hyperparameters on the following datasets:

* [Dogs](https://huggingface.co/datasets/diffusers/dog-example)
* [Starbucks logo](https://huggingface.co/datasets/diffusers/starbucks-example)
* [Mr. Potato Head](https://huggingface.co/datasets/diffusers/potato-head-example)
* [Keramer face](https://huggingface.co/datasets/diffusers/keramer-face-example)