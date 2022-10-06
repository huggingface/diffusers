# Stable Diffusion text-to-image fine-tuning

The `train_text_to_image.py` script shows how to fine-tune stable diffusion model on your own dataset.


## Running locally 
### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

```bash
pip install git+https://github.com/huggingface/diffusers.git
pip install -U -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

### Pokemon example

You need to accept the model license before downloading or using the weights. In this example we'll use model version `v1-4`, so you'll need to visit [its card](https://huggingface.co/CompVis/stable-diffusion-v1-4), read the license and tick the checkbox if you agree. 

You have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work. For more information on access tokens, please refer to [this section of the documentation](https://huggingface.co/docs/hub/security-tokens).

Run the following command to authenticate your token

```bash
huggingface-cli login
```

If you have already cloned the repo, then you won't need to go through these steps.

<br>

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export dataset_name="lambdalabs/pokemon-blip-captions"

accelerate launch ../diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME --use_auth_token \
  --dataset_name=$dataset_name \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 --gradient_checkpointing \
  --gradient_accumulation_steps=4 --max_grad_norm=1 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 --use_ema \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model" \
```