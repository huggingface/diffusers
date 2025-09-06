# Stable Diffusion Inpainting fine-tuning

The `train_inpainting.py` script shows how to train/fine-tune stable diffusion model for inpainting on your own dataset.

___Note___:

___This script is experimental. The script fine-tunes the whole model and often times the model overfits and runs into issues like catastrophic forgetting. It's recommended to try different hyperparamters to get the best result on your dataset.___


## Running locally with PyTorch
### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then cd in the example folder  and run
```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Note also that we use PEFT library as backend for LoRA training, make sure to have `peft>=0.6.0` installed in your environment.

### Pokemon example

You need to accept the model license before downloading or using the weights. In this example we'll use model version `sd-v1-5-inpainting` or `v1-5` from runwayml, so you'll need to visit [inpainting card](https://huggingface.co/runwayml/stable-diffusion-inpainting) or [v1-5 card](https://huggingface.co/runwayml/stable-diffusion-v1-5), read the license and tick the checkbox if you agree.

You have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work. For more information on access tokens, please refer to [this section of the documentation](https://huggingface.co/docs/hub/security-tokens).

Run the following command to authenticate your token

```bash
huggingface-cli login
```

If you have already cloned the repo, then you won't need to go through these steps.

<br>

#### Hardware
With `gradient_checkpointing` and `mixed_precision` it should be possible to fine tune the model on a single 24GB GPU. For higher `batch_size` and faster training it's better to use GPUs with >30GB memory.

**___Note: Change the `resolution` to 768 if you are using the [stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) or [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.___**
<!-- accelerate_snippet_start -->
```bash
export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"


accelerate launch --mixed_precision="fp16" train_inpainting.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 --seed=42 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model-inpaint" \
  --validation_size=3
```
<!-- accelerate_snippet_end -->


To run on your own training files prepare the dataset according to the format required by `datasets`, you can find the instructions for how to do that in this [document](https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder-with-metadata).
If you wish to use custom loading logic, you should modify the script, we have left pointers for that in the training script.

```bash
export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export DATASET_NAME="path_to_your_dataset" (NOT IMPLEMENTED)


accelerate launch --mixed_precision="fp16" train_inpainting.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 --seed=42 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model-inpaint" \
  --validation_size=3
```


Once the training is finished the model will be saved in the `output_dir` specified in the command. In this example it's `sd-pokemon-model-inpaint`. To load the fine-tuned model for inference just pass that path to `StableDiffusionInpaintPipeline`

```python
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

init_image = Image.open("path_to_image").resize((512, 512))
mask_image = Image.open("path_to_mask").resize((512, 512))

model_path = "path_to_saved_model"
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

inpainted_image = pipe(
    prompt = "yoda", 
    image = init_image, 
    mask_image = mask_image,
).images[0]

inpainted_image.save("inpainted-yoda-pokemon.png")
```

Checkpoints only save the unet, so to run inference from a checkpoint, just load the unet

```python
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel

init_image = Image.open("path_to_image").resize((512, 512))
mask_image = Image.open("path_to_mask").resize((512, 512))

model_path = "path_to_saved_model"
unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-<N>/unet", torch_dtype=torch.float16)

pipe = StableDiffusionInpaintPipeline.from_pretrained("<initial model>", unet=unet, torch_dtype=torch.float16)
pipe.to("cuda")

inpainted_image = pipe(
    prompt = "yoda", 
    image = init_image, 
    mask_image = mask_image,
).images[0]

inpainted_image.save("inpainted-yoda-pokemon.png")
```

#### Training with multiple GPUs

`accelerate` allows for seamless multi-GPU training. Follow the instructions [here](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
for running distributed training with `accelerate`. Here is an example command:

```bash
export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"


accelerate launch --mixed_precision="fp16" --multi_gpu train_inpainting.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 --seed=42 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model-inpaint" \
  --validation_size=3  
```


#### Training with Min-SNR weighting

We support training with the Min-SNR weighting strategy proposed in [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556) which helps to achieve faster convergence
by rebalancing the loss. In order to use it, one needs to set the `--snr_gamma` argument. The recommended
value when using it is 5.0.

You can find [this project on Weights and Biases](https://wandb.ai/sayakpaul/text2image-finetune-minsnr) that compares the loss surfaces of the following setups:

* Training without the Min-SNR weighting strategy
* Training with the Min-SNR weighting strategy (`snr_gamma` set to 5.0)
* Training with the Min-SNR weighting strategy (`snr_gamma` set to 1.0)

For our small Pokemons dataset, the effects of Min-SNR weighting strategy might not appear to be pronounced, but for larger datasets, we believe the effects will be more pronounced.

Also, note that in this example, we either predict `epsilon` (i.e., the noise) or the `v_prediction`. For both of these cases, the formulation of the Min-SNR weighting strategy that we have used holds.