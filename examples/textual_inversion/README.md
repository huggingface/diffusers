## Textual Inversion fine-tuning example

[Textual inversion](https://arxiv.org/abs/2208.01618) is a method to personalize text2image models like stable diffusion on your own images using just 3-5 examples.
The `textual_inversion.py` script shows how to implement the training procedure and adapt it for stable diffusion.


## Modifications
This repository is made so that training can be done in 6gb. To accomplish this, everything except the unet has been moved to cpu while unet remains in GPU.
The training time is roughly 21 hours while on colab it's roughly 13 hours.

Additional changes:
- Added wandb and made it so that model runs inference every log_frequency number of steps and saves every save_frequency number of steps
- Added support for having multiple tokens represent the concept. This works as so: given a placeholder token frida, in place of where frida was it'll place frida_1 frida_2 ... frida_n where n is the number of vectors we want to use to represent the concept(num_vec_per_token). For example, "A picture of frida" becomes "A picture of frida_1 frida_2 ... frida_n"
This idea was taken from the original textual inversion repository (here)[https://github.com/rinongal/textual_inversion]. Will detail how this works with initializing words later.
- Added support to guess the words that represent the image. This was an idea I got from the clip interrogator (here)[https://github.com/AUTOMATIC1111/stable-diffusion-webui]. The main idea is that we are not sure that the initial word we provide is the closest token to the given concept. So, if we can have a model to generate the prompt given the image, we can guess what the right prompt is. I used the blip model for this and the result for my dog example is "white dog". Definintely needs improvement but it's pretty nice.


Added support for multiple tokens 
For 6gb gpu memory, run the below command
```
accelerate launch textual_inversion.py --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" --use_auth_token --train_data_dir="frida" --learnable_property="object" --placeholder_token="<frida>" --initializer_token="dog" --resolution=256 --train_batch_size=1  --gradient_accumulation_steps=4 --max_train_steps=3000 --learning_rate=5.0e-04 --scale_lr --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="textual_inversion_frida" --slice_div=1 --mixed_precision="no"
```
## Running on Colab 

Colab for training 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb)

Colab for inference
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_conceptualizer_inference.ipynb)

## Running locally 
### Installing the dependencies

Before running the scipts, make sure to install the library's training dependencies:
Go to the base repo and do
```bash
pip install -e .
```

```bash
pip install accelerate transformers timm fairscale
git clone https://github.com/salesforce/BLIP.git
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```
Modification: Set to cpu only for it to fit in 6gb.

### Cat toy example

You need to accept the model license before downloading or using the weights. In this example we'll use model version `v1-4`, so you'll need to visit [its card](https://huggingface.co/CompVis/stable-diffusion-v1-4), read the license and tick the checkbox if you agree. 

You have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work. For more information on access tokens, please refer to [this section of the documentation](https://huggingface.co/docs/hub/security-tokens).

Run the following command to autheticate your token

```bash
huggingface-cli login
```

If you have already cloned the repo, then you won't need to go through these steps. You can simple remove the `--use_auth_token` arg from the following command.

<br>

Now let's get our dataset.Download 3-4 images from [here](https://drive.google.com/drive/folders/1fmJMs25nxS_rSNqS5hTcRdLem_YQXbq5) and save them in a directory. This will be our training data.

And launch the training using


```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="path-to-dir-containing-images"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME --use_auth_token \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="textual_inversion_cat"
```

```
accelerate launch textual_inversion.py --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" --use_auth_token --train_data_dir="frida" --learnable_property="object" --placeholder_token="<frida>" --initializer_token="dog" --resolution=512 --train_batch_size=1  --gradient_accumulation_steps=4 --max_train_steps=3000 --learning_rate=5.0e-04 --scale_lr --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="textual_inversion_frida"
```

A full training run takes ~1 hour on one V100 GPU.


### Inference

Once you have trained a model using above command, the inference can be done simply using the `StableDiffusionPipeline`. Make sure to include the `placeholder_token` in your prompt.

```python

from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "path-to-your-trained-model"
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")

prompt = "A <cat-toy> backpack"

with autocast("cuda"):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("cat-backpack.png")
```
