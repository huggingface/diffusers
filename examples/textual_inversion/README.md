## Textual Inversion fine-tuning example

[Textual inversion](https://arxiv.org/abs/2208.01618) is a method to personalize text2image models like stable diffusion on your own images using just 3-5 examples.
The `textual_inversion.py` script shows how to implement the training procedure and adapt it for stable diffusion.

## Running on Colab 

Colab for training 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb)

Colab for inference
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_conceptualizer_inference.ipynb)

## Running locally with PyTorch
### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

```bash
pip install diffusers"[training]" accelerate "transformers>=4.21.0"
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```


### Cat toy example

You need to accept the model license before downloading or using the weights. In this example we'll use model version `v1-5`, so you'll need to visit [its card](https://huggingface.co/runwayml/stable-diffusion-v1-5), read the license and tick the checkbox if you agree. 

You have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work. For more information on access tokens, please refer to [this section of the documentation](https://huggingface.co/docs/hub/security-tokens).

Run the following command to authenticate your token

```bash
huggingface-cli login
```

If you have already cloned the repo, then you won't need to go through these steps. 

<br>

Now let's get our dataset.Download 3-4 images from [here](https://drive.google.com/drive/folders/1fmJMs25nxS_rSNqS5hTcRdLem_YQXbq5) and save them in a directory. This will be our training data.

And launch the training using

**___Note: Change the `resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.___**

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="path-to-dir-containing-images"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
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

A full training run takes ~1 hour on one V100 GPU.

### Inference

Once you have trained a model using above command, the inference can be done simply using the `StableDiffusionPipeline`. Make sure to include the `placeholder_token` in your prompt.

```python
from diffusers import StableDiffusionPipeline

model_id = "path-to-your-trained-model"
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")

prompt = "A <cat-toy> backpack"

image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("cat-backpack.png")
```


## Training with Flax/JAX

For faster training on TPUs and GPUs you can leverage the flax training example. Follow the instructions above to get the model and dataset before running the script.

Before running the scripts, make sure to install the library's training dependencies:

```bash
pip install -U -r requirements_flax.txt
```

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export DATA_DIR="path-to-dir-containing-images"

python textual_inversion_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --output_dir="textual_inversion_cat"
```
It should be at least 70% faster than the PyTorch script with the same configuration.

## Training with Intel Extension for PyTorch

Intel Extension for PyTorch provides the optimizations for faster training and inference on CPUs. You can leverage the training example "textual_inversion_ipex.py". Follow the instructions above to get the model and dataset before running the script.

The example supports both single node and multi-node distributed training:

### Single node training

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="path-to-dir-containing-dicoo-images"

python textual_inversion_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<dicoo>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --output_dir="textual_inversion_dicoo"
```

You also enable "--use_bf16" to accelerate the training on Intel Xeon Scalable Processors with BF16 support.

### Multi-node distributed training

Before running the scripts, make sure to install the library's training dependencies:

```bash
python -m pip install oneccl_bind_pt==1.13 -f https://developer.intel.com/ipex-whl-stable-cpu
```

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="path-to-dir-containing-dicoo-images"

oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh

python -m intel_extension_for_pytorch.cpu.launch --distributed \
  --hostfile hostfile --nnodes 2 --nproc_per_node 2 \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<dicoo>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --output_dir="textual_inversion_dicoo"
```
Here is a simple distributed training usage on 2 nodes and 2 processes per each node. Add the right hostname or ip address in the "hostfile" and make sure 2 nodes are reachable. For more details, please refer to the [user guide](https://github.com/intel/torch-ccl).


### Reference

We publish a [Medium blog](https://medium.com/intel-analytics-software/personalized-stable-diffusion-with-few-shot-fine-tuning-on-a-single-cpu-f01a3316b13) on how to create your own Stable Diffusion model on CPUs. Try it out now, if you have interests.
