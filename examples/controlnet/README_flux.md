# ControlNet training example for FLUX

The `train_controlnet_flux.py` script shows how to implement the ControlNet training procedure and adapt it for [FLUX](https://github.com/black-forest-labs/flux).

Training script provided by LibAI, which is an institution dedicated to the progress and achievement of artificial general intelligence. LibAI is the developer of [cutout.pro](https://www.cutout.pro/) and [promeai.pro](https://www.promeai.pro/).

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

Then cd in the `examples/controlnet` folder and run
```bash
pip install -r requirements_flux.txt
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

## Custom Datasets

We support importing data from jsonl(xxx.jsonl),here is a brief example:
```sh
{"image_path": "xxx", "caption": "xxx", "control_path": "xxx"}
{"image_path": "xxx", "caption": "xxx", "control_path": "xxx"}
```


## Training

Our training examples use two test conditioning images. They can be downloaded by running

```sh
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

Then run `huggingface-cli login` to log into your Hugging Face account. This is needed to be able to push the trained ControlNet parameters to Hugging Face Hub.

we can define the num_layers, num_single_layers, which determines the size of the control(default values are num_layers=4, num_single_layers=10)


```bash
export MODEL_DIR="black-forest-labs/FLUX.1-dev"
export OUTPUT_DIR="path to save model"
export TRAIN_JSON_FILE="path to your jsonl file"


accelerate launch train_controlnet_flux.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --conditioning_image_column=control_path \
    --image_column=image_path \
    --caption_column=caption \
    --output_dir=$OUTPUT_DIR \
    --jsonl_for_train=$TRAIN_JSON_FILE \
    --mixed_precision="bf16" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --max_train_steps=15000 \
    --validation_steps=100 \
    --checkpointing_steps=200 \
    --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
    --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --report_to="tensorboard" \
    --num_double_layers=4 \
    --num_single_layers=0 \
    --seed=42 \
```

To better track our training experiments, we're using the following flags in the command above:

* `report_to="tensorboard` will ensure the training runs are tracked on Weights and Biases.
* `validation_image`, `validation_prompt`, and `validation_steps` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.

Our experiments were conducted on a single 40GB A100 GPU.

### Inference

Once training is done, we can perform inference like so:

```python
import torch
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model = 'path to controlnet'
controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(
    base_model, 
    controlnet=controlnet, 
    torch_dtype=torch.bfloat16
)
# enable memory optimizations   
pipe.enable_model_cpu_offload()

control_image = load_image("./conditioning_image_1.png").resize((1024, 1024))
prompt = "pale golden rod circle with old lace background"

image = pipe(
    prompt, 
    control_image=control_image,
    controlnet_conditioning_scale=0.6,
    num_inference_steps=28, 
    guidance_scale=3.5,
).images[0]
image.save("./output.png")
```

## Notes

### T5 dont support bf16 autocast and i dont know why, will cause black image.

```diff
if is_final_validation or torch.backends.mps.is_available():
    autocast_ctx = nullcontext()
else:
    # t5 seems not support autocast and i don't know why
+   autocast_ctx = nullcontext()
-   autocast_ctx = torch.autocast(accelerator.device.type)
```

### TO Fix Error

```bash
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

#### we need to change some code in `diffusers/src/diffusers/pipelines/flux/pipeline_flux_controlnet.py` to ensure the dtype

```diff
noise_pred = self.transformer(
    hidden_states=latents,
    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
    timestep=timestep / 1000,
    guidance=guidance,
    pooled_projections=pooled_prompt_embeds,
    encoder_hidden_states=prompt_embeds,
-   controlnet_block_samples=controlnet_block_samples,
-   controlnet_single_block_samples=controlnet_single_block_samples,
+   controlnet_block_samples=[sample.to(dtype=latents.dtype) for sample in controlnet_block_samples]if controlnet_block_samples is not None else None,
+   controlnet_single_block_samples=[sample.to(dtype=latents.dtype) for sample in controlnet_single_block_samples] if controlnet_single_block_samples is not None else None,
    txt_ids=text_ids,
    img_ids=latent_image_ids,
    joint_attention_kwargs=self.joint_attention_kwargs,
    return_dict=False,
)[0]
```