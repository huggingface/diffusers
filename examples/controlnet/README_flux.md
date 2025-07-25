# ControlNet training example for FLUX

The `train_controlnet_flux.py` script shows how to implement the ControlNet training procedure and adapt it for [FLUX](https://github.com/black-forest-labs/flux).

Training script provided by LibAI, which is an institution dedicated to the progress and achievement of artificial general intelligence. LibAI is the developer of [cutout.pro](https://www.cutout.pro/) and [promeai.pro](https://www.promeai.pro/).
> [!NOTE]
> **Memory consumption**
>
> Flux can be quite expensive to run on consumer hardware devices and as a result, ControlNet training of it comes with higher memory requirements than usual.

Here is a gpu memory consumption for reference, tested on a single A100 with 80G.

| period | GPU |
| - | - | 
| load as float32 | ~70G |
| mv transformer and vae to bf16 | ~48G |
| pre compute txt embeddings | ~62G |
| **offload te to cpu** | ~30G |
| training | ~58G |
| validation | ~71G |


> **Gated access**
>
> As the model is gated, before using it with diffusers you first need to go to the [FLUX.1 [dev] Hugging Face page](https://huggingface.co/black-forest-labs/FLUX.1-dev), fill in the form and accept the gate. Once you are in, you need to log in so that your system knows you’ve accepted the gate. Use the command below to log in: `hf auth login`


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

We support dataset formats:
The original dataset is hosted in the [ControlNet repo](https://huggingface.co/lllyasviel/ControlNet/blob/main/training/fill50k.zip). We re-uploaded it to be compatible with `datasets` [here](https://huggingface.co/datasets/fusing/fill50k). Note that `datasets` handles dataloading within the training script. To use our example, add `--dataset_name=fusing/fill50k \` to the script and remove line `--jsonl_for_train` mentioned below.


We also support importing data from jsonl(xxx.jsonl),using `--jsonl_for_train` to enable it, here is a brief example of jsonl files:
```sh
{"image": "xxx", "text": "xxx", "conditioning_image": "xxx"}
{"image": "xxx", "text": "xxx", "conditioning_image": "xxx"}
```

## Training

Our training examples use two test conditioning images. They can be downloaded by running

```sh
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

Then run `hf auth login` to log into your Hugging Face account. This is needed to be able to push the trained ControlNet parameters to Hugging Face Hub.

we can define the num_layers, num_single_layers, which determines the size of the control(default values are num_layers=4, num_single_layers=10)


```bash
accelerate launch train_controlnet_flux.py \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
    --dataset_name=fusing/fill50k \
    --conditioning_image_column=conditioning_image \
    --image_column=image \
    --caption_column=text \
    --output_dir="path to save model" \
    --mixed_precision="bf16" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --max_train_steps=15000 \
    --validation_steps=100 \
    --checkpointing_steps=200 \
    --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
    --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=16 \
    --report_to="wandb" \
    --lr_scheduler="cosine" \
    --num_double_layers=4 \
    --num_single_layers=0 \
    --seed=42 \
    --push_to_hub \
```

To better track our training experiments, we're using the following flags in the command above:

* `report_to="wandb` will ensure the training runs are tracked on Weights and Biases.
* `validation_image`, `validation_prompt`, and `validation_steps` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.

Our experiments were conducted on a single 80GB A100 GPU.

### Inference

Once training is done, we can perform inference like so:

```python
import torch
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model = 'promeai/FLUX.1-controlnet-lineart-promeai'
controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(
    base_model, 
    controlnet=controlnet, 
    torch_dtype=torch.bfloat16
)
# enable memory optimizations   
pipe.enable_model_cpu_offload()

control_image = load_image("https://huggingface.co/promeai/FLUX.1-controlnet-lineart-promeai/resolve/main/images/example-control.jpg")resize((1024, 1024))
prompt = "cute anime girl with massive fluffy fennec ears and a big fluffy tail blonde messy long hair blue eyes wearing a maid outfit with a long black gold leaf pattern dress and a white apron mouth open holding a fancy black forest cake with candles on top in the kitchen of an old dark Victorian mansion lit by candlelight with a bright window to the foggy forest and very expensive stuff everywhere"

image = pipe(
    prompt, 
    control_image=control_image,
    controlnet_conditioning_scale=0.6,
    num_inference_steps=28, 
    guidance_scale=3.5,
).images[0]
image.save("./output.png")
```

## Apply Deepspeed Zero3 

This is an experimental process, I am not sure if it is suitable for everyone, we used this process to successfully train 512 resolution on A100(40g) * 8.
Please modify some of the code in the script.
### 1.Customize zero3 settings

Copy the **accelerate_config_zero3.yaml**,modify `num_processes` according to the number of gpus you want to use:

```bash
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 8
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### 2.Precompute all inputs (latent, embeddings)

In the train_controlnet_flux.py, We need to pre-calculate all parameters and put them into batches.So we first need to rewrite the `compute_embeddings` function. 

```python
def compute_embeddings(batch, proportion_empty_prompts, vae, flux_controlnet_pipeline, weight_dtype, is_train=True):
    
    ### compute text embeddings
    prompt_batch = batch[args.caption_column]
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
    prompt_batch = captions
    prompt_embeds, pooled_prompt_embeds, text_ids = flux_controlnet_pipeline.encode_prompt(
        prompt_batch, prompt_2=prompt_batch
    )
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)
    text_ids = text_ids.to(dtype=weight_dtype)

    # text_ids [512,3] to [bs,512,3]
    text_ids = text_ids.unsqueeze(0).expand(prompt_embeds.shape[0], -1, -1)

    ### compute latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    # vae encode
    pixel_values = batch["pixel_values"]
    pixel_values = torch.stack([image for image in pixel_values]).to(dtype=weight_dtype).to(vae.device)
    pixel_latents_tmp = vae.encode(pixel_values).latent_dist.sample()
    pixel_latents_tmp = (pixel_latents_tmp - vae.config.shift_factor) * vae.config.scaling_factor
    pixel_latents = _pack_latents(
        pixel_latents_tmp,
        pixel_values.shape[0],
        pixel_latents_tmp.shape[1],
        pixel_latents_tmp.shape[2],
        pixel_latents_tmp.shape[3],
    ) 

    control_values = batch["conditioning_pixel_values"]
    control_values = torch.stack([image for image in control_values]).to(dtype=weight_dtype).to(vae.device)
    control_latents = vae.encode(control_values).latent_dist.sample()
    control_latents = (control_latents - vae.config.shift_factor) * vae.config.scaling_factor
    control_latents = _pack_latents(
        control_latents,
        control_values.shape[0],
        control_latents.shape[1],
        control_latents.shape[2],
        control_latents.shape[3],
    )

    # copied from pipeline_flux_controlnet
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)
    latent_image_ids = _prepare_latent_image_ids(
        batch_size=pixel_latents_tmp.shape[0],
        height=pixel_latents_tmp.shape[2],
        width=pixel_latents_tmp.shape[3],
        device=pixel_values.device,
        dtype=pixel_values.dtype,
    )

    # unet_added_cond_kwargs = {"pooled_prompt_embeds": pooled_prompt_embeds, "text_ids": text_ids}
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds, "text_ids": text_ids, "pixel_latents": pixel_latents, "control_latents": control_latents, "latent_image_ids": latent_image_ids}
```

Because we need images to pass through vae, we need to preprocess the images in the dataset first. At the same time, vae requires more gpu memory, so you may need to modify the `batch_size` below
```diff
+train_dataset = prepare_train_dataset(train_dataset, accelerator)
with accelerator.main_process_first():
    from datasets.fingerprint import Hasher

    # fingerprint used by the cache for the other processes to load the result
    # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
    new_fingerprint = Hasher.hash(args)
    train_dataset = train_dataset.map(
-        compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint, batch_size=100
+        compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint, batch_size=10
    )

del text_encoders, tokenizers
gc.collect()
torch.cuda.empty_cache()

# Then get the training dataset ready to be passed to the dataloader.
-train_dataset = prepare_train_dataset(train_dataset, accelerator)
```
### 3.Redefine the behavior of getting batchsize

Now that we have all the preprocessing done, we need to modify the `collate_fn` function.

```python
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    pixel_latents = torch.stack([torch.tensor(example["pixel_latents"]) for example in examples])
    pixel_latents = pixel_latents.to(memory_format=torch.contiguous_format).float()

    control_latents = torch.stack([torch.tensor(example["control_latents"]) for example in examples])
    control_latents = control_latents.to(memory_format=torch.contiguous_format).float()
    
    latent_image_ids= torch.stack([torch.tensor(example["latent_image_ids"]) for example in examples])
    
    prompt_ids = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])

    pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples])
    text_ids = torch.stack([torch.tensor(example["text_ids"]) for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "pixel_latents": pixel_latents,
        "control_latents": control_latents,
        "latent_image_ids": latent_image_ids,
        "prompt_ids": prompt_ids,
        "unet_added_conditions": {"pooled_prompt_embeds": pooled_prompt_embeds, "time_ids": text_ids},
    }
```
Finally, we just need to modify the way of obtaining various parameters during training.
```python
for epoch in range(first_epoch, args.num_train_epochs):
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(flux_controlnet):
            # Convert images to latent space
            pixel_latents = batch["pixel_latents"].to(dtype=weight_dtype)
            control_image = batch["control_latents"].to(dtype=weight_dtype)
            latent_image_ids = batch["latent_image_ids"].to(dtype=weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(pixel_latents).to(accelerator.device).to(dtype=weight_dtype)
            bsz = pixel_latents.shape[0]

            # Sample a random timestep for each image
            t = torch.sigmoid(torch.randn((bsz,), device=accelerator.device, dtype=weight_dtype))

            # apply flow matching
            noisy_latents = (
                1 - t.unsqueeze(1).unsqueeze(2).repeat(1, pixel_latents.shape[1], pixel_latents.shape[2])
            ) * pixel_latents + t.unsqueeze(1).unsqueeze(2).repeat(
                1, pixel_latents.shape[1], pixel_latents.shape[2]
            ) * noise

            guidance_vec = torch.full(
                (noisy_latents.shape[0],), 3.5, device=noisy_latents.device, dtype=weight_dtype
            )

            controlnet_block_samples, controlnet_single_block_samples = flux_controlnet(
                hidden_states=noisy_latents,
                controlnet_cond=control_image,
                timestep=t,
                guidance=guidance_vec,
                pooled_projections=batch["unet_added_conditions"]["pooled_prompt_embeds"].to(dtype=weight_dtype),
                encoder_hidden_states=batch["prompt_ids"].to(dtype=weight_dtype),
                txt_ids=batch["unet_added_conditions"]["time_ids"][0].to(dtype=weight_dtype),
                img_ids=latent_image_ids[0],
                return_dict=False,
            )

            noise_pred = flux_transformer(
                hidden_states=noisy_latents,
                timestep=t,
                guidance=guidance_vec,
                pooled_projections=batch["unet_added_conditions"]["pooled_prompt_embeds"].to(dtype=weight_dtype),
                encoder_hidden_states=batch["prompt_ids"].to(dtype=weight_dtype),
                controlnet_block_samples=[sample.to(dtype=weight_dtype) for sample in controlnet_block_samples]
                if controlnet_block_samples is not None
                else None,
                controlnet_single_block_samples=[
                    sample.to(dtype=weight_dtype) for sample in controlnet_single_block_samples
                ]
                if controlnet_single_block_samples is not None
                else None,
                txt_ids=batch["unet_added_conditions"]["time_ids"][0].to(dtype=weight_dtype),
                img_ids=latent_image_ids[0],
                return_dict=False,
            )[0]
```
Congratulations! You have completed all the required code modifications required for deepspeedzero3.

### 4.Training with deepspeedzero3

Start!!!

```bash
export pretrained_model_name_or_path='flux-dev-model-path'
export MODEL_TYPE='train_model_type'
export TRAIN_JSON_FILE="your_json_file"
export CONTROL_TYPE='control_preprocessor_type'
export CAPTION_COLUMN='caption_column'

export CACHE_DIR="/data/train_csr/.cache/huggingface/"
export OUTPUT_DIR='/data/train_csr/FLUX/MODEL_OUT/'$MODEL_TYPE
# The first step is to use Python to precompute all caches.Replace the first line below with this line. (I am not sure why using accelerate would cause problems.)

CUDA_VISIBLE_DEVICES=0 python3 train_controlnet_flux.py \

# The second step is to use the above accelerate config to train
accelerate  launch  --config_file "./accelerate_config_zero3.yaml" train_controlnet_flux.py \
    --pretrained_model_name_or_path=$pretrained_model_name_or_path \
    --jsonl_for_train=$TRAIN_JSON_FILE \
    --conditioning_image_column=$CONTROL_TYPE \
    --image_column=image \
    --caption_column=$CAPTION_COLUMN\
    --cache_dir=$CACHE_DIR \
    --tracker_project_name=$MODEL_TYPE \
    --output_dir=$OUTPUT_DIR \
    --max_train_steps=500000 \
    --mixed_precision bf16 \
    --checkpointing_steps=1000 \
    --gradient_accumulation_steps=8 \
    --resolution=512 \
    --train_batch_size=1 \
    --learning_rate=1e-5 \
    --num_double_layers=4 \
    --num_single_layers=0 \
    --gradient_checkpointing \
    --resume_from_checkpoint="latest" \
    # --use_adafactor \ dont use
    # --validation_steps=3 \ not support 
    # --validation_image $VALIDATION_IMAGE \ not support 
    # --validation_prompt "xxx" \ not support 
```