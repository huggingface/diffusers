# Training Control LoRA with Flux

This (experimental) example shows how train Control LoRA with [Flux](https://huggingface.co/black-forest-labs/FLUX.1-dev) to condition it with additional structural controls (like depth maps, poses, etc.). 

We simply expand the input channels of Flux.1 Dev from 64 to 128 to allow for additional inputs and then train a regular LoRA on top of it. To account for the newly added input channels, we additional append a LoRA on the underlying layer (`x_embedder`). Inference, however, is performed with the `FluxControlPipeline`. 

> [!NOTE]
> **Gated model**
>
> As the model is gated, before using it with diffusers you first need to go to the [FLUX.1 [dev] Hugging Face page](https://huggingface.co/black-forest-labs/FLUX.1-dev), fill in the form and accept the gate. Once you are in, you need to log in so that your system knows you’ve accepted the gate. Use the command below to log in:

```bash
huggingface-cli login
```

Example command:

```bash
accelerate launch train_control_lora_flux.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --dataset_name="raulc0399/open_pose_controlnet" \
  --output_dir="pose-control-lora" \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --rank=64 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --validation_image="openpose.png" \
  --validation_prompt="A couple, 4k photo, highly detailed" \
  --seed="0" \
  --push_to_hub
```

`openpose.png` comes from [here](https://huggingface.co/Adapter/t2iadapter/resolve/main/openpose.png).

You need to install `diffusers` from the branch of [this PR](https://github.com/huggingface/diffusers/pull/9999). When it's merged, you should install `diffusers` from the `main`.

The training script exposes additional CLI args that might be useful to experiment with:

* `use_lora_bias`: When set, additionally trains the biases of the `lora_B` layer. 
* `train_norm_layers`: When set, additionally trains the normalization scales. Takes care of saving and loading.
* `lora_layers`: Specify the layers you want to apply LoRA to. If you specify "all-linear", all the linear layers will be LoRA-attached.

## Training with DeepSpeed

It's possible to train with [DeepSpeed](https://github.com/microsoft/DeepSpeed), specifically leveraging the Zero2 system optimization. To use it, save the following config to an YAML file (feel free to modify as needed):

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

And then while launching training, pass the config file:

```bash
accelerate launch --config_file=CONFIG_FILE.yaml ...
```

## Full fine-tuning

We provide a non-LoRA version of the training script `train_control_flux.py`. Here is an example command:

```bash
accelerate launch --config_file=accelerate_ds2.yaml train_control_flux.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --dataset_name="raulc0399/open_pose_controlnet" \
  --output_dir="pose-control" \
  --mixed_precision="bf16" \
  --train_batch_size=2 \
  --dataloader_num_workers=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --proportion_empty_prompts=0.2 \
  --learning_rate=5e-5 \
  --adam_weight_decay=1e-4 \
  --set_grads_to_none \
  --report_to="wandb" \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=1000 \
  --checkpointing_steps=1000 \
  --max_train_steps=10000 \
  --validation_steps=200 \
  --validation_image "2_pose_1024.jpg" "3_pose_1024.jpg" \
  --validation_prompt "two friends sitting by each other enjoying a day at the park, full hd, cinematic" "person enjoying a day at the park, full hd, cinematic" \
  --seed="0" \
  --push_to_hub
```

Change the `validation_image` and `validation_prompt` as needed.