# Training Control LoRA with Flux

This example shows how train Control LoRA with Flux to condition it with additional structural controls (like depth maps, poses, etc.). 

This is still an experimental version and the following differences exist:

* No use of bias on `lora_B`.
* Mo updates on the norm scales.

We simply expand the input channels of Flux.1 Dev from 64 to 128 to allow for additional inputs and then train a regular LoRA on top of it. To account for the newly added input channels, we additional append a LoRA on the underlying layer (`x_embedder`). Inference, however, is performed with the `FluxControlPipeline`. 

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

You need to install `diffusers` from the branch of [this PR](https://github.com/huggingface/diffusers/pull/9999). 