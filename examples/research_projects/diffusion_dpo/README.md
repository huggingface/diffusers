# Diffusion Model Alignment Using Direct Preference Optimization

This directory provides LoRA implementations of Diffusion DPO proposed in [DiffusionModel Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908) by Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik.

We provide implementations for both Stable Diffusion (SD) and Stable Diffusion XL (SDXL). The original checkpoints are available at the URLs below:

* [mhdang/dpo-sd1.5-text2image-v1](https://huggingface.co/mhdang/dpo-sd1.5-text2image-v1)
* [mhdang/dpo-sdxl-text2image-v1](https://huggingface.co/mhdang/dpo-sdxl-text2image-v1)

> 💡 Note: The scripts are highly experimental and were only tested on low-data regimes. Proceed with caution. Feel free to let us know about your findings via GitHub issues.

## SD training command

```bash
accelerate launch train_diffusion_dpo.py \
  --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5  \
  --output_dir="diffusion-dpo" \
  --mixed_precision="fp16" \
  --dataset_name=kashif/pickascore \
  --resolution=512 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rank=8 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10000 \
  --checkpointing_steps=2000 \
  --run_validation --validation_steps=200 \
  --seed="0" \
  --report_to="wandb" \
  --push_to_hub
```

## SDXL training command

```bash
accelerate launch train_diffusion_dpo_sdxl.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0  \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --output_dir="diffusion-sdxl-dpo" \
  --mixed_precision="fp16" \
  --dataset_name=kashif/pickascore \
  --train_batch_size=8 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rank=8 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --checkpointing_steps=500 \
  --run_validation --validation_steps=50 \
  --seed="0" \
  --report_to="wandb" \
  --push_to_hub
```

## SDXL Turbo training command

```bash
accelerate launch train_diffusion_dpo_sdxl.py \
  --pretrained_model_name_or_path=stabilityai/sdxl-turbo \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --output_dir="diffusion-sdxl-turbo-dpo" \
  --mixed_precision="fp16" \
  --dataset_name=kashif/pickascore \
  --train_batch_size=8 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rank=8 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --checkpointing_steps=500 \
  --run_validation --validation_steps=50 \
  --seed="0" \
  --report_to="wandb" \
  --is_turbo --resolution 512 \
  --push_to_hub
```


## Acknowledgements

This is based on the amazing work done by [Bram](https://github.com/bram-w) here for Diffusion DPO: https://github.com/bram-w/trl/blob/dpo/.
