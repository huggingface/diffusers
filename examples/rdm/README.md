# Retrieval Augmented Diffusion

To run
```
accelerate launch train_rdm.py --pretrained_model_name_or_path="fusing/rdm" --dataset_name="lambdalabs/pokemon-blip-captions" --use_ema --resolution=512 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=15000 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="sd-pokemon-model" --use_clip_retrieval --dataset_save_path="./data/pokemon_embedding" --center_crop
```