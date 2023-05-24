python train_unconditional.py \
  --dataset_name="huggan/pokemon" \
  --resolution=64 --center_crop --random_flip \
  --output_dir="ddpm-ema-pokemon-64" \
  --train_batch_size=64 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  # --push_to_hub