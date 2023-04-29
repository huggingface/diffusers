# test CM
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --multi_gpu ./examples/unconditional_image_generation/train_unconditional_consistency_models_CT.py --train_data_dir="datasets/valid_64x64" --resolution=64 --center_crop --random_flip --output_dir="ddpm-ema-flowers-64" --train_batch_size=32 --num_epochs=100 --gradient_accumulation_steps=1 --use_ema --learning_rate=1e-4 --lr_warmup_steps=500 


CUDA_VISIBLE_DEVICES=6,7 accelerate launch --multi_gpu ./examples/unconditional_image_generation/train_unconditional_consistency_models_CT.py --train_data_dir="datasets/valid_64x64" --resolution=64 --center_crop --random_flip --output_dir="ddpm-ema-flowers-64" --train_batch_size=16 --num_epochs=100 --gradient_accumulation_steps=1 --use_ema --learning_rate=1e-4 --lr_warmup_steps=500 

--model_config_name_or_path CMPipeline












pipeline_stochastic_karras_ve

src/diffusers/pipelines/stochastic_karras_ve/pipeline_stochastic_karras_ve.py


CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu ./examples/unconditional_image_generation/train_unconditional.py --train_data_dir="datasets/valid_64x64" --resolution=64 --center_crop --random_flip --output_dir="ddpm-ema-flowers-64" --train_batch_size=16 --num_epochs=100 --gradient_accumulation_steps=1 --use_ema --learning_rate=1e-4 --lr_warmup_steps=500 --model_config_name_or_path KarrasVePipeline







CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu ./examples/unconditional_image_generation/train_unconditional.py --train_data_dir="datasets/valid_64x64" --resolution=64 --center_crop --random_flip --output_dir="ddpm-ema-flowers-64" --train_batch_size=16 --num_epochs=100 --gradient_accumulation_steps=1 --use_ema --learning_rate=1e-4 --lr_warmup_steps=500







CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --mixed_precision="fp16" --multi_gpu ./examples/unconditional_image_generation/train_unconditional.py --train_data_dir="datasets/valid_64x64" --resolution=64 --center_crop --random_flip --output_dir="ddpm-ema-flowers-64" --train_batch_size=16 --num_epochs=100 --gradient_accumulation_steps=1 --use_ema --learning_rate=1e-4 --lr_warmup_steps=500 --mixed_precision="fp16" 


--push_to_hub



accelerate launch --mixed_precision="fp16" --multi_gpu train_unconditional.py \
  --dataset_name="huggan/pokemon" \
  --resolution=64 --center_crop --random_flip \
  --output_dir="ddpm-ema-pokemon-64" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision="fp16" \
  --logger="wandb"