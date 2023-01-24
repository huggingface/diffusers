# Retrieval Augmented Diffusion
## Install
```
conda install faiss-gpu cudatoolkit=11.6
pip install accelerate transformers timm fairscale albumentations wandb datasets clip-retrieval triton
```
To install xformers on windows, go [here](https://github.com/facebookresearch/xformers/actions/runs/3543179717) and download windows-2019.zip file and pip install the file that corresponds to the version that fits.
## Inference
```
python rdm_inference.py --pretrained_model_name_or_path="fusing/rdm" --dataset_name="Isamu136/oxford_pets_with_l14_emb" --image_column="image" --caption_column="label" --resolution=768 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=15000 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="pet-model" --dataset_save_path="./data/oxford_pets" --center_crop --img_dir="frida" --prompt="A photo of dog" --num_queries=4 --num_log_imgs=3
```
## Training
If you want to train it's recommended to do it with colossalai as without it, the training can exceed 24gb vram. With colossal ai it can be cut down to around 6gb requirement. The command to run is below. However, this only works with ubuntu and currently there's a bug where when images are logged or the model is saved, it'll crash.
```
torchrun --nproc_per_node 1 train_rdm_colossalai.py --pretrained_model_name_or_path="fusing/rdm" --dataset_name="Isamu136/oxford_pets_with_l14_emb" --image_column="image" --caption_column="label" --resolution=768 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=15000 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="pet-model" --dataset_save_path="./data/oxford_pets" --center_crop --placement="cuda"
```

However, if you don't have ubuntu, you can try

```
accelerate launch train_rdm.py --pretrained_model_name_or_path="fusing/rdm" --dataset_name="Isamu136/oxford_pets_with_l14_emb" --image_column="image" --caption_column="label" --resolution=768 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=15000 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="pet-model" --dataset_save_path="./data/oxford_pets" --center_crop --use_ema
```
If you also want to train with clip-retrieval
```
accelerate launch train_rdm.py --pretrained_model_name_or_path="fusing/rdm" --dataset_name="Isamu136/oxford_pets_with_l14_emb" --image_column="image" --caption_column="label" --resolution=768 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=15000 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="pet-model" --dataset_save_path="./data/oxford_pets" --center_crop --use_ema --use_clip_retrieval
```