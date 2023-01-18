# Retrieval Augmented Diffusion
## Install
```
conda install faiss-gpu cudatoolkit=11.6
pip install datasets clip-retrieval
```
To install xformers on windows, go [here](https://github.com/facebookresearch/xformers/actions/runs/3543179717) and download windows-2019.zip file and pip install the file that corresponds to the version that fits.
## Run
If you want to train do
```
```
accelerate launch train_rdm_colossalai.py --pretrained_model_name_or_path="fusing/rdm" --dataset_name="pcuenq/oxford-pets" --image_column="image" --caption_column="label" --resolution=728 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=15000 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="pet-model" --dataset_save_path="./data/oxford_pets" --center_crop --use_ema
```
If you want to do clip-retrieval
```
accelerate launch train_rdm.py --pretrained_model_name_or_path="fusing/rdm" --dataset_name="pcuenq/oxford-pets" --image_column="image" --caption_column="label" --resolution=512 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=15000 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="pet-model" --dataset_save_path="./data/oxford_pets" --center_crop --use_ema --use_clip_retrieval
```
Training on colossalai. If torchrun doesn't work do 
python -m torch.distributed.launch
```
torchrun --nproc_per_node 1 train_rdm_colossalai.py --pretrained_model_name_or_path="fusing/rdm" --dataset_name="pcuenq/oxford-pets" --image_column="image" --caption_column="label" --resolution=728 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=15000 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="pet-model" --dataset_save_path="./data/oxford_pets" --center_crop --placement="cuda"
```
For training on 6gb, it's unstable but you can try
```
accelerate launch train_rdm.py --pretrained_model_name_or_path="fusing/rdm" --dataset_name="pcuenq/oxford-pets" --image_column="image" --caption_column="label" --resolution=512 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=15000 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="sd-pokemon-model" --dataset_save_path="./data/oxford_pets" --center_crop --enable_xformers_memory_efficient_attention --clip_model="openai/clip-vit-base-patch32" --revision="fp16" --unet_config="config.json" --gradient_checkpointing
```