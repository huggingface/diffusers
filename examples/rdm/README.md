# Retrieval Augmented Diffusion
## Install
```
conda install faiss-gpu -c pytorch
pip install accelerate transformers timm fairscale albumentations wandb datasets clip-retrieval triton
```
To install xformers on windows, go [here](https://github.com/facebookresearch/xformers/actions/runs/3543179717) and download windows-2019.zip file and pip install the file that corresponds to the version that fits.
If you encounter module 'faiss._swigfaiss_avx2' has no attribute 'delete_GpuResourcesVector',
do
```
pip uninstall faiss
conda uninstall faiss-gpu -c pytorch
conda install faiss-gpu -c pytorch
```
should fix this
## Inference
```
python rdm_inference.py --pretrained_model_name_or_path="fusing/rdm" --dataset_name="Isamu136/oxford_pets_with_l14_emb" --image_column="image" --caption_column="label" --resolution=768 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=15000 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="pet-model" --dataset_save_path="./data/oxford_pets" --center_crop --img_dir="frida" --prompt="A photo of dog" --num_queries=4 --num_log_imgs=3
```