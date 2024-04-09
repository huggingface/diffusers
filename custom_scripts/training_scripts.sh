nohup accelerate launch --num_cpu_threads_per_process 1 train_controlnet_sdxl.py --config_file=configs/controlnet_sdxl_training.yaml > train_controlnet_sdxl.out &

# canny format
nohup accelerate launch --num_cpu_threads_per_process 1 train_controlnet_sdxl.py --config_file=configs/controlnet_sdxl_canny_format_training.yaml &



accelerate launch examples/controlnet/train_controlnet_sdxl.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"
--output_dir="checkpoints/jector_inpaint_controlnet_exp"
--dataset_name="/home/gkalstn000/diffusers/datasets/inpainting"
--caption_column="text"
--conditioning_image_column=conditioning_image
--mixed_precision="fp16"
--resolution=1024
--learning_rate=1e-5
--max_train_steps=15000
--validation_image
"./conditioning_image_1.png"
"./conditioning_image_2.png"
--validation_prompt
"red circle with blue background"
"cyan circle with brown floral background"
--validation_steps=100
--train_batch_size=1
--gradient_accumulation_steps=4
--report_to="wandb"
--seed=42
--checkpointing_steps=1