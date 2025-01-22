accelerate launch train_autoencoderkl.py \
    --pretrained_model_name_or_path stabilityai/sd-vae-ft-mse \
    --dataset_name=cifar10 \
    --image_column=img \
    --validation_image /home/azureuser/v-yuqianhong/ImageNet/ILSVRC2012/val/n01491361/ILSVRC2012_val_00002922.JPEG \
    --num_train_epochs 100 \
    --gradient_accumulation_steps 2 \
    --learning_rate 4.5e-6 \
    --lr_scheduler cosine \
    --report_to wandb \