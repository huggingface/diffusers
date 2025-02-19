# AutoencoderKL training example

## Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then cd in the example folder  and run
```bash
pip install -r requirements.txt
```


And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

## Training on CIFAR10

Please replace the validation image with your own image.

```bash
accelerate launch train_autoencoderkl.py \
    --pretrained_model_name_or_path stabilityai/sd-vae-ft-mse \
    --dataset_name=cifar10 \
    --image_column=img \
    --validation_image images/bird.jpg images/car.jpg images/dog.jpg images/frog.jpg \
    --num_train_epochs 100 \
    --gradient_accumulation_steps 2 \
    --learning_rate 4.5e-6 \
    --lr_scheduler cosine \
    --report_to wandb \
```

## Training on ImageNet

```bash
accelerate launch train_autoencoderkl.py \
    --pretrained_model_name_or_path stabilityai/sd-vae-ft-mse \
    --num_train_epochs 100 \
    --gradient_accumulation_steps 2 \
    --learning_rate 4.5e-6 \
    --lr_scheduler cosine \
    --report_to wandb \
    --mixed_precision bf16 \
    --train_data_dir /path/to/ImageNet/train \
    --validation_image ./image.png \
    --decoder_only
```
