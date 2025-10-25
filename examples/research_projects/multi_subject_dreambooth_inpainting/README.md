# Multi Subject Dreambooth for Inpainting Models

Please note that this project is not actively maintained. However, you can open an issue and tag @gzguevara.

[DreamBooth](https://huggingface.co/papers/2208.12242) is a method to personalize text2image models like stable diffusion given just a few(3~5) images of a subject. This project consists of **two parts**. Training Stable Diffusion for inpainting requires prompt-image-mask pairs. The Unet of inpainiting models have 5 additional input channels (4 for the encoded masked-image and 1 for the mask itself).

**The first part**, the `multi_inpaint_dataset.ipynb` notebook, demonstrates how make a 🤗 dataset of prompt-image-mask pairs. You can, however, skip the first part and move straight to the second part with the example datasets in this project. ([cat toy dataset masked](https://huggingface.co/datasets/gzguevara/cat_toy_masked), [mr. potato head dataset masked](https://huggingface.co/datasets/gzguevara/mr_potato_head_masked))

**The second part**, the `train_multi_subject_inpainting.py` training script, demonstrates how to implement a training procedure for one or more subjects and adapt it for stable diffusion for inpainting.

## 1. Data Collection: Make Prompt-Image-Mask Pairs

 Earlier training scripts have provided approaches like random masking for the training images. This project provides a notebook for more precise mask setting.

The notebook can be found here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JNEASI_B7pLW1srxhgln6nM0HoGAQT32?usp=sharing)

The `multi_inpaint_dataset.ipynb` notebook, takes training & validation images, on which the user draws masks and provides prompts to make a prompt-image-mask pairs. This ensures that during training, the loss is computed on the area masking the object of interest, rather than on random areas. Moreover, the `multi_inpaint_dataset.ipynb` notebook allows you to build a validation dataset with corresponding masks for monitoring the training process. Example below:

![train_val_pairs](https://drive.google.com/uc?id=1PzwH8E3icl_ubVmA19G0HZGLImFX3x5I)

You can build multiple datasets for every subject and upload them to the 🤗 hub. Later, when launching the training script you can indicate the paths of the datasets, on which you would like to finetune Stable Diffusion for inpaining.

## 2. Train Multi Subject Dreambooth for Inpainting

### 2.1. Setting The Training Configuration

Before launching the training script, make sure to select the inpainting the target model, the output directory and the 🤗 datasets.

```bash
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-inpainting"
export OUTPUT_DIR="path-to-save-model"

export DATASET_1="gzguevara/mr_potato_head_masked"
export DATASET_2="gzguevara/cat_toy_masked"
... # Further paths to 🤗 datasets
```

### 2.2. Launching The Training Script

```bash
accelerate launch train_multi_subject_dreambooth_inpaint.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir $DATASET_1 $DATASET_2 \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --learning_rate=3e-6 \
  --max_train_steps=500 \
  --report_to_wandb
```

### 2.3. Fine-tune text encoder with the UNet.

The script also allows to fine-tune the `text_encoder` along with the `unet`. It's been observed experimentally that fine-tuning `text_encoder` gives much better results especially on faces.
Pass the `--train_text_encoder` argument to the script to enable training `text_encoder`.

___Note: Training text encoder requires more memory, with this option the training won't fit on 16GB GPU. It needs at least 24GB VRAM.___

```bash
accelerate launch train_multi_subject_dreambooth_inpaint.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir $DATASET_1 $DATASET_2 \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --learning_rate=2e-6 \
  --max_train_steps=500 \
  --report_to_wandb \
  --train_text_encoder
```

## 3. Results

A [![Weights & Biases](https://img.shields.io/badge/Weights%20&%20Biases-Report-blue)](https://wandb.ai/gzguevara/uncategorized/reports/Multi-Subject-Dreambooth-for-Inpainting--Vmlldzo2MzY5NDQ4?accessToken=y0nya2d7baguhbryxaikbfr1203amvn1jsmyl07vk122mrs7tnph037u1nqgse8t) is provided showing the training progress by every 50 steps. Note, the reported weights & biases run was performed on a A100 GPU with the following stetting:

```bash
accelerate launch train_multi_subject_dreambooth_inpaint.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir $DATASET_1 $DATASET_2 \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=10 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --max_train_steps=500 \
  --report_to_wandb \
  --train_text_encoder
```
Here you can see the target objects on my desk and next to my plant:

![Results](https://drive.google.com/uc?id=1kQisOiiF5cj4rOYjdq8SCZenNsUP2aK0)
