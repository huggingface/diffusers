# Scheduled Pseudo-Huber Loss for Diffusers

These are the modifications of to include the possibility of training text2image models with Scheduled Pseudo Huber loss, introduced in https://huggingface.co/papers/2403.16728. (https://github.com/kabachuha/SPHL-for-stable-diffusion)

## Why this might be useful?

- If you suspect that the part of the training dataset might be corrupted, and you don't want these outliers to distort the model's supposed output

- If you want to improve the aesthetic quality of pictures by helping the model disentangle concepts and be less influenced by another sorts of pictures.

See https://github.com/huggingface/diffusers/issues/7488 for the detailed description.

## Instructions

The same usage as in the case of the corresponding vanilla Diffusers scripts https://github.com/huggingface/diffusers/tree/main/examples
