#!/bin/bash

## 1. Extract features from ImageNet
#torchrun --nnodes=1 --nproc_per_node=8 extract_features.py --model DiT-XL/2 --data-path <your imagenet data path> --features-path <your feature saving path>

## 2. Train UFO on ImageNet
accelerate launch --main_process_port 60882 --multi_gpu --num_processes 8 --mixed_precision bf16 train_ufo.py --model DiT-XL/2 --feature-path <your feature saving path> --pretrained-ckpt-dir <your ptrained DiT-XL/2 ckpt> --global-batch-size 512 --num_times 4