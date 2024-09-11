# Stable Diffusion text-to-image fine-tuning using PyTorch/XLA

The `train_text_to_image_xla.py` script shows how to fine-tune stable diffusion model on TPU devices using PyTorch/XLA.

It has been tested on v4 and v5p TPU versions. Training code has been tested on multi-host. 

This script implements Distributed Data Parallel using GSPMD feature in XLA compiler
where we shard the input batches over the TPU devices. 

As of 9-11-2024, these are some expected step times.

| accelerator | global batch size | step time (seconds) |
| ----------- | ----------------- | --------- |
| v5p-128 | 1024 | 0.245 |
| v5p-256 | 2048 | 0.234 |
| v5p-512 | 4096 | 0.2498 |

## Create TPU

To create a TPU on Google Cloud first set these environment variables:

```bash
export TPU_NAME=<tpu-name>
export PROJECT_ID=<project-id>
export ZONE=<google-cloud-zone>
export ACCELERATOR_TYPE=<accelerator type like v5p-8>
export RUNTIME_VERSION=<runtime version like v2-alpha-tpuv5 for v5p>
```

Then run the create TPU command:
```bash
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} --project ${PROJECT_ID} 
--zone ${ZONE} --accelerator-type ${ACCELERATOR_TYPE} --version ${RUNTIME_VERSION} 
--reserved
```

You can also use other ways to reserve TPUs like GKE or queued resources.

## Setup TPU environment

Install PyTorch and PyTorch/XLA nightly versions:
```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--project=${PROJECT_ID} --zone=${ZONE} --worker=all \
--command='
pip3 install --pre torch==2.5.0.dev20240905+cpu torchvision==0.20.0.dev20240905+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install "torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.5.0.dev20240905-cp310-cp310-linux_x86_64.whl" -f https://storage.googleapis.com/libtpu-releases/index.html
'
```

Verify that PyTorch and PyTorch/XLA were installed correctly:

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--project ${PROJECT_ID} --zone ${ZONE} --worker=all \
--command='python3 -c "import torch; import torch_xla;"'
```

Install dependencies:
```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--project=${PROJECT_ID} --zone=${ZONE} --worker=all \
--command='
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git checkout main
cd examples/research_projects/pytorch_xla
pip3 install -r requirements.txt
pip3 install pillow --upgrade
cd ../../..
pip3 install .'
```

## Run the training job

### Authenticate

Run the following command to authenticate your token.

```bash
huggingface-cli login
```

This script only trains the unet part of the network. The VAE and text encoder
are fixed.

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--project=${PROJECT_ID} --zone=${ZONE} --worker=all \
--command='
export XLA_DISABLE_FUNCTIONALIZATION=1 
export PROFILE_DIR=/tmp/
export CACHE_DIR=/tmp/
export DATASET_NAME=lambdalabs/naruto-blip-captions
export PER_HOST_BATCH_SIZE=32 # This is known to work on TPU v4. Can set this to 64 for TPU v5p
export TRAIN_STEPS=50
export OUTPUT_DIR=/tmp/trained-model/
python diffusers/examples/research_projects/pytorch_xla/train_text_to_image_xla.py --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-base --dataset_name=$DATASET_NAME --resolution=512 --center_crop --random_flip --train_batch_size=$PER_HOST_BATCH_SIZE  --max_train_steps=$TRAIN_STEPS --learning_rate=1e-06 --mixed_precision=bf16 --profile_duration=80000 --output_dir=$OUTPUT_DIR --dataloader_num_workers=4 --loader_prefetch_size=4 --device_prefetch_size=4'
   
```

### Environment Envs Explained

*   `XLA_DISABLE_FUNCTIONALIZATION`: To optimize the performance for AdamW optimizer.
*   `PROFILE_DIR`: Specify where to put the profiling results.
*   `CACHE_DIR`: Directory to store XLA compiled graphs for persistent caching.
*   `DATASET_NAME`: Dataset to train the model. 
*   `PER_HOST_BATCH_SIZE`: Size of the batch to load per CPU host. For e.g. for a v5p-16 with 2 CPU hosts, the global batch size will be 2xPER_HOST_BATCH_SIZE. The input batch is sharded along the batch axis.
*    `TRAIN_STEPS`: Total number of training steps to run the training for.
*    `OUTPUT_DIR`: Directory to store the fine-tuned model.

## Run inference using the output model

To run inference using the output, you can simply load the model and pass it
input prompts. The first pass will compile the graph and takes longer with the following passes running much faster.

```bash
export CACHE_DIR=/tmp/
```

```python
import torch
import os
import sys
import  numpy as np

import torch_xla.core.xla_model as xm
from time import time
from diffusers import StableDiffusionPipeline
import torch_xla.runtime as xr

CACHE_DIR = os.environ.get("CACHE_DIR", None)
if CACHE_DIR:
    xr.initialize_cache(CACHE_DIR, readonly=False)

def main():
    device = xm.xla_device()
    model_path = "jffacevedo/pxla_trained_model"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16
    )
    pipe.to(device)
    prompt = ["A naruto with green eyes and red legs."]
    start = time()
    print("compiling...")
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    print(f"compile time: {time() - start}")
    print("generate...")
    start = time()
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    print(f"generation time (after compile) : {time() - start}")
    image.save("naruto.png")

if __name__ == '__main__':
    main()
```

Expected Results:

```bash
compiling...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [10:03<00:00, 20.10s/it]
compile time: 720.656970500946
generate...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:01<00:00, 17.65it/s]
generation time (after compile) : 1.8461642265319824