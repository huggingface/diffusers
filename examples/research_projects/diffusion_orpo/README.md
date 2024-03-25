This project is an attempt to check if it's possible to apply to [ORPO](https://arxiv.org/abs/2403.07691) on a text-conditioned diffusion model to align it on preference data WITHOUT a reference model. The implementation is based on https://github.com/huggingface/trl/pull/1435/. 

> [!WARNING] 
> We assume that MSE in the diffusion formulation approximates the log-probs as required by ORPO (hat-tip to [@kashif](https://github.com/kashif) for the idea). So, please consider this to be extremely experimental.

## Training

Here's training command you can use on a 40GB A100 to validate things on a [small preference
dataset](https://hf.co/datasets/kashif/pickascore): 

```bash
accelerate launch train_diffusion_orpo_sdxl_lora.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0  \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --output_dir="diffusion-sdxl-orpo" \
  --mixed_precision="fp16" \
  --dataset_name=kashif/pickascore \
  --train_batch_size=8 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rank=8 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --checkpointing_steps=500 \
  --run_validation --validation_steps=50 \
  --seed="0" \
  --report_to="wandb" \
  --push_to_hub
```

We also provide a simple script to scale up the training on the [yuvalkirstain/pickapic_v2](https://huggingface.co/datasets/yuvalkirstain/pickapic_v2) dataset:

```bash
accelerate launch --multi_gpu train_diffusion_orpo_sdxl_lora_wds.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0  \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --dataset_path="pipe:aws s3 cp s3://diffusion-preference-opt/{00000..00644}.tar -" \
  --output_dir="diffusion-sdxl-orpo-wds" \
  --mixed_precision="fp16" \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rank=8 \
  --dataloader_num_workers=8 \
  --learning_rate=3e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50000 \
  --checkpointing_steps=2000 \
  --run_validation --validation_steps=500 \
  --seed="0" \
  --report_to="wandb" \
  --push_to_hub
```

We tested the above on a node of 8 H100s but it should also work on A100s. It requires the `webdataset` library for faster dataloading. Note that we kept the dataset shards on an S3 bucket but it should be also possible to have them stored locally. 

You can use the code below to convert the original dataset into `webdataset` shards:

```python
import os
import io
import ray
import webdataset as wds
from datasets import Dataset
from PIL import Image

ray.init(num_cpus=8)


def convert_to_image(im_bytes):
    return Image.open(io.BytesIO(im_bytes)).convert("RGB")

def main():
    dataset_path = "/pickapic_v2/data"
    wds_shards_path = "/pickapic_v2_webdataset"
    # get all .parquet files in the dataset path
    dataset_files = [
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path)
        if f.endswith(".parquet")
    ]

    @ray.remote
    def create_shard(path):
        # get basename of the file
        basename = os.path.basename(path)
        # get the shard number data-00123-of-01034.parquet -> 00123
        shard_num = basename.split("-")[1]
        dataset = Dataset.from_parquet(path)
        # create a webdataset shard
        shard = wds.TarWriter(os.path.join(wds_shards_path, f"{shard_num}.tar"))
        
        for i, example in enumerate(dataset):
            wds_example = {
                "__key__": str(i),
                "original_prompt.txt": example["caption"],
                "jpg_0.jpg": convert_to_image(example["jpg_0"]),
                "jpg_1.jpg": convert_to_image(example["jpg_1"]),
                "label_0.txt": str(example["label_0"]),
                "label_1.txt": str(example["label_1"])
            }
            shard.write(wds_example)
        shard.close()

    futures = [create_shard.remote(path) for path in dataset_files]
    ray.get(futures)


if __name__ == "__main__":
    main()
```

## Inference

Refer to [sayakpaul/diffusion-sdxl-orpo](https://huggingface.co/sayakpaul/diffusion-sdxl-orpo) for an experimental checkpoint.