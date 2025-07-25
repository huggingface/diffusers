# GLIGEN: Open-Set Grounded Text-to-Image Generation

These scripts contain the code to prepare the grounding data and train the GLIGEN model on COCO dataset.

### Install the requirements

```bash
conda create -n diffusers python==3.10
conda activate diffusers
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a notebook

```python
from accelerate.utils import write_basic_config

write_basic_config()
```

### Prepare the training data

If you want to make your own grounding data, you need to install the requirements.

I used [RAM](https://github.com/xinyu1205/recognize-anything) to tag
images, [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO/issues?q=refer) to detect objects,
and [BLIP2](https://huggingface.co/docs/transformers/en/model_doc/blip-2) to caption instances.

Only RAM needs to be installed manually:

```bash
pip install git+https://github.com/xinyu1205/recognize-anything.git --no-deps
```

Download the pre-trained model:

```bash
hf download --resume-download xinyu1205/recognize_anything_model ram_swin_large_14m.pth
hf download --resume-download IDEA-Research/grounding-dino-base
hf download --resume-download Salesforce/blip2-flan-t5-xxl
hf download --resume-download clip-vit-large-patch14
hf download --resume-download masterful/gligen-1-4-generation-text-box
```

Make the training data on 8 GPUs:

```bash
torchrun --master_port 17673 --nproc_per_node=8 make_datasets.py \
    --data_root /mnt/workspace/workgroup/zhizhonghuang/dataset/COCO/train2017 \
    --save_root /root/gligen_data \
    --ram_checkpoint /root/.cache/huggingface/hub/models--xinyu1205--recognize_anything_model/snapshots/ebc52dc741e86466202a5ab8ab22eae6e7d48bf1/ram_swin_large_14m.pth
```

You can download the COCO training data from

```bash
hf download --resume-download Hzzone/GLIGEN_COCO coco_train2017.pth
```

It's in the format of

```json
[
  ...
  {
    'file_path': Path,
    'annos': [
      {
        'caption': Instance
        Caption,
        'bbox': bbox
        in
        xyxy,
        'text_embeddings_before_projection': CLIP
        text
        embedding
        before
        linear
        projection
      }
    ]
  }
  ...
]
```

### Training commands

The training script is heavily based
on https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py

```bash
accelerate launch train_gligen_text.py \
    --data_path /root/data/zhizhonghuang/coco_train2017.pth \
    --image_path /mnt/workspace/workgroup/zhizhonghuang/dataset/COCO/train2017 \
    --train_batch_size 8 \
    --max_train_steps 100000 \
    --checkpointing_steps 1000 \
    --checkpoints_total_limit 10 \
    --learning_rate 5e-5 \
    --dataloader_num_workers 16 \
    --mixed_precision fp16 \
    --report_to wandb \
    --tracker_project_name gligen \
    --output_dir /root/data/zhizhonghuang/ckpt/GLIGEN_Text_Retrain_COCO
```

I trained the model on 8 A100 GPUs for about 11 hours (at least 24GB GPU memory). The generated images will follow the
layout possibly at 50k iterations.

Note that although the pre-trained GLIGEN model has been loaded, the parameters of `fuser` and `position_net` have been reset (see line 420 in `train_gligen_text.py`)

The trained model can be downloaded from

```bash
hf download --resume-download Hzzone/GLIGEN_COCO config.json diffusion_pytorch_model.safetensors
```

You can run `demo.ipynb` to visualize the generated images.

Example prompts:

```python
prompt = 'A realistic image of landscape scene depicting a green car parking on the left of a blue truck, with a red air balloon and a bird in the sky'
boxes = [[0.041015625, 0.548828125, 0.453125, 0.859375],
         [0.525390625, 0.552734375, 0.93359375, 0.865234375],
         [0.12890625, 0.015625, 0.412109375, 0.279296875],
         [0.578125, 0.08203125, 0.857421875, 0.27734375]]
gligen_phrases = ['a green car', 'a blue truck', 'a red air balloon', 'a bird']
```

Example images:
![alt text](generated-images-100000-00.png)

### Citation

```
@article{li2023gligen,
  title={GLIGEN: Open-Set Grounded Text-to-Image Generation},
  author={Li, Yuheng and Liu, Haotian and Wu, Qingyang and Mu, Fangzhou and Yang, Jianwei and Gao, Jianfeng and Li, Chunyuan and Lee, Yong Jae},
  journal={CVPR},
  year={2023}
}
```