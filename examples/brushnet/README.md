# BrushNet

This is the implementation of the ECCV2024 paper "BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion". The original repo is at [here](https://github.com/TencentARC/BrushNet).

Keywords: Image Inpainting, Diffusion Models, Image Generation

> [Xuan Ju](https://github.com/juxuan27)<sup>12</sup>, [Xian Liu](https://alvinliu0.github.io/)<sup>12</sup>, [Xintao Wang](https://xinntao.github.io/)<sup>1*</sup>, [Yuxuan Bian](https://scholar.google.com.hk/citations?user=HzemVzoAAAAJ&hl=zh-CN&oi=ao)<sup>2</sup>, [Ying Shan](https://www.linkedin.com/in/YingShanProfile/)<sup>1</sup>, [Qiang Xu](https://cure-lab.github.io/)<sup>2*</sup><br>
> <sup>1</sup>ARC Lab, Tencent PCG <sup>2</sup>The Chinese University of Hong Kong <sup>*</sup>Corresponding Author


<p align="center">
  <a href="https://tencentarc.github.io/BrushNet/">🌐Project Page</a> |
  <a href="https://arxiv.org/abs/2403.06976">📜Arxiv</a> |
  <a href="https://forms.gle/9TgMZ8tm49UYsZ9s5">🗄️Data</a> |
  <a href="https://drive.google.com/file/d/1IkEBWcd2Fui2WHcckap4QFPcCI0gkHBh/view">📹Video</a> |
  <a href="https://huggingface.co/spaces/TencentARC/BrushNet">🤗Hugging Face Demo</a> |
</p>



**📖 Table of Contents**


- [BrushNet](#brushnet)
  - [🛠️ Method Overview](#️-method-overview)
  - [🚀 Getting Started](#-getting-started)
    - [Environment Requirement 🌍](#environment-requirement-)
    - [Data Download ⬇️](#data-download-️)
  - [🏃🏼 Running Scripts](#-running-scripts)
    - [Training 🤯](#training-)
    - [Inference 📜](#inference-)
    - [Evaluation 📏](#evaluation-)
  - [🤝🏼 Cite Us](#-cite-us)
  - [💖 Acknowledgement](#-acknowledgement)


## 🛠️ Method Overview

BrushNet is a diffusion-based text-guided image inpainting model that can be plug-and-play into any pre-trained diffusion model. Our architectural design incorporates two key insights: (1) dividing the masked image features and noisy latent reduces the model's learning load, and (2) leveraging dense per-pixel control over the entire pre-trained model enhances its suitability for image inpainting tasks. More analysis can be found in the main paper.

![](examples/brushnet/src/model.png)



## 🚀 Getting Started

### Environment Requirement 🌍

BrushNet has been implemented and tested on Pytorch 1.12.1 with python 3.9.

Clone the repo:

```
git clone https://github.com/TencentARC/BrushNet.git
```

We recommend you first use `conda` to create virtual environment, and install `pytorch` following [official instructions](https://pytorch.org/). For example:


```
conda create -n diffusers python=3.9 -y
conda activate diffusers
python -m pip install --upgrade pip
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Then, you can install diffusers (implemented in this repo) with:

```
pip install -e .
```

After that, you can install required packages thourgh:

```
cd examples/brushnet/
pip install -r requirements.txt
```

### Data Download ⬇️


**Dataset**

You can download the BrushData and BrushBench [here](https://forms.gle/9TgMZ8tm49UYsZ9s5) (as well as the EditBench we re-processed), which are used for training and testing the BrushNet. By downloading the data, you are agreeing to the terms and conditions of the license. The data structure should be like:

```
|-- data
    |-- BrushData
        |-- 00200.tar
        |-- 00201.tar
        |-- ...
    |-- BrushDench
        |-- images
        |-- mapping_file.json
    |-- EditBench
        |-- images
        |-- mapping_file.json
```


Noted: *We only provide a part of the BrushData in google drive due to the space limit. [random123123](https://huggingface.co/random123123) has helped upload a full dataset on hugging face [here](https://huggingface.co/datasets/random123123/BrushData). Thank for his help!*


**Checkpoints**

Checkpoints of BrushNet can be downloaded from [here](https://drive.google.com/drive/folders/1fqmS1CEOvXCxNWFrsSYd_jHYXxrydh1n?usp=drive_link). The ckpt folder contains 

- BrushNet pretrained checkpoints for Stable Diffusion v1.5 (`segmentation_mask_brushnet_ckpt` and `random_mask_brushnet_ckpt`)
- pretrinaed Stable Diffusion v1.5 checkpoint (e.g., realisticVisionV60B1_v51VAE from [Civitai](https://civitai.com/)). You can use `scripts/convert_original_stable_diffusion_to_diffusers.py` to process other models downloaded from Civitai. 
- BrushNet pretrained checkpoints for Stable Diffusion XL (`segmentation_mask_brushnet_ckpt_sdxl_v1` and `random_mask_brushnet_ckpt_sdxl_v0`).  A better version will be shortly released by [yuanhang](https://github.com/yuanhangio). Please stay tuned!
- pretrinaed Stable Diffusion XL checkpoint (e.g., juggernautXL_juggernautX from [Civitai](https://civitai.com/)). You can use `StableDiffusionXLPipeline.from_single_file("path of safetensors").save_pretrained("path to save",safe_serialization=False)` to process other models downloaded from Civitai. 



The data structure should be like:


```
|-- data
    |-- BrushData
    |-- BrushDench
    |-- EditBench
    |-- ckpt
        |-- realisticVisionV60B1_v51VAE
            |-- model_index.json
            |-- vae
            |-- ...
        |-- segmentation_mask_brushnet_ckpt
        |-- segmentation_mask_brushnet_ckpt_sdxl_v0
        |-- random_mask_brushnet_ckpt
        |-- random_mask_brushnet_ckpt_sdxl_v0
        |-- ...
```

The checkpoint in `segmentation_mask_brushnet_ckpt` and `segmentation_mask_brushnet_ckpt_sdxl_v0` provide checkpoints trained on BrushData, which has segmentation prior (mask are with the same shape of objects). The `random_mask_brushnet_ckpt` and `random_mask_brushnet_ckpt_sdxl` provide a more general ckpt for random mask shape.

## 🏃🏼 Running Scripts


### Training 🤯

You can train with segmentation mask using the script:

```
# sd v1.5
accelerate launch examples/brushnet/train_brushnet.py \
--pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
--output_dir runs/logs/brushnet_segmentationmask \
--train_data_dir data/BrushData \
--resolution 512 \
--learning_rate 1e-5 \
--train_batch_size 2 \
--tracker_project_name brushnet \
--report_to tensorboard \
--resume_from_checkpoint latest \
--validation_steps 300 \
--checkpointing_steps 10000 

# sdxl
accelerate launch examples/brushnet/train_brushnet_sdxl.py \
--pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
--output_dir runs/logs/brushnetsdxl_segmentationmask \
--train_data_dir data/BrushData \
--resolution 1024 \
--learning_rate 1e-5 \
--train_batch_size 1 \
--gradient_accumulation_steps 4 \
--tracker_project_name brushnet \
--report_to tensorboard \
--resume_from_checkpoint latest \
--validation_steps 300 \
--checkpointing_steps 10000 
```

To use custom dataset, you can process your own data to the format of BrushData and revise `--train_data_dir`.

You can train with random mask using the script (by adding `--random_mask`):

```
# sd v1.5
accelerate launch examples/brushnet/train_brushnet.py \
--pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
--output_dir runs/logs/brushnet_randommask \
--train_data_dir data/BrushData \
--resolution 512 \
--learning_rate 1e-5 \
--train_batch_size 2 \
--tracker_project_name brushnet \
--report_to tensorboard \
--resume_from_checkpoint latest \
--validation_steps 300 \
--random_mask

# sdxl
accelerate launch examples/brushnet/train_brushnet_sdxl.py \
--pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
--output_dir runs/logs/brushnetsdxl_randommask \
--train_data_dir data/BrushData \
--resolution 1024 \
--learning_rate 1e-5 \
--train_batch_size 1 \
--gradient_accumulation_steps 4 \
--tracker_project_name brushnet \
--report_to tensorboard \
--resume_from_checkpoint latest \
--validation_steps 300 \
--checkpointing_steps 10000 \
--random_mask
```



### Inference 📜

You can inference with the script:

```
# sd v1.5
python examples/brushnet/test_brushnet.py
# sdxl
python examples/brushnet/test_brushnet_sdxl.py
```

Since BrushNet is trained on Laion, it can only guarantee the performance on general scenarios. We recommend you train on your own data (e.g., product exhibition, virtual try-on) if you have high-quality industrial application requirements. We would also be appreciate if you would like to contribute your trained model!

You can also inference through gradio demo:

```
# sd v1.5
python examples/brushnet/app_brushnet.py
```


### Evaluation 📏

You can evaluate using the script:

```
python examples/brushnet/evaluate_brushnet.py \
--brushnet_ckpt_path data/ckpt/segmentation_mask_brushnet_ckpt \
--image_save_path runs/evaluation_result/BrushBench/brushnet_segmask/inside \
--mapping_file data/BrushBench/mapping_file.json \
--base_dir data/BrushBench \
--mask_key inpainting_mask
```

The `--mask_key` indicates which kind of mask to use, `inpainting_mask` for inside inpainting and `outpainting_mask` for outside inpainting. The evaluation results (images and metrics) will be saved in `--image_save_path`. 



*Noted that you need to ignore the nsfw detector in `src/diffusers/pipelines/brushnet/pipeline_brushnet.py#1261` to get the correct evaluation results. Moreover, we find different machine may generate different images, thus providing the results on our machine [here](https://drive.google.com/drive/folders/1dK3oIB2UvswlTtnIS1iHfx4s57MevWdZ?usp=sharing).*


## 🤝🏼 Cite Us

```
@misc{ju2024brushnet,
  title={BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion}, 
  author={Xuan Ju and Xian Liu and Xintao Wang and Yuxuan Bian and Ying Shan and Qiang Xu},
  year={2024},
  eprint={2403.06976},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```


## 💖 Acknowledgement
<span id="acknowledgement"></span>

Our code is modified based on [diffusers](https://github.com/huggingface/diffusers), thanks to all the contributors!

