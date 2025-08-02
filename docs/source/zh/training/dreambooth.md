<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
# DreamBooth 训练指南

[DreamBooth](https://huggingface.co/papers/2208.12242) 是一种通过少量主题或风格样本图像即可更新整个扩散模型的训练技术。其核心原理是将提示词中的特殊标记与示例图像关联起来。

## 硬件要求与优化

若使用显存有限的GPU进行训练，建议在训练命令中启用 `gradient_checkpointing`（梯度检查点）和 `mixed_precision`（混合精度）参数。还可通过 [xFormers](../optimization/xformers) 启用内存高效注意力机制来降低显存占用。虽然JAX/Flax训练支持TPU和GPU高效训练，但不支持梯度检查点和xFormers。若需使用Flax加速训练，建议配备显存>30GB的GPU。

## 环境配置

本指南基于 [train_dreambooth.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py) 脚本解析。运行前请先安装库：

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

进入示例目录并安装依赖：

<hfoptions id="installation">
<hfoption id="PyTorch">

```bash
cd examples/dreambooth
pip install -r requirements.txt
```

</hfoption>
<hfoption id="Flax">

```bash
cd examples/dreambooth
pip install -r requirements_flax.txt
```

</hfoption>
</hfoptions>

<Tip>

🤗 Accelerate 是支持多GPU/TPU和混合精度训练的库，能根据硬件自动配置训练环境。参阅 [快速入门](https://huggingface.co/docs/accelerate/quicktour) 了解更多。

</Tip>

初始化加速环境：

```bash
accelerate config
```

或创建默认配置：

```bash
accelerate config default
```

非交互式环境（如notebook）可使用：

```py
from accelerate.utils import write_basic_config
write_basic_config()
```

自定义数据集训练请参阅 [创建训练数据集](create_dataset) 指南。

## 核心参数解析

<Tip warning={true}>

DreamBooth对超参数极其敏感，容易过拟合。建议阅读 [训练博客](https://huggingface.co/blog/dreambooth) 获取不同主题的推荐参数。

</Tip>

主要参数说明（完整参数见 [`parse_args()`](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L228)）：

- `--pretrained_model_name_or_path`：Hub模型ID或本地路径
- `--instance_data_dir`：训练图像目录
- `--instance_prompt`：包含特殊标记的提示词
- `--train_text_encoder`：是否同时训练文本编码器
- `--output_dir`：模型保存路径
- `--push_to_hub`：是否推送至Hub
- `--checkpointing_steps`：检查点保存频率，配合`--resume_from_checkpoint`可恢复训练

### Min-SNR加权策略

[Min-SNR](https://huggingface.co/papers/2303.09556) 策略通过损失重平衡加速收敛，支持`epsilon`和`v_prediction`预测类型（仅PyTorch可用）：

```bash
accelerate launch train_dreambooth.py --snr_gamma=5.0
```

### 先验保留损失

通过模型自生成样本增强多样性：

```bash
accelerate launch train_dreambooth.py \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --class_data_dir="path/to/class/images" \
  --class_prompt="类别描述文本"
```

### 文本编码器训练

需24GB+显存，可显著提升生成质量（特别面部生成）：

```bash
accelerate launch train_dreambooth.py --train_text_encoder
```

## 训练流程解析

核心组件：
- [`DreamBoothDataset`](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L604)：预处理图像与提示词
- [`PromptDataset`](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L738)：生成类别图像提示词

先验保留损失的类别图像生成逻辑：

```py
# 生成示例代码片段
```

模型加载流程：

```py
# 加载tokenizer、调度器和模型的代码示例
```

训练数据集构建：

```py
# 数据集创建代码示例
```

去噪训练循环详见 [理解pipeline](../using-diffusers/write_own_pipeline) 教程。

## 启动训练

示例使用 [狗狗数据集](https://huggingface.co/datasets/diffusers/dog-example)：

```py
from huggingface_hub import snapshot_download
local_dir = "./dog"
snapshot_download("diffusers/dog-example", local_dir=local_dir, repo_type="dataset", ignore_patterns=".gitattributes")
```

训练过程可视化参数：

```bash
--validation_prompt="a photo of a sks dog" --num_validation_images=4 --validation_steps=100
```

<hfoptions id="gpu-select">
<hfoption id="16GB">

16GB显卡优化方案：

```bash
pip install bitsandbytes
accelerate launch train_dreambooth.py --gradient_checkpointing --use_8bit_adam
```

</hfoption>
<hfoption id="12GB">

12GB显卡优化方案：

```bash
accelerate launch train_dreambooth.py \
  --use_8bit_adam \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none
```

</hfoption>
<hfoption id="8GB">

8GB显卡需使用DeepSpeed：

```bash
accelerate config  # 配置阶段选择DeepSpeed
```

需配合`deepspeed.ops.adam.DeepSpeedCPUAdam`优化器。

</hfoption>
</hfoptions>

<hfoptions id="training-inference">
<hfoption id="PyTorch">

```bash
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="path_to_saved_model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --push_to_hub
```

</hfoption>
<hfoption id="Flax">

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --max_train_steps=400 \
  --push_to_hub
```

</hfoption>
</hfoptions>

## 推理应用

<hfoptions id="training-inference">
<hfoption id="PyTorch">

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("path_to_saved_model", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
image = pipeline("A photo of sks dog in a bucket", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("dog-bucket.png")
```

</hfoption>
<hfoption id="Flax">

```py
# Flax推理代码示例
```

</hfoption>
</hfoptions>

## 进阶训练方案

### LoRA训练

参数高效训练技术，详见 [LoRA指南](lora)。

### SDXL训练

支持高分辨率生成的增强模型，详见 [SDXL指南](sdxl)。

### DeepFloyd IF训练

三阶段级联模型，关键参数：
- `--resolution=64`（基础模型）
- `--pre_compute_text_embeddings`（预计算文本嵌入）
- `--tokenizer_max_length=77`（T5编码器设置）

<hfoptions id="IF-DreamBooth">
<hfoption id="Stage 1 LoRA DreamBooth">

```bash
# 第一阶段LoRA训练命令
```

</hfoption>
<hfoption id="Stage 2 LoRA DreamBooth">

```bash
# 第二阶段LoRA训练命令
```

</hfoption>
<hfoption id="Stage 1 DreamBooth">

```bash
# 第一阶段全模型训练命令
```

</hfoption>
<hfoption id="Stage 2 DreamBooth">

```bash
# 第二阶段全模型训练命令
```

</hfoption>
</hfoptions>

### 训练建议
1. 常见物体可不微调上采样器
2. 面部等细节建议全参数训练
3. 使用较低学习率
4. 推荐`DDPMScheduler`

## 后续步骤

- 学习如何加载 [DreamBooth模型](../using-diffusers/loading_adapters) 进行推理