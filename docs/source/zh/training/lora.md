<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LoRA 低秩适配

> [!WARNING]
> 当前功能处于实验阶段，API可能在未来版本中变更。

[LoRA（大语言模型的低秩适配）](https://hf.co/papers/2106.09685) 是一种轻量级训练技术，能显著减少可训练参数量。其原理是通过向模型注入少量新权重参数，仅训练这些新增参数。这使得LoRA训练速度更快、内存效率更高，并生成更小的模型权重文件（通常仅数百MB），便于存储和分享。LoRA还可与DreamBooth等其他训练技术结合以加速训练过程。

> [!TIP]
> LoRA具有高度通用性，目前已支持以下应用场景：[DreamBooth](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py)、[Kandinsky 2.2](https://github.com/huggingface/diffusers/blob/main/examples/kandinsky2_2/text_to_image/train_text_to_image_lora_decoder.py)、[Stable Diffusion XL](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py)、[文生图](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)以及[Wuerstchen](https://github.com/huggingface/diffusers/blob/main/examples/wuerstchen/text_to_image/train_text_to_image_lora_prior.py)。

本指南将通过解析[train_text_to_image_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)脚本，帮助您深入理解其工作原理，并掌握如何针对具体需求进行定制化修改。

运行脚本前，请确保从源码安装库：

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

进入包含训练脚本的示例目录，并安装所需依赖：

<hfoptions id="installation">
<hfoption id="PyTorch">

```bash
cd examples/text_to_image
pip install -r requirements.txt
```

</hfoption>
</hfoptions>

> [!TIP]
> 🤗 Accelerate是一个支持多GPU/TPU训练和混合精度计算的库，它能根据硬件环境自动配置训练方案。参阅🤗 Accelerate[快速入门](https://huggingface.co/docs/accelerate/quicktour)了解更多。

初始化🤗 Accelerate环境：

```bash
accelerate config
```

若要创建默认配置环境（不进行交互式设置）：

```bash
accelerate config default
```

若在非交互环境（如Jupyter notebook）中使用：

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

如需训练自定义数据集，请参考[创建训练数据集指南](create_dataset)了解数据准备流程。

> [!TIP]
> 以下章节重点解析训练脚本中与LoRA相关的核心部分，但不会涵盖所有实现细节。如需完整理解，建议直接阅读[脚本源码](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)，如有疑问欢迎反馈。

## 脚本参数

训练脚本提供众多参数用于定制训练过程。所有参数及其说明均定义在[`parse_args()`](https://github.com/huggingface/diffusers/blob/dd9a5caf61f04d11c0fa9f3947b69ab0010c9a0f/examples/text_to_image/train_text_to_image_lora.py#L85)函数中。多数参数设有默认值，您也可以通过命令行参数覆盖：

例如增加训练轮次：

```bash
accelerate launch train_text_to_image_lora.py \
  --num_train_epochs=150 \
```

基础参数说明可参考[文生图训练指南](text2image#script-parameters)，此处重点介绍LoRA相关参数：

- `--rank`：低秩矩阵的内部维度，数值越高可训练参数越多
- `--learning_rate`：默认学习率为1e-4，但使用LoRA时可适当提高

## 训练脚本实现

数据集预处理和训练循环逻辑位于[`main()`](https://github.com/huggingface/diffusers/blob/dd9a5caf61f04d11c0fa9f3947b69ab0010c9a0f/examples/text_to_image/train_text_to_image_lora.py#L371)函数，如需定制训练流程，可在此处进行修改。

与参数说明类似，训练流程的完整解析请参考[文生图指南](text2image#training-script)，下文重点介绍LoRA相关实现。

<hfoptions id="lora">
<hfoption id="UNet">

Diffusers使用[PEFT](https://hf.co/docs/peft)库的[`~peft.LoraConfig`]配置LoRA适配器参数，包括秩(rank)、alpha值以及目标模块。适配器被注入UNet后，通过`lora_layers`筛选出需要优化的LoRA层。

```py
unet_lora_config = LoraConfig(
    r=args.rank,
    lora_alpha=args.rank,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)

unet.add_adapter(unet_lora_config)
lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
```

</hfoption>
<hfoption id="text encoder">

当需要微调文本编码器时（如SDXL模型），Diffusers同样支持通过[PEFT](https://hf.co/docs/peft)库实现。[`~peft.LoraConfig`]配置适配器参数后注入文本编码器，并筛选LoRA层进行训练。

```py
text_lora_config = LoraConfig(
    r=args.rank,
    lora_alpha=args.rank,
    init_lora_weights="gaussian",
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
)

text_encoder_one.add_adapter(text_lora_config)
text_encoder_two.add_adapter(text_lora_config)
text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
text_lora_parameters_two = list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))
```

</hfoption>
</hfoptions>

[优化器](https://github.com/huggingface/diffusers/blob/e4b8f173b97731686e290b2eb98e7f5df2b1b322/examples/text_to_image/train_text_to_image_lora.py#L529)仅对`lora_layers`参数进行优化：

```py
optimizer = optimizer_cls(
    lora_layers,
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```

除LoRA层设置外，该训练脚本与标准train_text_to_image.py基本相同！

## 启动训练

完成所有配置后，即可启动训练脚本！🚀

以下示例使用[Naruto BLIP captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions)训练生成火影角色。请设置环境变量`MODEL_NAME`和`DATASET_NAME`指定基础模型和数据集，`OUTPUT_DIR`设置输出目录，`HUB_MODEL_ID`指定Hub存储库名称。脚本运行后将生成以下文件：

- 模型检查点
- `pytorch_lora_weights.safetensors`（训练好的LoRA权重）

多GPU训练请添加`--multi_gpu`参数。

> [!WARNING]
> 在11GB显存的2080 Ti显卡上完整训练约需5小时。

```bash
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="/sddata/finetune/lora/naruto"
export HUB_MODEL_ID="naruto-lora"
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="蓝色眼睛的火影忍者角色" \
  --seed=1337
```

训练完成后，您可以通过以下方式进行推理：

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("path/to/lora/model", weight_name="pytorch_lora_weights.safetensors")
image = pipeline("A naruto with blue eyes").images[0]
```

## 后续步骤

恭喜完成LoRA模型训练！如需进一步了解模型使用方法，可参考以下指南：

- 学习如何加载[不同格式的LoRA权重](../using-diffusers/loading_adapters#LoRA)（如Kohya或TheLastBen训练的模型）
- 掌握使用PEFT进行[多LoRA组合推理](../tutorials/using_peft_for_inference)的技巧