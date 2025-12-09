<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ControlNet

[ControlNet](https://hf.co/papers/2302.05543) 是一种基于预训练模型的适配器架构。它通过额外输入的条件图像（如边缘检测图、深度图、人体姿态图等），实现对生成图像的精细化控制。

在显存有限的GPU上训练时，建议启用训练命令中的 `gradient_checkpointing`（梯度检查点）、`gradient_accumulation_steps`（梯度累积步数）和 `mixed_precision`（混合精度）参数。还可使用 [xFormers](../optimization/xformers) 的内存高效注意力机制进一步降低显存占用。虽然JAX/Flax训练支持在TPU和GPU上高效运行，但不支持梯度检查点和xFormers。若需通过Flax加速训练，建议使用显存大于30GB的GPU。

本指南将解析 [train_controlnet.py](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py) 训练脚本，帮助您理解其逻辑并适配自定义需求。

运行脚本前，请确保从源码安装库：

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

然后进入包含训练脚本的示例目录，安装所需依赖：

<hfoptions id="installation">
<hfoption id="PyTorch">
```bash
cd examples/controlnet
pip install -r requirements.txt
```
</hfoption>
<hfoption id="Flax">

若可访问TPU设备，Flax训练脚本将运行得更快！以下是在 [Google Cloud TPU VM](https://cloud.google.com/tpu/docs/run-calculation-jax) 上的配置流程。创建单个TPU v4-8虚拟机并连接：

```bash
ZONE=us-central2-b
TPU_TYPE=v4-8
VM_NAME=hg_flax

gcloud alpha compute tpus tpu-vm create $VM_NAME \
 --zone $ZONE \
 --accelerator-type $TPU_TYPE \
 --version  tpu-vm-v4-base

gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone $ZONE -- \
```

安装JAX 0.4.5：

```bash
pip install "jax[tpu]==0.4.5" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

然后安装Flax脚本的依赖：

```bash
cd examples/controlnet
pip install -r requirements_flax.txt
```

</hfoption>
</hfoptions>

> [!TIP]
> 🤗 Accelerate 是一个支持多GPU/TPU训练和混合精度的库，它能根据硬件环境自动配置训练方案。参阅 🤗 Accelerate [快速入门](https://huggingface.co/docs/accelerate/quicktour) 了解更多。

初始化🤗 Accelerate环境：

```bash
accelerate config
```

若要创建默认配置（不进行交互式选择）：

```bash
accelerate config default
```

若环境不支持交互式shell（如notebook），可使用：

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

最后，如需训练自定义数据集，请参阅 [创建训练数据集](create_dataset) 指南了解数据准备方法。

> [!TIP]
> 下文重点解析脚本中的关键模块，但不会覆盖所有实现细节。如需深入了解，建议直接阅读 [脚本源码](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py)，如有疑问欢迎反馈。

## 脚本参数

训练脚本提供了丰富的可配置参数，所有参数及其说明详见 [`parse_args()`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/controlnet/train_controlnet.py#L231) 函数。虽然该函数已为每个参数提供默认值（如训练批大小、学习率等），但您可以通过命令行参数覆盖这些默认值。

例如，使用fp16混合精度加速训练, 可使用`--mixed_precision`参数

```bash
accelerate launch train_controlnet.py \
  --mixed_precision="fp16"
```

基础参数说明可参考 [文生图](text2image#script-parameters) 训练指南，此处重点介绍ControlNet相关参数：

- `--max_train_samples`: 训练样本数量，减少该值可加快训练，但对超大数据集需配合 `--streaming` 参数使用
- `--gradient_accumulation_steps`: 梯度累积步数，通过分步计算实现显存受限情况下的更大批次训练

### Min-SNR加权策略

[Min-SNR](https://huggingface.co/papers/2303.09556) 加权策略通过重新平衡损失函数加速模型收敛。虽然训练脚本支持预测 `epsilon`（噪声）或 `v_prediction`，但Min-SNR对两种预测类型均兼容。该策略仅适用于PyTorch版本，Flax训练脚本暂不支持。

推荐值设为5.0：

```bash
accelerate launch train_controlnet.py \
  --snr_gamma=5.0
```

## 训练脚本

与参数说明类似，训练流程的通用解析可参考 [文生图](text2image#training-script) 指南。此处重点分析ControlNet特有的实现。

脚本中的 [`make_train_dataset`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/controlnet/train_controlnet.py#L582) 函数负责数据预处理，除常规的文本标注分词和图像变换外，还包含条件图像的特效处理：

> [!TIP]
> 在TPU上流式加载数据集时，🤗 Datasets库可能成为性能瓶颈（因其未针对图像数据优化）。建议考虑 [WebDataset](https://webdataset.github.io/webdataset/)、[TorchData](https://github.com/pytorch/data) 或 [TensorFlow Datasets](https://www.tensorflow.org/datasets/tfless_tfds) 等高效数据格式。

```py
conditioning_image_transforms = transforms.Compose(
    [
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
    ]
)
```

在 [`main()`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/controlnet/train_controlnet.py#L713) 函数中，代码会加载分词器、文本编码器、调度器和模型。此处也是ControlNet模型的加载点（支持从现有权重加载或从UNet随机初始化）：

```py
if args.controlnet_model_name_or_path:
    logger.info("Loading existing controlnet weights")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
else:
    logger.info("Initializing controlnet weights from unet")
    controlnet = ControlNetModel.from_unet(unet)
```

[优化器](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/controlnet/train_controlnet.py#L871) 专门针对ControlNet参数进行更新：

```py
params_to_optimize = controlnet.parameters()
optimizer = optimizer_class(
    params_to_optimize,
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```

在 [训练循环](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/controlnet/train_controlnet.py#L943) 中，条件文本嵌入和图像被输入到ControlNet的下采样和中层模块：

```py
encoder_hidden_states = text_encoder(batch["input_ids"])[0]
controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

down_block_res_samples, mid_block_res_sample = controlnet(
    noisy_latents,
    timesteps,
    encoder_hidden_states=encoder_hidden_states,
    controlnet_cond=controlnet_image,
    return_dict=False,
)
```

若想深入理解训练循环机制，可参阅 [理解管道、模型与调度器](../using-diffusers/write_own_pipeline) 教程，该教程详细解析了去噪过程的基本原理。

## 启动训练

现在可以启动训练脚本了！🚀

本指南使用 [fusing/fill50k](https://huggingface.co/datasets/fusing/fill50k) 数据集，当然您也可以按照 [创建训练数据集](create_dataset) 指南准备自定义数据。

设置环境变量 `MODEL_NAME` 为Hub模型ID或本地路径，`OUTPUT_DIR` 为模型保存路径。

下载训练用的条件图像：

```bash
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

根据GPU型号，可能需要启用特定优化。默认配置需要约38GB显存。若使用多GPU训练，请在 `accelerate launch` 命令中添加 `--multi_gpu` 参数。

<hfoptions id="gpu-select">
<hfoption id="16GB">

16GB显卡可使用bitsandbytes 8-bit优化器和梯度检查点：

```py
pip install bitsandbytes
```

训练命令添加以下参数：

```bash
accelerate launch train_controlnet.py \
  --gradient_checkpointing \
  --use_8bit_adam \
```

</hfoption>
<hfoption id="12GB">

12GB显卡需组合使用bitsandbytes 8-bit优化器、梯度检查点、xFormers，并将梯度置为None而非0：

```bash
accelerate launch train_controlnet.py \
  --use_8bit_adam \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none \
```

</hfoption>
<hfoption id="8GB">

8GB显卡需使用 [DeepSpeed](https://www.deepspeed.ai/) 将张量卸载到CPU或NVME：

运行以下命令配置环境：

```bash
accelerate config
```

选择DeepSpeed stage 2，结合fp16混合精度和参数卸载到CPU的方案。注意这会增加约25GB内存占用。配置示例如下：

```bash
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 4
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
```

建议将优化器替换为DeepSpeed特化版 [`deepspeed.ops.adam.DeepSpeedCPUAdam`](https://deepspeed.readthedocs.io/en/latest/optimizers.html#adam-cpu)，注意CUDA工具链版本需与PyTorch匹配。

当前bitsandbytes与DeepSpeed存在兼容性问题。

无需额外添加训练参数。

</hfoption>
</hfoptions>

<hfoptions id="training-inference">
<hfoption id="PyTorch">

```bash
export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="path/to/save/model"

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --push_to_hub
```

</hfoption>
<hfoption id="Flax">

Flax版本支持通过 `--profile_steps==5` 参数进行性能分析：

```bash
pip install tensorflow tensorboard-plugin-profile
tensorboard --logdir runs/fill-circle-100steps-20230411_165612/
```

在 [http://localhost:6006/#profile](http://localhost:6006/#profile) 查看分析结果。

> [!WARNING]
> 若遇到插件版本冲突，建议重新安装TensorFlow和Tensorboard。注意性能分析插件仍处实验阶段，部分视图可能不完整。`trace_viewer` 会截断超过1M的事件记录，在编译步骤分析时可能导致设备轨迹丢失。

```bash
python3 train_controlnet_flax.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --validation_steps=1000 \
 --train_batch_size=2 \
 --revision="non-ema" \
 --from_pt \
 --report_to="wandb" \
 --tracker_project_name=$HUB_MODEL_ID \
 --num_train_epochs=11 \
 --push_to_hub \
 --hub_model_id=$HUB_MODEL_ID
```

</hfoption>
</hfoptions>

训练完成后即可进行推理：

```py
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch

controlnet = ControlNetModel.from_pretrained("path/to/controlnet", torch_dtype=torch.float16)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "path/to/base/model", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

control_image = load_image("./conditioning_image_1.png")
prompt = "pale golden rod circle with old lace background"

generator = torch.manual_seed(0)
image = pipeline(prompt, num_inference_steps=20, generator=generator, image=control_image).images[0]
image.save("./output.png")
```

## Stable Diffusion XL

Stable Diffusion XL (SDXL) 是新一代文生图模型，通过添加第二文本编码器支持生成更高分辨率图像。使用 [`train_controlnet_sdxl.py`](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet_sdxl.py) 脚本可为SDXL训练ControlNet适配器。

SDXL训练脚本的详细解析请参阅 [SDXL训练](sdxl) 指南。

## 后续步骤

恭喜完成ControlNet训练！如需进一步了解模型应用，以下指南可能有所帮助：

- 学习如何 [使用ControlNet](../using-diffusers/controlnet) 进行多样化任务的推理
