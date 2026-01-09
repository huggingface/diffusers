<!--版权所有 2025 HuggingFace 团队。保留所有权利。

根据 Apache 许可证 2.0 版本（"许可证"）授权；除非遵守许可证，否则您不得使用此文件。您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件按"原样"分发，不附带任何明示或暗示的担保或条件。请参阅许可证以了解具体的语言管理权限和限制。
-->

# Kandinsky 2.2

> [!WARNING]
> 此脚本是实验性的，容易过拟合并遇到灾难性遗忘等问题。尝试探索不同的超参数以在您的数据集上获得最佳结果。

Kandinsky 2.2 是一个多语言文本到图像模型，能够生成更逼真的图像。该模型包括一个图像先验模型，用于从文本提示创建图像嵌入，以及一个解码器模型，基于先验模型的嵌入生成图像。这就是为什么在 Diffusers 中您会找到两个独立的脚本用于 Kandinsky 2.2，一个用于训练先验模型，另一个用于训练解码器模型。您可以分别训练这两个模型，但为了获得最佳结果，您应该同时训练先验和解码器模型。

根据您的 GPU，您可能需要启用 `gradient_checkpointing`（⚠️ 不支持先验模型！）、`mixed_precision` 和 `gradient_accumulation_steps` 来帮助将模型装入内存并加速训练。您可以通过启用 [xFormers](../optimization/xformers) 的内存高效注意力来进一步减少内存使用（版本 [v0.0.16](https://github.com/huggingface/diffusers/issues/2234#issuecomment-1416931212) 在某些 GPU 上训练时失败，因此您可能需要安装开发版本）。

本指南探讨了 [train_text_to_image_prior.py](https://github.com/huggingface/diffusers/blob/main/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py) 和 [train_text_to_image_decoder.py](https://github.com/huggingface/diffusers/blob/main/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py) 脚本，以帮助您更熟悉它，以及如何根据您的用例进行调整。

在运行脚本之前，请确保从源代码安装库：

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

然后导航到包含训练脚本的示例文件夹，并安装脚本所需的依赖项：

```bash
cd examples/kandinsky2_2/text_to_image
pip install -r requirements.txt
```

> [!TIP]
> 🤗 Accelerate 是一个帮助您在多个 GPU/TPU 上或使用混合精度进行训练的库。它会根据您的硬件和环境自动配置训练设置。查看 🤗 Accelerate 的 [快速入门](https://huggingface.co/docs/accelerate/quicktour
> ) 了解更多。

初始化一个 🤗 Accelerate 环境：

```bash
accelerate config
```

要设置一个默认的 🤗 Accelerate 环境而不选择任何配置：

```bash
accelerate config default
```

或者，如果您的环境不支持交互式 shell，比如 notebook，您可以使用：

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

最后，如果您想在自己的数据集上训练模型，请查看 [创建用于训练的数据集](create_dataset) 指南，了解如何创建与训练脚本兼容的数据集。

> [!TIP]
> 以下部分重点介绍了训练脚本中对于理解如何修改它很重要的部分，但并未详细涵盖脚本的每个方面。如果您有兴趣了解更多，请随时阅读脚本，并让我们知道您有任何疑问或顾虑。

## 脚本参数

训练脚本提供了许多参数来帮助您自定义训练运行。所有参数及其描述都可以在 [`parse_args()`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py#L190) 函数中找到。训练脚本为每个参数提供了默认值，例如训练批次大小和学习率，但如果您愿意，也可以在训练命令中设置自己的值。

例如，要使用 fp16 格式的混合精度加速训练，请在训练命令中添加 `--mixed_precision` 参数：

```bash
accelerate launch train_text_to_image_prior.py \
  --mixed_precision="fp16"
```

大多数参数与 [文本到图像](text2image#script-parameters) 训练指南中的参数相同，所以让我们直接进入 Kandinsky 训练脚本的 walkthrough！

### Min-SNR 加权

[Min-SNR](https://huggingface.co/papers/2303.09556) 加权策略可以通过重新平衡损失来帮助训练，实现更快的收敛。训练脚本支持预测 `epsilon`（噪声）或 `v_prediction`，但 Min-SNR 与两种预测类型都兼容。此加权策略仅由 PyTorch 支持，在 Flax 训练脚本中不可用。

添加 `--snr_gamma` 参数并将其设置为推荐值 5.0：

```bash
accelerate launch train_text_to_image_prior.py \
  --snr_gamma=5.0
```

## 训练脚本

训练脚本也类似于 [文本到图像](text2image#training-script) 训练指南，但已修改以支持训练 prior 和 decoder 模型。本指南重点介绍 Kandinsky 2.2 训练脚本中独特的代码。

<hfoptions id="script">
<hfoption id="prior model">

[`main()`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py#L441) 函数包含代码 f
或准备数据集和训练模型。

您会立即注意到的主要区别之一是，训练脚本除了调度器和分词器外，还加载了一个 [`~transformers.CLIPImageProcessor`] 用于预处理图像，以及一个 [`~transformers.CLIPVisionModelWithProjection`] 模型用于编码图像：

```py
noise_scheduler = DDPMScheduler(beta_schedule="squaredcos_cap_v2", prediction_type="sample")
image_processor = CLIPImageProcessor.from_pretrained(
    args.pretrained_prior_model_name_or_path, subfolder="image_processor"
)
tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_prior_model_name_or_path, subfolder="tokenizer")

with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_prior_model_name_or_path, subfolder="image_encoder", torch_dtype=weight_dtype
    ).eval()
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_prior_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
    ).eval()
```

Kandinsky 使用一个 [`PriorTransformer`] 来生成图像嵌入，因此您需要设置优化器来学习先验模型的参数。

```py
prior = PriorTransformer.from_pretrained(args.pretrained_prior_model_name_or_path, subfolder="prior")
prior.train()
optimizer = optimizer_cls(
    prior.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```

接下来，输入标题被分词，图像由 [`~transformers.CLIPImageProcessor`] [预处理](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py#L632)：

```py
def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["clip_pixel_values"] = image_processor(images, return_tensors="pt").pixel_values
    examples["text_input_ids"], examples["text_mask"] = tokenize_captions(examples)
    return examples
```

最后，[训练循环](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py#L718) 将输入图像转换为潜在表示，向图像嵌入添加噪声，并进行预测：

```py
model_pred = prior(
    noisy_latents,
    timestep=timesteps,
    proj_embedding=prompt_embeds,
    encoder_hidden_states=text_encoder_hidden_states,
    attention_mask=text_mask,
).predicted_image_embedding
```

如果您想了解更多关于训练循环的工作原理，请查看 [理解管道、模型和调度器](../using-diffusers/write_own_pipeline) 教程，该教程分解了去噪过程的基本模式。

</hfoption>
<hfoption id="decoder model">

The [`main()`](https://github.com/huggingface/di
ffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py#L440) 函数包含准备数据集和训练模型的代码。

与之前的模型不同，解码器初始化一个 [`VQModel`] 来将潜在变量解码为图像，并使用一个 [`UNet2DConditionModel`]：

```py
with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    vae = VQModel.from_pretrained(
        args.pretrained_decoder_model_name_or_path, subfolder="movq", torch_dtype=weight_dtype
    ).eval()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_prior_model_name_or_path, subfolder="image_encoder", torch_dtype=weight_dtype
    ).eval()
unet = UNet2DConditionModel.from_pretrained(args.pretrained_decoder_model_name_or_path, subfolder="unet")
```

接下来，脚本包括几个图像变换和一个用于对图像应用变换并返回像素值的[预处理](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py#L622)函数：

```py
def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["clip_pixel_values"] = image_processor(images, return_tensors="pt").pixel_values
    return examples
```

最后，[训练循环](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py#L706)处理将图像转换为潜在变量、添加噪声和预测噪声残差。

如果您想了解更多关于训练循环如何工作的信息，请查看[理解管道、模型和调度器](../using-diffusers/write_own_pipeline)教程，该教程分解了去噪过程的基本模式。

```py
model_pred = unet(noisy_latents, timesteps, None, added_cond_kwargs=added_cond_kwargs).sample[:, :4]
```

</hfoption>
</hfoptions>

## 启动脚本

一旦您完成了所有更改或接受默认配置，就可以启动训练脚本了！🚀

您将在[Naruto BLIP 字幕](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions)数据集上进行训练，以生成您自己的Naruto角色，但您也可以通过遵循[创建用于训练的数据集](create_dataset)指南来创建和训练您自己的数据集。将环境变量 `DATASET_NAME` 设置为Hub上数据集的名称，或者如果您在自己的文件上训练，将环境变量 `TRAIN_DIR` 设置为数据集的路径。

如果您在多个GPU上训练，请在 `accelerate launch` 命令中添加 `--multi_gpu` 参数。

> [!TIP]
> 要使用Weights & Biases监控训练进度，请在训练命令中添加 `--report_to=wandb` 参数。您还需要
> 建议在训练命令中添加 `--validation_prompt` 以跟踪结果。这对于调试模型和查看中间结果非常有用。

<hfoptions id="training-inference">
<hfoption id="prior model">

```bash
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_prior.py \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_prompts="A robot naruto, 4k photo" \
  --report_to="wandb" \
  --push_to_hub \
  --output_dir="kandi2-prior-naruto-model"
```

</hfoption>
<hfoption id="decoder model">

```bash
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_decoder.py \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_prompts="A robot naruto, 4k photo" \
  --report_to="wandb" \
  --push_to_hub \
  --output_dir="kandi2-decoder-naruto-model"
```

</hfoption>
</hfoptions>

训练完成后，您可以使用新训练的模型进行推理！

<hfoptions id="training-inference">
<hfoption id="prior model">

```py
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
import torch

prior_pipeline = DiffusionPipeline.from_pretrained(output_dir, torch_dtype=torch.float16)
prior_components = {"prior_" + k: v for k,v in prior_pipeline.components.items()}
pipeline = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", **prior_components, torch_dtype=torch.float16)

pipe.enable_model_cpu_offload()
prompt="A robot naruto, 4k photo"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt).images[0]
```

> [!TIP]
> 可以随意将 `kandinsky-community/kandinsky-2-2-decoder` 替换为您自己训练的 decoder 检查点！

</hfoption>
<hfoption id="decoder model">

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("path/to/saved/model", torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

prompt="A robot naruto, 4k photo"
image = pipeline(prompt=prompt).images[0]
```

对于 decoder 模型，您还可以从保存的检查点进行推理，这对于查看中间结果很有用。在这种情况下，将检查点加载到 UNet 中：

```py
from diffusers import AutoPipelineForText2Image, UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("path/to/saved/model" + "/checkpoint-<N>/unet")

pipeline = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", unet=unet, torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

image = pipeline(prompt="A robot naruto, 4k photo").images[0]
```

</hfoption>
</hfoptions>

## 后续步骤

恭喜您训练了一个 Kandinsky 2.2 模型！要了解更多关于如何使用您的新模型的信息，以下指南可能会有所帮助：

- 阅读 [Kandinsky](../using-diffusers/kandinsky) 指南，学习如何将其用于各种不同的任务（文本到图像、图像到图像、修复、插值），以及如何与 ControlNet 结合使用。
- 查看 [DreamBooth](dreambooth) 和 [LoRA](lora) 训练指南，学习如何使用少量示例图像训练个性化的 Kandinsky 模型。这两种训练技术甚至可以结合使用！