<!--版权声明 2025 由 HuggingFace 团队所有。保留所有权利。

根据 Apache 许可证 2.0 版（"许可证"）授权；除非符合许可证要求，否则不得使用本文件。
您可以通过以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，本软件按"原样"分发，不附带任何明示或暗示的担保或条件。详见许可证中规定的特定语言权限和限制。
-->

# 文本反转（Textual Inversion）

[文本反转](https://hf.co/papers/2208.01618)是一种训练技术，仅需少量示例图像即可个性化图像生成模型。该技术通过学习和更新文本嵌入（新嵌入会绑定到提示中必须使用的特殊词汇）来匹配您提供的示例图像。

如果在显存有限的GPU上训练，建议在训练命令中启用`gradient_checkpointing`和`mixed_precision`参数。您还可以通过[xFormers](../optimization/xformers)使用内存高效注意力机制来减少内存占用。JAX/Flax训练也支持在TPU和GPU上进行高效训练，但不支持梯度检查点或xFormers。在配置与PyTorch相同的情况下，Flax训练脚本的速度至少应快70%！

本指南将探索[textual_inversion.py](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py)脚本，帮助您更熟悉其工作原理，并了解如何根据自身需求进行调整。

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
cd examples/textual_inversion
pip install -r requirements.txt
```

</hfoption>
<hfoption id="Flax">

```bash
cd examples/textual_inversion
pip install -r requirements_flax.txt
```

</hfoption>
</hfoptions>

> [!TIP]
> 🤗 Accelerate 是一个帮助您在多GPU/TPU或混合精度环境下训练的工具库。它会根据硬件和环境自动配置训练设置。查看🤗 Accelerate [快速入门](https://huggingface.co/docs/accelerate/quicktour)了解更多。

初始化🤗 Accelerate环境：

```bash
accelerate config
```

要设置默认的🤗 Accelerate环境（不选择任何配置）：

```bash
accelerate config default
```

如果您的环境不支持交互式shell（如notebook），可以使用：

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

最后，如果想在自定义数据集上训练模型，请参阅[创建训练数据集](create_dataset)指南，了解如何创建适用于训练脚本的数据集。

> [!TIP]
> 以下部分重点介绍训练脚本中需要理解的关键修改点，但未涵盖脚本所有细节。如需深入了解，可随时查阅[脚本源码](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py)，如有疑问欢迎反馈。

## 脚本参数

训练脚本包含众多参数，便于您定制训练过程。所有参数及其说明都列在[`parse_args()`](https://github.com/huggingface/diffusers/blob/839c2a5ece0af4e75530cb520d77bc7ed8acf474/examples/textual_inversion/textual_inversion.py#L176)函数中。Diffusers为每个参数提供了默认值（如训练批次大小和学习率），但您可以通过训练命令自由调整这些值。

例如，将梯度累积步数增加到默认值1以上：

```bash
accelerate launch textual_inversion.py \
  --gradient_accumulation_steps=4
```

其他需要指定的基础重要参数包括：

- `--pretrained_model_name_or_path`：Hub上的模型名称或本地预训练模型路径
- `--train_data_dir`：包含训练数据集（示例图像）的文件夹路径
- `--output_dir`：训练模型保存位置
- `--push_to_hub`：是否将训练好的模型推送至Hub
- `--checkpointing_steps`：训练过程中保存检查点的频率；若训练意外中断，可通过在命令中添加`--resume_from_checkpoint`从该检查点恢复训练
- `--num_vectors`：学习嵌入的向量数量；增加此参数可提升模型效果，但会提高训练成本
- `--placeholder_token`：绑定学习嵌入的特殊词汇（推理时需在提示中使用该词）
- `--initializer_token`：大致描述训练目标的单字词汇（如物体或风格）
- `--learnable_property`：训练目标是学习新"风格"（如梵高画风）还是"物体"（如您的宠物狗）

## 训练脚本

与其他训练脚本不同，textual_inversion.py包含自定义数据集类[`TextualInversionDataset`](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L487)，用于创建数据集。您可以自定义图像尺寸、占位符词汇、插值方法、是否裁剪图像等。如需修改数据集创建方式，可调整`TextualInversionDataset`类。

接下来，在[`main()`](https://github.com/huggingface/diffusers/blob/839c2a5ece0af4e75530cb520d77bc7ed8acf474/examples/textual_inversion/textual_inversion.py#L573)函数中可找到数据集预处理代码和训练循环。

脚本首先加载[tokenizer](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L616)、[scheduler和模型](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L622)：

```py
# 加载tokenizer
if args.tokenizer_name:
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
elif args.pretrained_model_name_or_path:
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

# 加载scheduler和模型
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
)
vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
)
```

随后将特殊[占位符词汇](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L632)加入tokenizer，并调整嵌入层以适配新词汇。

接着，脚本通过`TextualInversionDataset`[创建数据集](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L716)：

```py
train_dataset = TextualInversionDataset(
    data_root=args.train_data_dir,
    tokenizer=tokenizer,
    size=args.resolution,
    placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))),
    repeats=args.repeats,
    learnable_property=args.learnable_property,
    center_crop=args.center_crop,
    set="train",
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
)
```

最后，[训练循环](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L784)处理从预测噪声残差到更新特殊占位符词汇嵌入权重的所有流程。

如需深入了解训练循环工作原理，请参阅[理解管道、模型与调度器](../using-diffusers/write_own_pipeline)教程，该教程解析了去噪过程的基本模式。

## 启动脚本

完成所有修改或确认默认配置后，即可启动训练脚本！🚀

本指南将下载[猫玩具](https://huggingface.co/datasets/diffusers/cat_toy_example)的示例图像并存储在目录中。当然，您也可以创建和使用自己的数据集（参见[创建训练数据集](create_dataset)指南）。

```py
from huggingface_hub import snapshot_download

local_dir = "./cat"
snapshot_download(
    "diffusers/cat_toy_example", local_dir=local_dir, repo_type="dataset", ignore_patterns=".gitattributes"
)
```

设置环境变量`MODEL_NAME`为Hub上的模型ID或本地模型路径，`DATA_DIR`为刚下载的猫图像路径。脚本会将以下文件保存至您的仓库：

- `learned_embeds.bin`：与示例图像对应的学习嵌入向量
- `token_identifier.txt`：特殊占位符词汇
- `type_of_concept.txt`：训练概念类型（"object"或"style"）

> [!WARNING]
> 在单块V100 GPU上完整训练约需1小时。

启动脚本前还有最后一步。如果想实时观察训练过程，可以定期保存生成图像。在训练命令中添加以下参数：

```bash
--validation_prompt="A <cat-toy> train"
--num_validation_images=4
--validation_steps=100
```

<hfoptions id="training-inference">
<hfoption id="PyTorch">

```bash
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export DATA_DIR="./cat"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" \
  --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="textual_inversion_cat" \
  --push_to_hub
```

</hfoption>
<hfoption id="Flax">

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export DATA_DIR="./cat"

python textual_inversion_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" \
  --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --output_dir="textual_inversion_cat" \
  --push_to_hub
```

</hfoption>
</hfoptions>

训练完成后，可以像这样使用新模型进行推理：

<hfoptions id="training-inference">
<hfoption id="PyTorch">

```py
from diffusers import StableDiffusionPipeline
import torch

pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline.load_textual_inversion("sd-concepts-library/cat-toy")
image = pipeline("A <cat-toy> train", num_inference_steps=50).images[0]
image.save("cat-train.png")
```

</hfoption>
<hfoption id="Flax">

Flax不支持[`~loaders.TextualInversionLoaderMixin.load_textual_inversion`]方法，但textual_inversion_flax.py脚本会在训练后[保存](https://github.com/huggingface/diffusers/blob/c0f058265161178f2a88849e92b37ffdc81f1dcc/examples/textual_inversion/textual_inversion_flax.py#L636C2-L636C2)学习到的嵌入作为模型的一部分。这意味着您可以像使用其他Flax模型一样进行推理：

```py
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline

model_path = "path-to-your-trained-model"
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(model_path, dtype=jax.numpy.bfloat16)

prompt = "A <cat-toy> train"
prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

num_samples = jax.device_count()
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# 分片输入和随机数生成器
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
image.save("cat-train.png")
```

</hfoption>
</hfoptions>

## 后续步骤

恭喜您成功训练了自己的文本反转模型！🎉 如需了解更多使用技巧，以下指南可能会有所帮助：

- 学习如何[加载文本反转嵌入](../using-diffusers/loading_adapters)，并将其用作负面嵌入
- 学习如何将[文本反转](textual_inversion_inference)应用于Stable Diffusion 1/2和Stable Diffusion XL的推理
