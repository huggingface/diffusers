<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LoRA

[LoRA (Low-Rank Adaptation)](https://huggingface.co/papers/2106.09685) 是一种让模型快速适配新任务的方法。它会冻结原始模型权重，并额外添加一小部分*新的*可训练参数。这样一来，在现有模型上适配新任务的速度会更快、成本也更低，比如生成某种新的图像风格。

LoRA的checkpoint通常只有几百 MB，因此非常轻量，也很容易存储。你可以使用 [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 将这组较小的权重加载到现有基础模型中，并通过 `weight_name` 指定文件名。

<hfoptions id="usage">
<hfoption id="text-to-image">

```py
import torch
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/super-cereal-sdxl-lora",
    weight_name="cereal_box_sdxl_v1.safetensors",
    adapter_name="cereal"
)
pipeline("bears, pizza bites").images[0]
```

</hfoption>
<hfoption id="text-to-video">

```py
import torch
from diffusers import LTXConditionPipeline
from diffusers.utils import export_to_video, load_image

pipeline = LTXConditionPipeline.from_pretrained(
    "Lightricks/LTX-Video-0.9.5", torch_dtype=torch.bfloat16
)

pipeline.load_lora_weights(
    "Lightricks/LTX-Video-Cakeify-LoRA",
    weight_name="ltxv_095_cakeify_lora.safetensors",
    adapter_name="cakeify"
)
pipeline.set_adapters("cakeify")

# 使用 "CAKEIFY" 触发这个 LoRA
prompt = "CAKEIFY a person using a knife to cut a cake shaped like a Pikachu plushie"
image = load_image("https://huggingface.co/Lightricks/LTX-Video-Cakeify-LoRA/resolve/main/assets/images/pikachu.png")

video = pipeline(
    prompt=prompt,
    image=image,
    width=576,
    height=576,
    num_frames=161,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=26)
```

</hfoption>
</hfoptions>

[`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 是把 LoRA 权重加载到 UNet 和 text encoder 中的首选方式，因为它能处理以下情况：

- LoRA 权重没有分别标注 UNet 和text encoder标识符
- LoRA 权重分别带有 UNet 和text encoder标识符

[`~loaders.PeftAdapterMixin.load_lora_adapter`] 则用于在*模型级别*直接加载 LoRA adapter，只要该模型是 Diffusers 模型并且继承自 [`PeftAdapterMixin`] 即可。它会为 adapter 构建并准备所需的模型配置。这个方法同样会把 LoRA adapter 加载到 UNet 中。

例如，如果你只想把 LoRA 加载到 UNet，[`~loaders.PeftAdapterMixin.load_lora_adapter`] 会忽略文本编码器对应的 key。使用 `prefix` 参数筛选并加载合适的 state dict，这里传入 `"unet"` 即可。

```py
import torch
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.unet.load_lora_adapter(
    "jbilcke-hf/sdxl-cinematic-1",
    weight_name="pytorch_lora_weights.safetensors",
    adapter_name="cinematic",
    prefix="unet"
)
# 在提示词中使用 cnmt 来触发这个 LoRA
pipeline("A cute cnmt eating a slice of pizza, stunning color scheme, masterpiece, illustration").images[0]
```

## torch.compile

[torch.compile](../optimization/fp16#torchcompile) 会通过编译 PyTorch 模型来使用优化内核，从而加速推理。在编译之前，需要先把 LoRA 权重融合进基础模型，并卸载原始 LoRA 权重。

```py
import torch
from diffusers import DiffusionPipeline

# 加载基础模型和 LoRA
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/ikea-instructions-lora-sdxl",
    weight_name="ikea_instructions_xl_v1_5.safetensors",
    adapter_name="ikea"
)

# 激活 LoRA 并设置 adapter 权重
pipeline.set_adapters("ikea", adapter_weights=0.7)

# 融合 LoRA 并卸载权重
pipeline.fuse_lora(adapter_names=["ikea"], lora_scale=1.0)
pipeline.unload_lora_weights()
```

通常会编译 UNet，因为它是整个管道里计算最密集的部分。

```py
pipeline.unet.to(memory_format=torch.channels_last)
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

pipeline("A bowl of ramen shaped like a cute kawaii bear").images[0]
```

如果你想在编译模型后配合多个 LoRA 一起使用，又不想每次都重新编译，可以查看下文的 [hotswapping](#hotswapping) 部分。

## 权重缩放

`scale` 参数用于控制 LoRA 的应用强度。值为 `0` 时等价于只使用基础模型权重；值为 `1` 时等价于完全使用 LoRA。

<hfoptions id="weight-scale">
<hfoption id="simple use case">

对于简单场景，可以直接把 `cross_attention_kwargs={"scale": 1.0}` 传给管道。

```py
import torch
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/super-cereal-sdxl-lora",
    weight_name="cereal_box_sdxl_v1.safetensors",
    adapter_name="cereal"
)
pipeline("bears, pizza bites", cross_attention_kwargs={"scale": 1.0}).images[0]
```

</hfoption>
<hfoption id="finer control">

> [!WARNING]
> [`~loaders.PeftAdapterMixin.set_adapters`] 只会缩放 attention 权重。如果某个 LoRA 还包含 ResNet、downsampler 或 upsampler，这些组件的缩放值仍会保持为 `1.0`。

如果你想更细粒度地控制 UNet 或文本编码器中每个组件的缩放比例，可以改为传入一个字典。下面这个例子里，UNet 中 `"down"` block 的缩放值是 0.9，而 `"up"` block 里还进一步指定了 `"block_0"` 和 `"block_1"` 中 transformer 的缩放值。如果像 `"mid"` 这样的 block 没有显式指定，就会使用默认值 `1.0`。

```py
import torch
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/super-cereal-sdxl-lora",
    weight_name="cereal_box_sdxl_v1.safetensors",
    adapter_name="cereal"
)
scales = {
    "text_encoder": 0.5,
    "text_encoder_2": 0.5,
    "unet": {
        "down": 0.9,
        "up": {
            "block_0": 0.6,
            "block_1": [0.4, 0.8, 1.0],
        }
    }
}
pipeline.set_adapters("cereal", scales)
pipeline("bears, pizza bites").images[0]
```

</hfoption>
</hfoptions>

### 缩放调度

在采样过程中动态调整 LoRA scale，通常可以让你更好地控制整体构图和布局，因为某些采样步骤可能更适合使用更高或更低的 scale。

下面的例子使用了一个 [character LoRA](https://huggingface.co/alvarobartt/ghibli-characters-flux-lora)。它在前 20 步使用较高的 scale，并逐步衰减，以便先把角色生成出来；在后续步骤中，只保留 0.2 的 scale，避免把 LoRA 学到的特征过多地施加到图像中其他并非训练目标的区域。

```py
import torch
from diffusers import FluxPipeline

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")

pipelne.load_lora_weights("alvarobartt/ghibli-characters-flux-lora", "lora")

num_inference_steps = 30
lora_steps = 20
lora_scales = torch.linspace(1.5, 0.7, lora_steps).tolist()
lora_scales += [0.2] * (num_inference_steps - lora_steps + 1)

pipeline.set_adapters("lora", lora_scales[0])

def callback(pipeline: FluxPipeline, step: int, timestep: torch.LongTensor, callback_kwargs: dict):
    pipeline.set_adapters("lora", lora_scales[step + 1])
    return callback_kwargs

prompt = """
Ghibli style The Grinch, a mischievous green creature with a sly grin, peeking out from behind a snow-covered tree while plotting his antics, 
in a quaint snowy village decorated for the holidays, warm light glowing from cozy homes, with playful snowflakes dancing in the air
"""
pipeline(
    prompt=prompt,
    guidance_scale=3.0,
    num_inference_steps=num_inference_steps,
    generator=torch.Generator().manual_seed(42),
    callback_on_step_end=callback,
).images[0]
```

## 热切换

LoRA 热切换（hotswapping）是一种高效的多 LoRA 工作方式。它可以避免多次调用 [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 带来的额外内存累积；在某些情况下，如果模型已经编译，还可以避免重新编译。这个工作流要求你先加载一个 LoRA，因为新的 LoRA 权重会原地替换当前已加载的 LoRA。

```py
import torch
from diffusers import DiffusionPipeline

# 加载基础模型和 LoRA
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/ikea-instructions-lora-sdxl",
    weight_name="ikea_instructions_xl_v1_5.safetensors",
    adapter_name="ikea"
)
```

> [!WARNING]
> 目标是文本编码器的 LoRA 目前不支持热切换。

在 [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 中设置 `hotswap=True`，即可替换第二个 LoRA。使用 `adapter_name` 参数指定要替换的是哪个 LoRA（默认名字是 `default_0`）。

```py
pipeline.load_lora_weights(
    "lordjia/by-feng-zikai",
    hotswap=True,
    adapter_name="ikea"
)
```

### 编译模型

对于已经编译的模型，可以使用 [`~loaders.lora_base.LoraBaseMixin.enable_lora_hotswap`] 来避免热切换时重新编译。这个方法应该在加载第一个 LoRA *之前*调用，而 `torch.compile` 则应该在加载第一个 LoRA *之后*调用。

> [!TIP]
> 如果第二个 LoRA 与第一个 LoRA 的 rank 和 scale 完全一致，那么 [`~loaders.lora_base.LoraBaseMixin.enable_lora_hotswap`] 不一定是必需的。

在 [`~loaders.lora_base.LoraBaseMixin.enable_lora_hotswap`] 中，`target_rank` 参数很重要，它决定了所有 LoRA adapter 的 rank。设为 `max_rank` 时，会自动取最大的 rank；如果 LoRA 的 rank 不同，你也可以手动设为更高的值。默认 rank 是 128。

```py
import torch
from diffusers import DiffusionPipeline

# 加载基础模型和 LoRA
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
# 1. 启用 enable_lora_hotswap
pipeline.enable_lora_hotswap(target_rank=max_rank)
pipeline.load_lora_weights(
    "ostris/ikea-instructions-lora-sdxl",
    weight_name="ikea_instructions_xl_v1_5.safetensors",
    adapter_name="ikea"
)
# 2. torch.compile
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

# 3. 热切换
pipeline.load_lora_weights(
    "lordjia/by-feng-zikai",
    hotswap=True,
    adapter_name="ikea"
)
```

> [!TIP]
> 你可以把代码放进 `with torch._dynamo.config.patch(error_on_recompile=True)` 上下文中，用来检测模型是否发生了重新编译。如果你严格按照上面的步骤做了，模型依然重新编译，请带着可复现示例提交一个 [issue](https://github.com/huggingface/diffusers/issues)。

如果你预计在推理时会使用不同分辨率，请在编译时设置 `dynamic=True`。更多细节可以参考[这篇文档](../optimization/fp16#dynamic-shape-compilation)。

有些情况下，重新编译依然无法避免，例如热切换进来的 LoRA 比初始 adapter 覆盖了更多层。这时，尽量*先*加载那个覆盖层数最多的 LoRA。关于这个限制的更多说明，可以参考 PEFT 的 [hotswapping](https://huggingface.co/docs/peft/main/en/package_reference/hotswap#peft.utils.hotswap.hotswap_adapter) 文档。

<details>
<summary>热切换的技术细节</summary>

[`~loaders.lora_base.LoraBaseMixin.enable_lora_hotswap`] 会把 LoRA 的缩放因子从 float 转成 torch.tensor，并把权重形状补齐到所需的最大形状，这样在替换权重数据时，就不用重新分配整个属性。

这也是为什么 `max_rank` 参数很重要。即使补出来的部分是零，也不会改变最终结果，只是补齐量越大，计算速度可能会更慢一些。

由于不会新增新的 LoRA 属性，因此后续热切换进来的 LoRA 只能作用于与第一个 LoRA 相同的层，或者其子集。LoRA 的加载顺序因此会很关键。如果多个 LoRA 的目标层彼此不相交，你最终可能需要先构造一个覆盖所有目标层并集的 dummy LoRA。

如果想了解更多实现细节，可以直接查看 [`hotswap.py`](https://github.com/huggingface/peft/blob/92d65cafa51c829484ad3d95cf71d09de57ff066/src/peft/utils/hotswap.py) 文件。

</details>

## 合并

你可以把多个 LoRA 的权重合并在一起，得到多种现有风格的混合效果。LoRA 合并有多种方法，不同方法主要区别在于*如何*合并权重，这也可能影响生成质量。

### set_adapters

[`~loaders.PeftAdapterMixin.set_adapters`] 会通过拼接多个 LoRA 的加权矩阵来完成合并。把 LoRA 名称传给 [`~loaders.PeftAdapterMixin.set_adapters`]，再通过 `adapter_weights` 参数控制每个 LoRA 的缩放权重。例如，当 `adapter_weights=[0.5, 0.5]` 时，输出就是两个 LoRA 的平均效果。

> [!TIP]
> `"scale"` 参数决定了应用合并后 LoRA 的强度。详情可参考前面的 [权重缩放](#权重缩放) 部分。

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/ikea-instructions-lora-sdxl",
    weight_name="ikea_instructions_xl_v1_5.safetensors",
    adapter_name="ikea"
)
pipeline.load_lora_weights(
    "lordjia/by-feng-zikai",
    weight_name="fengzikai_v1.0_XL.safetensors",
    adapter_name="feng"
)
pipeline.set_adapters(["ikea", "feng"], adapter_weights=[0.7, 0.8])
# 在提示词中使用 by Feng Zikai 来激活 lordjia/by-feng-zikai 这个 LoRA
pipeline("A bowl of ramen shaped like a cute kawaii bear, by Feng Zikai", cross_attention_kwargs={"scale": 1.0}).images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lora_merge_set_adapters.png"/>
</div>

### add_weighted_adapter

> [!TIP]
> 这是一个实验性方法。更多背景可以参考 PEFT 的 [Model merging](https://huggingface.co/docs/peft/developer_guides/model_merging) 文档。如果你想了解这项集成背后的动机和设计，也可以看看这个 [issue](https://github.com/huggingface/diffusers/issues/6892)。

[`~peft.LoraModel.add_weighted_adapter`] 支持使用更高效的合并方法，比如 [TIES](https://huggingface.co/papers/2306.01708) 或 [DARE](https://huggingface.co/papers/2311.03099)。这些方法会从合并后的模型中移除冗余或可能互相干扰的参数。需要注意的是，要进行合并，各个 LoRA 的 rank 必须一致。

请先确保安装的是最新版稳定版 Diffusers 和 PEFT。

```bash
pip install -U -q diffusers peft
```

先加载一个与 LoRA UNet 对应的 UNet。

```py
import copy
import torch
from diffusers import AutoModel, DiffusionPipeline
from peft import get_peft_model, LoraConfig, PeftModel

unet = AutoModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    subfolder="unet",
).to("cuda")
```

加载一个管道，把这个 UNet 传进去，然后再加载 LoRA。

```py
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    torch_dtype=torch.float16,
    unet=unet
).to("cuda")
pipeline.load_lora_weights(
    "ostris/ikea-instructions-lora-sdxl",
    weight_name="ikea_instructions_xl_v1_5.safetensors",
    adapter_name="ikea"
)
```

通过前面加载的第一个 UNet 和管道中的 LoRA UNet，创建一个来自该 LoRA 检查点的 [`~peft.PeftModel`]。

```py
sdxl_unet = copy.deepcopy(unet)
ikea_peft_model = get_peft_model(
    sdxl_unet,
    pipeline.unet.peft_config["ikea"],
    adapter_name="ikea"
)

original_state_dict = {f"base_model.model.{k}": v for k, v in pipeline.unet.state_dict().items()}
ikea_peft_model.load_state_dict(original_state_dict, strict=True)
```

> [!TIP]
> 你也可以像下面这样把 `ikea_peft_model` 推送到 Hub，之后保存并复用。
> ```py
> ikea_peft_model.push_to_hub("ikea_peft_model", token=TOKEN)
> ```

重复这一步，为第二个 LoRA 再创建一个 [`~peft.PeftModel`]。

```py
pipeline.delete_adapters("ikea")
sdxl_unet.delete_adapters("ikea")

pipeline.load_lora_weights(
    "lordjia/by-feng-zikai",
    weight_name="fengzikai_v1.0_XL.safetensors",
    adapter_name="feng"
)
pipeline.set_adapters(adapter_names="feng")

feng_peft_model = get_peft_model(
    sdxl_unet,
    pipeline.unet.peft_config["feng"],
    adapter_name="feng"
)

original_state_dict = {f"base_model.model.{k}": v for k, v in pipe.unet.state_dict().items()}
feng_peft_model.load_state_dict(original_state_dict, strict=True)
```

加载一个基础 UNet，并加载 adapters。

```py
base_unet = AutoModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    subfolder="unet",
).to("cuda")

model = PeftModel.from_pretrained(
    base_unet,
    "stevhliu/ikea_peft_model",
    use_safetensors=True,
    subfolder="ikea",
    adapter_name="ikea"
)
model.load_adapter(
    "stevhliu/feng_peft_model",
    use_safetensors=True,
    subfolder="feng",
    adapter_name="feng"
)
```

使用 [`~peft.LoraModel.add_weighted_adapter`] 合并 LoRA，并通过 `combination_type` 指定合并方式。下面的例子使用 `"dare_linear"` 方法（想了解这些合并方法，可以参考[这篇博客](https://huggingface.co/blog/peft_merging)），它会先随机裁剪一部分权重，再根据 `weights` 中给定的权重，对各个 LoRA 的张量做加权求和。

再使用 [`~loaders.PeftAdapterMixin.set_adapters`] 激活合并后的 LoRA。

```py
model.add_weighted_adapter(
    adapters=["ikea", "feng"],
    combination_type="dare_linear",
    weights=[1.0, 1.0],
    adapter_name="ikea-feng"
)
model.set_adapters("ikea-feng")

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    unet=model,
    variant="fp16",
    torch_dtype=torch.float16,
).to("cuda")
pipeline("A bowl of ramen shaped like a cute kawaii bear, by Feng Zikai").images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ikea-feng-dare-linear.png"/>
</div>

### fuse_lora

[`~loaders.lora_base.LoraBaseMixin.fuse_lora`] 会把 LoRA 权重直接融合到基础模型底层的 UNet 和文本编码器权重中。这样做可以减少每个 LoRA 都重新加载底层模型的开销，因为基础模型只需加载一次，从而降低内存占用并提升推理速度。

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/ikea-instructions-lora-sdxl",
    weight_name="ikea_instructions_xl_v1_5.safetensors",
    adapter_name="ikea"
)
pipeline.load_lora_weights(
    "lordjia/by-feng-zikai",
    weight_name="fengzikai_v1.0_XL.safetensors",
    adapter_name="feng"
)
pipeline.set_adapters(["ikea", "feng"], adapter_weights=[0.7, 0.8])
```

调用 [`~loaders.lora_base.LoraBaseMixin.fuse_lora`] 进行融合。`lora_scale` 参数控制 LoRA 权重对输出的缩放强度。这里必须现在就设置好，因为在这个场景下，向 `cross_attention_kwargs` 传 `scale` 不会生效。

```py
pipeline.fuse_lora(adapter_names=["ikea", "feng"], lora_scale=1.0)
```

由于 LoRA 权重已经融合到底层模型中，可以把它们卸载掉。然后通过 [`~DiffusionPipeline.save_pretrained`] 保存到本地，或者通过 [`~PushToHubMixin.push_to_hub`] 保存到 Hub。

<hfoptions id="save">
<hfoption id="save locally">

```py
pipeline.unload_lora_weights()
pipeline.save_pretrained("path/to/fused-pipeline")
```

</hfoption>
<hfoption id="save to Hub">

```py
pipeline.unload_lora_weights()
pipeline.push_to_hub("fused-ikea-feng")
```

</hfoption>
</hfoptions>

之后，你就可以快速加载这个融合后的管道进行推理，而不需要分别加载每个 LoRA。

```py
pipeline = DiffusionPipeline.from_pretrained(
    "username/fused-ikea-feng", torch_dtype=torch.float16,
).to("cuda")
pipeline("A bowl of ramen shaped like a cute kawaii bear, by Feng Zikai").images[0]
```

如果你想恢复底层模型原始权重，例如想改用不同的 `lora_scale`，可以使用 [`~loaders.LoraLoaderMixin.unfuse_lora`]。不过只有融合了单个 LoRA 时才能反融合。比如上面那个含多个融合 LoRA 的管道就无法这样做，这种情况下你需要重新加载整个模型。

```py
pipeline.unfuse_lora()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/fuse_lora.png"/>
</div>

## 管理

Diffusers 提供了多种方法来帮助你管理 LoRA，尤其是在同时使用多个 LoRA 时会很有帮助。

### set_adapters

[`~loaders.PeftAdapterMixin.set_adapters`] 也会在多个活跃 LoRA 中激活当前要使用的那个 LoRA。你可以通过指定名字，在不同 LoRA 之间切换。

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/ikea-instructions-lora-sdxl",
    weight_name="ikea_instructions_xl_v1_5.safetensors",
    adapter_name="ikea"
)
pipeline.load_lora_weights(
    "lordjia/by-feng-zikai",
    weight_name="fengzikai_v1.0_XL.safetensors",
    adapter_name="feng"
)
# 激活 feng LoRA，而不是 ikea LoRA
pipeline.set_adapters("feng")
```

### save_lora_adapter

使用 [`~loaders.PeftAdapterMixin.save_lora_adapter`] 保存 adapter。

```py
import torch
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.unet.load_lora_adapter(
    "jbilcke-hf/sdxl-cinematic-1",
    weight_name="pytorch_lora_weights.safetensors",
    adapter_name="cinematic"
    prefix="unet"
)
pipeline.save_lora_adapter("path/to/save", adapter_name="cinematic")
```

### unload_lora_weights

[`~loaders.lora_base.LoraBaseMixin.unload_lora_weights`] 会卸载管道中的所有 LoRA 权重，并恢复到底层模型原始权重。

```py
pipeline.unload_lora_weights()
```

### disable_lora

[`~loaders.PeftAdapterMixin.disable_lora`] 会禁用所有 LoRA（但仍保留在管道中），并让管道恢复到底层模型权重。

```py
pipeline.disable_lora()
```

### get_active_adapters

[`~loaders.lora_base.LoraBaseMixin.get_active_adapters`] 会返回挂载在管道上的活跃 LoRA 列表。

```py
pipeline.get_active_adapters()
["cereal", "ikea"]
```

### get_list_adapters

[`~loaders.lora_base.LoraBaseMixin.get_list_adapters`] 会返回管道中每个组件当前有哪些活跃 LoRA。

```py
pipeline.get_list_adapters()
{"unet": ["cereal", "ikea"], "text_encoder_2": ["cereal"]}
```

### delete_adapters

[`~loaders.PeftAdapterMixin.delete_adapters`] 会把某个 LoRA 及其对应层从模型中彻底移除。

```py
pipeline.delete_adapters("ikea")
```

## 资源

你可以在 [LoRA Studio](https://lorastudio.co/models) 浏览可用的 LoRA，也可以使用下面这个 Civitai Space，把自己喜欢的 LoRA 上传到 Hub。

<iframe
	src="https://multimodalart-civitai-to-hf.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

你还可以在 [FLUX LoRA the Explorer](https://huggingface.co/spaces/multimodalart/flux-lora-the-explorer) 和 [LoRA the Explorer](https://huggingface.co/spaces/multimodalart/LoraTheExplorer) 这两个仓库中找到更多 LoRA。

如果你想了解如何结合 FlashAttention-3 和 fp8 量化等方法优化 LoRA 推理，也可以看看这篇博客：[Fast LoRA inference for Flux with Diffusers and PEFT](https://huggingface.co/blog/lora-fast)。
