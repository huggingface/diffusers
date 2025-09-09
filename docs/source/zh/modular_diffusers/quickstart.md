<!--版权所有 2025 The HuggingFace Team。保留所有权利。

根据Apache许可证2.0版（"许可证"）授权；除非符合许可证，否则不得使用此文件。您可以在

http://www.apache.org/licenses/LICENSE-2.0

获取许可证的副本。

除非适用法律要求或书面同意，根据许可证分发的软件按"原样"分发，无任何明示或暗示的担保或条件。有关许可证下特定语言的管理权限和限制，请参阅许可证。
-->

# 快速入门

模块化Diffusers是一个快速构建灵活和可定制管道的框架。模块化Diffusers的核心是[`ModularPipelineBlocks`]，可以与其他块组合以适应新的工作流程。这些块被转换为[`ModularPipeline`]，一个开发者可以使用的友好用户界面。

本文档将向您展示如何使用模块化框架实现[Differential Diffusion](https://differential-diffusion.github.io/)管道。

## ModularPipelineBlocks

[`ModularPipelineBlocks`]是*定义*，指定管道中单个步骤的组件、输入、输出和计算逻辑。有四种类型的块。

- [`ModularPipelineBlocks`]是最基本的单一步骤块。
- [`SequentialPipelineBlocks`]是一个多块，线性组合其他块。一个块的输出是下一个块的输入。
- [`LoopSequentialPipelineBlocks`]是一个多块，迭代运行，专为迭代工作流程设计。
- [`AutoPipelineBlocks`]是一个针对不同工作流程的块集合，它根据输入选择运行哪个块。它旨在方便地将多个工作流程打包到单个管道中。

[Differential Diffusion](https://differential-diffusion.github.io/)是一个图像到图像的工作流程。从`IMAGE2IMAGE_BLOCKS`预设开始，这是一个用于图像到图像生成的`ModularPipelineBlocks`集合。

```py
from diffusers.modular_pipelines.stable_diffusion_xl import IMAGE2IMAGE_BLOCKS
IMAGE2IMAGE_BLOCKS = InsertableDict([
    ("text_encoder", StableDiffusionXLTextEncoderStep),
    ("image_encoder", StableDiffusionXLVaeEncoderStep),
    ("input", StableDiffusionXLInputStep),
    ("set_timesteps", StableDiffusionXLImg2ImgSetTimestepsStep),
    ("prepare_latents", StableDiffusionXLImg2ImgPrepareLatentsStep),
    ("prepare_add_cond", StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep),
    ("denoise", StableDiffusionXLDenoiseStep),
    ("decode", StableDiffusionXLDecodeStep)
])
```

## 管道和块状态

模块化Diffusers使用*状态*在块之间通信数据。有两种类型的状态。

- [`PipelineState`]是一个全局状态，可用于跟踪所有块的所有输入和输出。
- [`BlockState`]是[`PipelineState`]中相关变量的局部视图，用于单个块。

## 自定义块

[Differential Diffusion](https://differential-diffusion.github.io/) 与标准的图像到图像转换在其 `prepare_latents` 和 `denoise` 块上有所不同。所有其他块都可以重用，但你需要修改这两个。

通过复制和修改现有的块，为 `prepare_latents` 和 `denoise` 创建占位符 `ModularPipelineBlocks`。

打印 `denoise` 块，可以看到它由 [`LoopSequentialPipelineBlocks`] 组成，包含三个子块，`before_denoiser`、`denoiser` 和 `after_denoiser`。只需要修改 `before_denoiser` 子块，根据变化图为去噪器准备潜在输入。

```py
denoise_blocks = IMAGE2IMAGE_BLOCKS["denoise"]()
print(denoise_blocks)
```

用新的 `SDXLDiffDiffLoopBeforeDenoiser` 块替换 `StableDiffusionXLLoopBeforeDenoiser` 子块。

```py
# 复制现有块作为占位符
class SDXLDiffDiffPrepareLatentsStep(ModularPipelineBlocks):
    """Copied from StableDiffusionXLImg2ImgPrepareLatentsStep - will modify later"""
    # ... 与 StableDiffusionXLImg2ImgPrepareLatentsStep 相同的实现

class SDXLDiffDiffDenoiseStep(StableDiffusionXLDenoiseLoopWrapper):
    block_classes = [SDXLDiffDiffLoopBeforeDenoiser, StableDiffusionXLLoopDenoiser, StableDiffusionXLLoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]
```

### prepare_latents

`prepare_latents` 块需要进行以下更改。

- 一个处理器来处理变化图
- 一个新的 `inputs` 来接受用户提供的变化图，`timestep` 用于预计算所有潜在变量和 `num_inference_steps` 来创建更新图像区域的掩码
- 更新 `__call__` 方法中的计算，用于处理变化图和创建掩码，并将其存储在 [`BlockState`] 中

```diff
class SDXLDiffDiffPrepareLatentsStep(ModularPipelineBlocks):
    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec("scheduler", EulerDiscreteScheduler),
+           ComponentSpec("mask_processor", VaeImageProcessor, config=FrozenDict({"do_normalize": False, "do_convert_grayscale": True}))
        ]
    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("generator"),
+           InputParam("diffdiff_map", required=True),
-           InputParam("latent_timestep", required=True, type_hint=torch.Tensor),
+           InputParam("timesteps", type_hint=torch.Tensor),
+           InputParam("num_inference_steps", type_hint=int),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
+           OutputParam("original_latents", type_hint=torch.Tensor),
+           OutputParam("diffdiff_masks", type_hint=torch.Tensor),
        ]
    def __call__(self, components, state: PipelineState):
        # ... existing logic ...
+       # Process change map and create masks
+       diffdiff_map = components.mask_processor.preprocess(block_state.diffdiff_map, height=latent_height, width=latent_width)
+       thresholds = torch.arange(block_state.num_inference_steps, dtype=diffdiff_map.dtype) / block_state.num_inference_steps
+       block_state.diffdiff_masks = diffdiff_map > (thresholds + (block_state.denoising_start or 0))
+       block_state.original_latents = block_state.latents
```

### 去噪

`before_denoiser` 子块需要进行以下更改。

- 新的 `inputs` 以接受 `denoising_start` 参数，`original_latents` 和 `diffdiff_masks` 来自 `prepare_latents` 块
- 更新 `__call__` 方法中的计算以应用 Differential Diffusion

```diff
class SDXLDiffDiffLoopBeforeDenoiser(ModularPipelineBlocks):
    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop for differential diffusion that prepare the latent input for the denoiser"
        )

    @property
    def inputs(self) -> List[str]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor),
+           InputParam("denoising_start"),
+           InputParam("original_latents", type_hint=torch.Tensor),
+           InputParam("diffdiff_masks", type_hint=torch.Tensor),
        ]

    def __call__(self, components, block_state, i, t):
+       # Apply differential diffusion logic
+       if i == 0 and block_state.denoising_start is None:
+           block_state.latents = block_state.original_latents[:1]
+       else:
+           block_state.mask = block_state.diffdiff_masks[i].unsqueeze(0).unsqueeze(1)
+           block_state.latents = block_state.original_latents[i] * block_state.mask + block_state.latents * (1 - block_state.mask)

        # ... rest of existing logic ...
```

## 组装块

此时，您应该拥有创建 [`ModularPipeline`] 所需的所有块。

复制现有的 `IMAGE2IMAGE_BLOCKS` 预设，对于 `set_timesteps` 块，使用 `TEXT2IMAGE_BLOCKS` 中的 `set_timesteps`，因为 Differential Diffusion 不需要 `strength` 参数。

将 `prepare_latents` 和 `denoise` 块设置为您刚刚修改的 `SDXLDiffDiffPrepareLatentsStep` 和 `SDXLDiffDiffDenoiseStep` 块。

调用 [`SequentialPipelineBlocks.from_blocks_dict`] 在块上创建一个 `SequentialPipelineBlocks`。

```py
DIFFDIFF_BLOCKS = IMAGE2IMAGE_BLOCKS.copy()
DIFFDIFF_BLOCKS["set_timesteps"] = TEXT2IMAGE_BLOCKS["set_timesteps"]
DIFFDIFF_BLOCKS["prepare_latents"] = SDXLDiffDiffPrepareLatentsStep
DIFFDIFF_BLOCKS["denoise"] = SDXLDiffDiffDenoiseStep

dd_blocks = SequentialPipelineBlocks.from_blocks_dict(DIFFDIFF_BLOCKS)
print(dd_blocks)
```

## ModularPipeline

将 [`SequentialPipelineBlocks`] 转换为 [`ModularPipeline`]，使用 [`ModularPipeline.init_pipeline`] 方法。这会初始化从 `modular_model_index.json` 文件加载的预期组件。通过调用 [`ModularPipeline.load_defau
lt_components`]。

初始化[`ComponentManager`]时传入pipeline是一个好主意，以帮助管理不同的组件。一旦调用[`~ModularPipeline.load_components`]，组件就会被注册到[`ComponentManager`]中，并且可以在工作流之间共享。下面的例子使用`collection`参数为组件分配了一个`"diffdiff"`标签，以便更好地组织。

```py
from diffusers.modular_pipelines import ComponentsManager

components = ComponentManager()

dd_pipeline = dd_blocks.init_pipeline("YiYiXu/modular-demo-auto", components_manager=components, collection="diffdiff")
dd_pipeline.load_default_componenets(torch_dtype=torch.float16)
dd_pipeline.to("cuda")
```

## 添加工作流

可以向[`ModularPipeline`]添加其他工作流以支持更多功能，而无需从头重写整个pipeline。

本节演示如何添加IP-Adapter或ControlNet。

### IP-Adapter

Stable Diffusion XL已经有一个预设的IP-Adapter块，你可以使用，并且不需要对现有的Differential Diffusion pipeline进行任何更改。

```py
from diffusers.modular_pipelines.stable_diffusion_xl.encoders import StableDiffusionXLAutoIPAdapterStep

ip_adapter_block = StableDiffusionXLAutoIPAdapterStep()
```

使用[`sub_blocks.insert`]方法将其插入到[`ModularPipeline`]中。下面的例子在位置`0`插入了`ip_adapter_block`。打印pipeline可以看到`ip_adapter_block`被添加了，并且它需要一个`ip_adapter_image`。这也向pipeline添加了两个组件，`image_encoder`和`feature_extractor`。

```py
dd_blocks.sub_blocks.insert("ip_adapter", ip_adapter_block, 0)
```

调用[`~ModularPipeline.init_pipeline`]来初始化一个[`ModularPipeline`]，并使用[`~ModularPipeline.load_components`]加载模型组件。加载并设置IP-Adapter以运行pipeline。

```py
dd_pipeline = dd_blocks.init_pipeline("YiYiXu/modular-demo-auto", collection="diffdiff")
dd_pipeline.load_components(torch_dtype=torch.float16)
dd_pipeline.loader.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
dd_pipeline.loader.set_ip_adapter_scale(0.6)
dd_pipeline = dd_pipeline.to(device)

ip_adapter_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/diffdiff_orange.jpeg")
image = load_image("https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/differential/20240329211129_4024911930.png?download=true")
mask = load_image("https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/differential/gradient_mask.png?download=true")

prompt = "a green pear"
negative_prompt = "blurry"
generator = torch.Generator(device=device).manual_seed(42)

image = dd_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    generator=generator,
    ip_adapter_image=ip_adapter_image,
    diffdiff_map=mask,
    image=image,

output="images"
)[0]
```

### ControlNet

Stable Diffusion XL 已经预设了一个可以立即使用的 ControlNet 块。

```py
from diffusers.modular_pipelines.stable_diffusion_xl.modular_blocks import StableDiffusionXLAutoControlNetInputStep

control_input_block = StableDiffusionXLAutoControlNetInputStep()
```

然而，它需要修改 `denoise` 块，因为那是 ControlNet 将控制信息注入到 UNet 的地方。

通过将 `StableDiffusionXLLoopDenoiser` 子块替换为 `StableDiffusionXLControlNetLoopDenoiser` 来修改 `denoise` 块。

```py
class SDXLDiffDiffControlNetDenoiseStep(StableDiffusionXLDenoiseLoopWrapper):
    block_classes = [SDXLDiffDiffLoopBeforeDenoiser, StableDiffusionXLControlNetLoopDenoiser, StableDiffusionXLDenoiseLoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

controlnet_denoise_block = SDXLDiffDiffControlNetDenoiseStep()
```

插入 `controlnet_input` 块并用新的 `controlnet_denoise_block` 替换 `denoise` 块。初始化一个 [`ModularPipeline`] 并将 [`~ModularPipeline.load_components`] 加载到其中。

```py
dd_blocks.sub_blocks.insert("controlnet_input", control_input_block, 7)
dd_blocks.sub_blocks["denoise"] = controlnet_denoise_block

dd_pipeline = dd_blocks.init_pipeline("YiYiXu/modular-demo-auto", collection="diffdiff")
dd_pipeline.load_components(torch_dtype=torch.float16)
dd_pipeline = dd_pipeline.to(device)

control_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/diffdiff_tomato_canny.jpeg")
image = load_image("https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/differential/20240329211129_4024911930.png?download=true")
mask = load_image("https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/differential/gradient_mask.png?download=true")

prompt = "a green pear"
negative_prompt = "blurry"
generator = torch.Generator(device=device).manual_seed(42)

image = dd_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    generator=generator,
    control_image=control_image,
    controlnet_conditioning_scale=0.5,
    diffdiff_map=mask,
    image=image,
    output="images"
)[0]
```

### AutoPipelineBlocks

差分扩散、IP-Adapter 和 ControlNet 工作流可以通过使用 [`AutoPipelineBlocks`] 捆绑到一个单一的 [`ModularPipeline`] 中。这允许根据输入如 `control_image` 或 `ip_adapter_image` 自动选择要运行的子块。如果没有传递这些输入，则默认为差分扩散。

使用 `block_trigger_inputs` 仅在提供 `control_image` 输入时运行 `SDXLDiffDiffControlNetDenoiseStep` 块。否则，使用 `SDXLDiffDiffDenoiseStep`。

```py
class SDXLDiffDiffAutoDenoiseStep(AutoPipelineBlocks):
    block_classes = [SDXLDiffDiffControlNetDenoiseStep, SDXLDiffDiffDenoiseStep]
    block_names = ["contr
olnet_denoise", "denoise"]
block_trigger_inputs = ["controlnet_cond", None]
```

添加 `ip_adapter` 和 `controlnet_input` 块。

```py
DIFFDIFF_AUTO_BLOCKS = IMAGE2IMAGE_BLOCKS.copy()
DIFFDIFF_AUTO_BLOCKS["prepare_latents"] = SDXLDiffDiffPrepareLatentsStep
DIFFDIFF_AUTO_BLOCKS["set_timesteps"] = TEXT2IMAGE_BLOCKS["set_timesteps"]
DIFFDIFF_AUTO_BLOCKS["denoise"] = SDXLDiffDiffAutoDenoiseStep
DIFFDIFF_AUTO_BLOCKS.insert("ip_adapter", StableDiffusionXLAutoIPAdapterStep, 0)
DIFFDIFF_AUTO_BLOCKS.insert("controlnet_input",StableDiffusionXLControlNetAutoInput, 7)
```

调用 [`SequentialPipelineBlocks.from_blocks_dict`] 来创建一个 [`SequentialPipelineBlocks`] 并创建一个 [`ModularPipeline`] 并加载模型组件以运行。

```py
dd_auto_blocks = SequentialPipelineBlocks.from_blocks_dict(DIFFDIFF_AUTO_BLOCKS)
dd_pipeline = dd_auto_blocks.init_pipeline("YiYiXu/modular-demo-auto", collection="diffdiff")
dd_pipeline.load_components(torch_dtype=torch.float16)
```

## 分享

使用 [`~ModularPipeline.save_pretrained`] 将您的 [`ModularPipeline`] 添加到 Hub，并将 `push_to_hub` 参数设置为 `True`。

```py
dd_pipeline.save_pretrained("YiYiXu/test_modular_doc", push_to_hub=True)
```

其他用户可以使用 [`~ModularPipeline.from_pretrained`] 加载 [`ModularPipeline`]。

```py
import torch
from diffusers.modular_pipelines import ModularPipeline, ComponentsManager

components = ComponentsManager()

diffdiff_pipeline = ModularPipeline.from_pretrained("YiYiXu/modular-diffdiff-0704", trust_remote_code=True, components_manager=components, collection="diffdiff")
diffdiff_pipeline.load_components(torch_dtype=torch.float16)
```
