<!--Copyright 2025 The HuggingFace Team. 保留所有权利。

根据Apache许可证2.0版（"许可证"）授权；除非符合许可证要求，否则不得使用此文件。您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件均按"原样"分发，不附带任何明示或暗示的担保或条件。有关许可证下特定语言规定的权限和限制，请参阅许可证。
-->

# 如何为Diffusers 🧨做贡献

我们❤️来自开源社区的贡献！欢迎所有人参与，所有类型的贡献——不仅仅是代码——都受到重视和赞赏。回答问题、帮助他人、主动交流以及改进文档对社区都极具价值，所以如果您愿意参与，请不要犹豫！

我们鼓励每个人先在公开Discord频道里打招呼👋。在那里我们讨论扩散模型的最新趋势、提出问题、展示个人项目、互相协助贡献，或者只是闲聊☕。<a href="https://Discord.gg/G7tWnz98XR"><img alt="加入Discord社区" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>

无论您选择以何种方式贡献，我们都致力于成为一个开放、友好、善良的社区。请阅读我们的[行为准则](https://github.com/huggingface/diffusers/blob/main/CODE_OF_CONDUCT.md)，并在互动时注意遵守。我们也建议您了解指导本项目的[伦理准则](https://huggingface.co/docs/diffusers/conceptual/ethical_guidelines)，并请您遵循同样的透明度和责任原则。

我们高度重视社区的反馈，所以如果您认为自己有能帮助改进库的有价值反馈，请不要犹豫说出来——每条消息、评论、issue和拉取请求（PR）都会被阅读和考虑。

## 概述

您可以通过多种方式做出贡献，从在issue和讨论区回答问题，到向核心库添加新的diffusion模型。

下面我们按难度升序列出不同的贡献方式，所有方式对社区都很有价值：

* 1. 在[Diffusers讨论论坛](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers)或[Discord](https://discord.gg/G7tWnz98XR)上提问和回答问题
* 2. 在[GitHub Issues标签页](https://github.com/huggingface/diffusers/issues/new/choose)提交新issue，或在[GitHub Discussions标签页](https://github.com/huggingface/diffusers/discussions/new/choose)发起新讨论
* 3. 在[GitHub Issues标签页](https://github.com/huggingface/diffusers/issues)解答issue，或在[GitHub Discussions标签页](https://github.com/huggingface/diffusers/discussions)参与讨论
* 4. 解决标记为"Good first issue"的简单问题，详见[此处](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
* 5. 参与[文档](https://github.com/huggingface/diffusers/tree/main/docs/source)建设
* 6. 贡献[社区Pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3Acommunity-examples)
* 7. 完善[示例代码](https://github.com/huggingface/diffusers/tree/main/examples)
* 8. 解决标记为"Good second issue"的中等难度问题，详见[此处](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22Good+second+issue%22)
* 9. 添加新pipeline/模型/调度器，参见["New Pipeline/Model"](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)和["New scheduler"](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)类issue。此类贡献请先阅读[设计哲学](https://github.com/huggingface/diffusers/blob/main/PHILOSOPHY.md)

重申：**所有贡献对社区都具有重要价值。**下文将详细说明各类贡献方式。

对于4-9类贡献，您需要提交PR（拉取请求），具体操作详见[如何提交PR](#how-to-open-a-pr)章节。

### 1. 在Diffusers讨论区或Discord提问与解答

任何与Diffusers库相关的问题或讨论都可以发布在[官方论坛](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers/)或[Discord频道](https://discord.gg/G7tWnz98XR)，包括但不限于：
- 分享训练/推理实验报告
- 展示个人项目
- 咨询非官方训练示例
- 项目提案
- 通用反馈
- 论文解读
- 基于Diffusers库的个人项目求助
- 一般性问题
- 关于diffusion模型的伦理讨论
- ...

论坛/Discord上的每个问题都能促使社区公开分享知识，很可能帮助未来遇到相同问题的初学者。请务必提出您的疑问。
同样地，通过回答问题您也在为社区创造公共知识文档，这种贡献极具价值。

**请注意**：提问/回答时投入的精力越多，产生的公共知识质量就越高。精心构建的问题与专业解答能形成高质量知识库，而表述不清的问题则可能降低讨论价值。

低质量的问题或回答会降低公共知识库的整体质量。  
简而言之，高质量的问题或回答应具备*精确性*、*简洁性*、*相关性*、*易于理解*、*可访问性*和*格式规范/表述清晰*等特质。更多详情请参阅[如何提交优质议题](#how-to-write-a-good-issue)章节。

**关于渠道的说明**：  
[*论坛*](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers/63)的内容能被谷歌等搜索引擎更好地收录，且帖子按热度而非时间排序，便于查找历史问答。此外，论坛内容更容易被直接链接引用。  
而*Discord*采用即时聊天模式，适合快速交流。虽然在Discord上可能更快获得解答，但信息会随时间淹没，且难以回溯历史讨论。因此我们强烈建议在论坛发布优质问答，以构建可持续的社区知识库。若Discord讨论产生有价值结论，建议将成果整理发布至论坛以惠及更多读者。

### 2. 在GitHub议题页提交新议题

🧨 Diffusers库的稳健性离不开用户的问题反馈，感谢您的报错。

请注意：GitHub议题仅限处理与Diffusers库代码直接相关的技术问题、错误报告、功能请求或库设计反馈。  
简言之，**与Diffusers库代码（含文档）无关**的内容应发布至[论坛](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers/63)或[Discord](https://discord.gg/G7tWnz98XR)。

**提交新议题时请遵循以下准则**：
- 确认是否已有类似议题（使用GitHub议题页的搜索栏）
- 请勿在现有议题下追加新问题。若存在高度关联议题，应新建议题并添加相关链接
- 确保使用英文提交。非英语用户可通过[DeepL](https://www.deepl.com/translator)等免费工具翻译
- 检查升级至最新Diffusers版本是否能解决问题。提交前请确认`python -c "import diffusers; print(diffusers.__version__)"`显示的版本号不低于最新版本
- 记请记住，你在提交新issue时投入的精力越多，得到的回答质量就越高，Diffusers项目的整体issue质量也会越好。

新issue通常包含以下内容：

#### 2.1 可复现的最小化错误报告

错误报告应始终包含可复现的代码片段，并尽可能简洁明了。具体而言：
- 尽量缩小问题范围，**不要直接粘贴整个代码文件**
- 规范代码格式
- 除Diffusers依赖库外，不要包含其他外部库
- **务必**提供环境信息：可在终端运行`diffusers-cli env`命令，然后将显示的信息复制到issue中
- 详细说明问题。如果读者不清楚问题所在及其影响，就无法解决问题
- **确保**读者能以最小成本复现问题。如果代码片段因缺少库或未定义变量而无法运行，读者将无法提供帮助。请确保提供的可复现代码尽可能精简，可直接复制到Python shell运行
- 如需特定模型/数据集复现问题，请确保读者能获取这些资源。可将模型/数据集上传至[Hub](https://huggingface.co)便于下载。尽量保持模型和数据集体积最小化，降低复现难度

更多信息请参阅[如何撰写优质issue](#how-to-write-a-good-issue)章节。

提交错误报告请点击[此处](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=bug&projects=&template=bug-report.yml)。

#### 2.2 功能请求

优质的功能请求应包含以下要素：

1. 首先说明动机：
* 是否与库的使用痛点相关？若是，请解释原因，最好提供演示问题的代码片段
* 是否因项目需求产生？我们很乐意了解详情！
* 是否是你已实现且认为对社区有价值的功能？请说明它为你解决了什么问题
2. 用**完整段落**描述功能特性
3. 提供**代码片段**演示预期用法
4. 如涉及论文，请附上链接
5. 可补充任何有助于理解的辅助材料（示意图、截图等）

提交功能请求请点击[此处](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feature_request.md&title=)。

#### 2.3 设计反馈

关于库设计的反馈（无论正面还是负面）能极大帮助核心维护者打造更友好的库。要了解当前设计理念，请参阅[此文档](https://huggingface.co/docs/diffusers/conceptual/philosophy)如果您认为某个设计选择与当前理念不符，请说明原因及改进建议。如果某个设计选择因过度遵循理念而限制了使用场景，也请解释原因并提出调整方案。  
若某个设计对您特别实用，请同样留下备注——这对未来的设计决策极具参考价值。

您可通过[此链接](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feedback.md&title=)提交设计反馈。

#### 2.4 技术问题

技术问题主要涉及库代码的实现逻辑或特定功能模块的作用。提问时请务必：  
- 附上相关代码链接  
- 详细说明难以理解的具体原因  

技术问题提交入口：[点击此处](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=bug&template=bug-report.yml)

#### 2.5 新模型/调度器/pipeline提案

若diffusion模型社区发布了您希望集成到Diffusers库的新模型、pipeline或调度器，请提供以下信息：  
* 简要说明并附论文或发布链接  
* 开源实现链接（如有）  
* 模型权重下载链接（如已公开）  

若您愿意参与开发，请告知我们以便指导。另请尝试通过GitHub账号标记原始组件作者。  

提案提交地址：[新建请求](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=New+model%2Fpipeline%2Fscheduler&template=new-model-addition.yml)

### 3. 解答GitHub问题

回答GitHub问题可能需要Diffusers的技术知识，但我们鼓励所有人尝试参与——即使您对答案不完全正确。高质量回答的建议：  
- 保持简洁精炼  
- 严格聚焦问题本身  
- 提供代码/论文等佐证材料  
- 优先用代码说话：若代码片段能解决问题，请提供完整可复现代码  

许多问题可能存在离题、重复或无关情况。您可以通过以下方式协助维护者：  
- 引导提问者精确描述问题  
- 标记重复issue并附原链接  
- 推荐用户至[论坛](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers/63)或[Discord](https://discord.gg/G7tWnz98XR)  

在确认提交的Bug报告正确且需要修改源代码后，请继续阅读以下章节内容。

以下所有贡献都需要提交PR（拉取请求）。具体操作步骤详见[如何提交PR](#how-to-open-a-pr)章节。

### 4. 修复"Good first issue"类问题

标有[Good first issue](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)标签的问题通常已说明解决方案建议，便于修复。若该问题尚未关闭且您想尝试解决，只需留言"我想尝试解决这个问题"。通常有三种情况：
- a.) 问题描述已提出解决方案。若您认可该方案，可直接提交PR或草稿PR进行修复
- b.) 问题描述未提出解决方案。您可询问修复建议，Diffusers团队会尽快回复。若有成熟解决方案，也可直接提交PR
- c.) 已有PR但问题未关闭。若原PR停滞，可新开PR并关联原PR（开源社区常见现象）。若PR仍活跃，您可通过建议、审查或协作等方式帮助原作者

### 5. 文档贡献

优秀库**必然**拥有优秀文档！官方文档是新用户的首要接触点，因此文档贡献具有**极高价值**。贡献形式包括：
- 修正拼写/语法错误
- 修复文档字符串格式错误（如显示异常或链接失效）
- 修正文档字符串中张量的形状/维度描述
- 优化晦涩或错误的说明
- 更新过时代码示例
- 文档翻译

[官方文档页面](https://huggingface.co/docs/diffusers/index)所有内容均属可修改范围，对应[文档源文件](https://github.com/huggingface/diffusers/tree/main/docs/source)可进行编辑。修改前请查阅[验证说明](https://github.com/huggingface/diffusers/tree/main/docs)。

### 6. 贡献社区流程

> [!TIP]
> 阅读[社区流程](../using-diffusers/custom_pipeline_overview#community-pipelines)指南了解GitHub与Hugging Face Hub社区流程的区别。若想了解我们设立社区流程的原因，请查看GitHub Issue [#841](https://github.com/huggingface/diffusers/issues/841)（简而言之，我们无法维护diffusion模型所有可能的推理使用方式，但也不希望限制社区构建这些流程）。

贡献社区流程是向社区分享创意与成果的绝佳方式。您可以在[`DiffusionPipeline`]基础上构建流程，任何人都能通过设置`custom_pipeline`参数加载使用。本节将指导您创建一个简单的"单步"流程——UNet仅执行单次前向传播并调用调度器一次。

1. 为社区流程创建one_step_unet.py文件。只要用户已安装相关包，该文件可包含任意所需包。确保仅有一个继承自[`DiffusionPipeline`]的流程类，用于从Hub加载模型权重和调度器配置。在`__init__`函数中添加UNet和调度器。

    同时添加`register_modules`函数，确保您的流程及其组件可通过[`~DiffusionPipeline.save_pretrained`]保存。

```py
from diffusers import DiffusionPipeline
import torch

class UnetSchedulerOneForwardPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)
```

2. 在前向传播中（建议定义为`__call__`），可添加任意功能。对于"单步"流程，创建随机图像并通过设置`timestep=1`调用UNet和调度器一次。

```py
  from diffusers import DiffusionPipeline
  import torch

  class UnetSchedulerOneForwardPipeline(DiffusionPipeline):
      def __init__(self, unet, scheduler):
          super().__init__()

          self.register_modules(unet=unet, scheduler=scheduler)

      def __call__(self):
          image = torch.randn(
              (1, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
          )
          timestep = 1

          model_output = self.unet(image, timestep).sample
          scheduler_output = self.scheduler.step(model_output, timestep, image).prev_sample

          return scheduler_output
```

现在您可以通过传入UNet和调度器来运行流程，若流程结构相同也可加载预训练权重。

```python
from diffusers import DDPMScheduler, UNet2DModel

scheduler = DDPMScheduler()
unet = UNet2DModel()

pipeline = UnetSchedulerOneForwardPipeline(unet=unet, scheduler=scheduler)
output = pipeline()
# 加载预训练权重
pipeline = UnetSchedulerOneForwardPipeline.from_pretrained("google/ddpm-cifar10-32", use_safetensors=True)
output = pipeline()
```

您可以选择将pipeline作为GitHub社区pipeline或Hub社区pipeline进行分享。

<hfoptions id="pipeline类型">
<hfoption id="GitHub pipeline">

通过向Diffusers[代码库](https://github.com/huggingface/diffusers)提交拉取请求来分享GitHub pipeline，将one_step_unet.py文件添加到[examples/community](https://github.com/huggingface/diffusers/tree/main/examples/community)子文件夹中。

</hfoption>
<hfoption id="Hub pipeline">

通过在Hub上创建模型仓库并上传one_step_unet.py文件来分享Hub pipeline。

</hfoption>
</hfoptions>

### 7. 贡献训练示例

Diffusers训练示例是位于[examples](https://github.com/huggingface/diffusers/tree/main/examples)目录下的训练脚本集合。

我们支持两种类型的训练示例：

- 官方训练示例
- 研究型训练示例

研究型训练示例位于[examples/research_projects](https://github.com/huggingface/diffusers/tree/main/examples/research_projects)，而官方训练示例包含[examples](https://github.com/huggingface/diffusers/tree/main/examples)目录下除`research_projects`和`community`外的所有文件夹。
官方训练示例由Diffusers核心维护者维护，研究型训练示例则由社区维护。
这与[6. 贡献社区pipeline](#6-contribute-a-community-pipeline)中关于官方pipeline与社区pipeline的原因相同：核心维护者不可能维护diffusion模型的所有可能训练方法。
如果Diffusers核心维护者和社区认为某种训练范式过于实验性或不够普及，相应训练代码应放入`research_projects`文件夹并由作者维护。

官方训练和研究型示例都包含一个目录，其中含有一个或多个训练脚本、`requirements.txt`文件和`README.md`文件。用户使用时需要先克隆代码库：

```bash
git clone https://github.com/huggingface/diffusers
```

并安装训练所需的所有额外依赖：

```bash
cd diffusers
pip install -r examples/<your-example-folder>/requirements.txt
```

因此添加示例时，`requirements.txt`文件应定义训练示例所需的所有pip依赖项，安装完成后用户即可运行示例训练脚本。可参考[DreamBooth的requirements.txt文件](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/requirements.txt)。
- 运行示例所需的所有代码应集中在单个Python文件中  
- 用户应能通过命令行`python <your-example>.py --args`直接运行示例  
- **示例**应保持简洁，主要展示如何使用Diffusers进行训练。示例脚本的目的**不是**创建最先进的diffusion模型，而是复现已知训练方案，避免添加过多自定义逻辑。因此，这些示例也力求成为优质的教学材料。

提交示例时，强烈建议参考现有示例（如[dreambooth](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py)）来了解规范格式。  
我们强烈建议贡献者使用[Accelerate库](https://github.com/huggingface/accelerate)，因其与Diffusers深度集成。  
当示例脚本完成后，请确保添加详细的`README.md`说明使用方法，包括：  
- 运行示例的具体命令（示例参见[此处](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth#running-locally-with-pytorch)）  
- 训练结果链接（日志/模型等），展示用户可预期的效果（示例参见[此处](https://api.wandb.ai/report/patrickvonplaten/xm6cd5q5)）  
- 若添加非官方/研究性训练示例，**必须注明**维护者信息（含Git账号），格式参照[此处](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/intel_opts#diffusers-examples-with-intel-optimizations)  

贡献官方训练示例时，还需在对应目录添加测试文件（如[examples/dreambooth/test_dreambooth.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/test_dreambooth.py)），非官方示例无需此步骤。

### 8. 处理"Good second issue"类问题

标有[Good second issue](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22Good+second+issue%22)标签的问题通常比[Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)更复杂。  
这类问题的描述通常不会提供详细解决指引，需要贡献者对库有较深理解。  
若您想解决此类问题，可直接提交PR并关联对应issue。若已有未合并的PR，请分析原因后提交改进版。需注意，Good second issue类PR的合并难度通常高于good first issues。在需要帮助的时候请不要犹豫，大胆的向核心维护者询问。

### 9. 添加管道、模型和调度器

管道（pipelines）、模型（models）和调度器（schedulers）是Diffusers库中最重要的组成部分。它们提供了对最先进diffusion技术的便捷访问，使得社区能够构建强大的生成式AI应用。

通过添加新的模型、管道或调度器，您可能为依赖Diffusers的任何用户界面开启全新的强大用例，这对整个生成式AI生态系统具有巨大价值。

Diffusers针对这三类组件都有一些开放的功能请求——如果您还不确定要添加哪个具体组件，可以浏览以下链接：
- [模型或管道](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
- [调度器](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

在添加任何组件之前，强烈建议您阅读[设计哲学指南](philosophy)，以更好地理解这三类组件的设计理念。请注意，如果添加的模型、调度器或管道与我们的设计理念存在严重分歧，我们将无法合并，因为这会导致API不一致。如果您从根本上不同意某个设计选择，请改为提交[反馈问题](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feedback.md&title=)，以便讨论是否应该更改库中的特定设计模式/选择，以及是否更新我们的设计哲学。保持库内的一致性对我们非常重要。

请确保在PR中添加原始代码库/论文的链接，并最好直接在PR中@原始作者，以便他们可以跟踪进展并在有疑问时提供帮助。

如果您在PR过程中遇到不确定或卡住的情况，请随时留言请求初步审查或帮助。

#### 复制机制（Copied from）

在添加任何管道、模型或调度器代码时，理解`# Copied from`机制是独特且重要的。您会在整个Diffusers代码库中看到这种机制，我们使用它的原因是为了保持代码库易于理解和维护。用`# Copied from`机制标记代码会强制标记的代码与复制来源的代码完全相同。这使得每当您运行`make fix-copies`时，可以轻松更新并将更改传播到多个文件。

例如，在下面的代码示例中，[`~diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput`]是原始代码，而`AltDiffusionPipelineOutput`使用`# Copied from`机制来复制它。唯一的区别是将类前缀从`Stable`改为`Alt`。

```py
# 从 diffusers.pipelines.stable_diffusion.pipeline_output.StableDiffusionPipelineOutput 复制并将 Stable 替换为 Alt
class AltDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Alt Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        nsfw_content_detected (`List[bool]`)
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """
```

要了解更多信息，请阅读[~不要~重复自己*](https://huggingface.co/blog/transformers-design-philosophy#4-machine-learning-models-are-static)博客文章的相应部分。

## 如何撰写优质问题

**问题描述越清晰，被快速解决的可能性就越高。**

1. 确保使用了正确的issue模板。您可以选择*错误报告*、*功能请求*、*API设计反馈*、*新模型/流水线/调度器添加*、*论坛*或空白issue。在[新建issue](https://github.com/huggingface/diffusers/issues/new/choose)时务必选择正确的模板。
2. **精确描述**：为issue起一个恰当的标题。尽量用最简练的语言描述问题。提交issue时越精确，理解问题和潜在解决方案所需的时间就越少。确保一个issue只针对一个问题，不要将多个问题放在同一个issue中。如果发现多个问题，请分别创建多个issue。如果是错误报告，请尽可能精确描述错误类型——不应只写"diffusers出错"。
3. **可复现性**：无法复现的代码片段 == 无法解决问题。如果遇到错误，维护人员必须能够**复现**它。确保包含一个可以复制粘贴到Python解释器中复现问题的代码片段。确保您的代码片段是可运行的，即没有缺少导入或图像链接等问题。issue应包含错误信息和可直接复制粘贴以复现相同错误的代码片段。如果issue涉及本地模型权重或无法被读者访问的本地数据，则问题无法解决。如果无法共享数据或模型，请尝试创建虚拟模型或虚拟数据。
4. **最小化原则**：通过尽可能简洁的描述帮助读者快速理解问题。删除所有与问题无关的代码/信息。如果发现错误，请创建最简单的代码示例来演示问题，不要一发现错误就把整个工作流程都转储到issue中。例如，如果在训练模型时某个阶段出现错误或训练过程中遇到问题时，应首先尝试理解训练代码的哪部分导致了错误，并用少量代码尝试复现。建议使用模拟数据替代完整数据集进行测试。
5. 添加引用链接。当提及特定命名、方法或模型时，请务必提供引用链接以便读者理解。若涉及具体PR或issue，请确保添加对应链接。不要假设读者了解你所指内容。issue中引用链接越丰富越好。
6. 规范格式。请确保规范格式化issue内容：Python代码使用代码语法块，错误信息使用标准代码语法。详见[GitHub官方格式文档](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)。
7. 请将issue视为百科全书的精美词条，而非待解决的工单。每个规范撰写的issue不仅是向维护者有效传递问题的方式，更是帮助社区深入理解库特性的公共知识贡献。

## 优质PR编写规范

1. 保持风格统一。理解现有设计模式和语法规范，确保新增代码与代码库现有结构无缝衔接。显著偏离现有设计模式或用户界面的PR将不予合并。
2. 聚焦单一问题。每个PR应当只解决一个明确问题，避免"顺手修复其他问题"的陷阱。包含多个无关修改的PR会极大增加审查难度。
3. 如适用，建议添加代码片段演示新增功能的使用方法。
4. PR标题应准确概括其核心贡献。
5. 若PR针对某个issue，请在描述中注明issue编号以建立关联（也让关注该issue的用户知晓有人正在处理）；
6. 进行中的PR请在标题添加`[WIP]`前缀。这既能避免重复劳动，也可与待合并PR明确区分；
7. 文本表述与格式要求请参照[优质issue编写规范](#how-to-write-a-good-issue)；
8. 确保现有测试用例全部通过；
9. 必须添加高覆盖率测试。未经充分测试的代码不予合并。
- 若新增`@slow`测试，请使用`RUN_SLOW=1 python -m pytest tests/test_my_new_model.py`确保通过。
CircleCI不执行慢速测试，但GitHub Actions会每日夜间运行！
10. 所有公开方法必须包含格式规范、兼容markdown的说明文档。可参考[`pipeline_latent_diffusion.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py) 
11. 由于代码库快速增长，必须确保不会添加明显增加仓库体积的文件（如图片、视频等非文本文件）。建议优先使用托管在hf.co的`dataset`（例如[`hf-internal-testing`](https://huggingface.co/hf-internal-testing)或[huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images)）存放这类文件。若为外部贡献，可将图片添加到PR中并请Hugging Face成员将其迁移至该数据集。

## 提交PR流程

编写代码前，强烈建议先搜索现有PR或issue，确认没有重复工作。如有疑问，建议先创建issue获取反馈。

贡献至🧨 Diffusers需要基本的`git`技能。虽然`git`学习曲线较高，但其拥有最完善的手册。在终端输入`git --help`即可查阅，或参考书籍[Pro Git](https://git-scm.com/book/en/v2)。

请按以下步骤操作（[支持的Python版本](https://github.com/huggingface/diffusers/blob/83bc6c94eaeb6f7704a2a428931cf2d9ad973ae9/setup.py#L270)）：

1. 在[仓库页面](https://github.com/huggingface/diffusers)点击"Fork"按钮创建代码副本至您的GitHub账户

2. 克隆fork到本地，并添加主仓库为远程源：
 ```bash
 $ git clone git@github.com:<您的GitHub账号>/diffusers.git
 $ cd diffusers
 $ git remote add upstream https://github.com/huggingface/diffusers.git
 ```

3. 创建新分支进行开发：
 ```bash
 $ git checkout -b 您的开发分支名称
 ```
**禁止**直接在`main`分支上修改

4. 在虚拟环境中运行以下命令配置开发环境：
 ```bash
 $ pip install -e ".[dev]"
 ```
若已克隆仓库，可能需要先执行`git pull`获取最新代码

5. 在您的分支上开发功能

开发过程中应确保测试通过。可运行受影响测试：
 ```bash
 $ pytest tests/<待测文件>.py
 ```
执行测试前请安装测试依赖：
 ```bash
 $ pip install -e ".[test]"
 ```
也可运行完整测试套件（需高性能机器）：
 ```bash
 $ make test
 ```

🧨 Diffusers使用`black`和`isort`工具保持代码风格统一。修改后请执行自动化格式校正与代码验证，以下内容无法通过以下命令一次性自动化完成：

```bash
$ make style
```

🧨 Diffusers 还使用 `ruff` 和一些自定义脚本来检查代码错误。虽然质量控制流程会在 CI 中运行，但您也可以通过以下命令手动执行相同的检查：

```bash
$ make quality
```

当您对修改满意后，使用 `git add` 添加更改的文件，并通过 `git commit` 在本地记录这些更改：

```bash
$ git add modified_file.py
$ git commit -m "关于您所做更改的描述性信息。"
```

定期将您的代码副本与原始仓库同步是一个好习惯。这样可以快速适应上游变更：

```bash
$ git pull upstream main
```

使用以下命令将更改推送到您的账户：

```bash
$ git push -u origin 此处替换为您的描述性分支名称
```

6. 确认无误后，请访问您 GitHub 账户中的派生仓库页面。点击「Pull request」将您的更改提交给项目维护者审核。

7. 如果维护者要求修改，这很正常——核心贡献者也会遇到这种情况！为了让所有人能在 Pull request 中看到变更，请在本地分支继续工作并将修改推送到您的派生仓库，这些变更会自动出现在 Pull request 中。

### 测试

我们提供了全面的测试套件来验证库行为和多个示例。库测试位于 [tests 文件夹](https://github.com/huggingface/diffusers/tree/main/tests)。

我们推荐使用 `pytest` 和 `pytest-xdist`，因为它们速度更快。在仓库根目录下运行以下命令执行库测试：

```bash
$ python -m pytest -n auto --dist=loadfile -s -v ./tests/
```

实际上，这就是 `make test` 的实现方式！

您可以指定更小的测试范围来仅验证您正在开发的功能。

默认情况下会跳过耗时测试。设置 `RUN_SLOW` 环境变量为 `yes` 可运行这些测试。注意：这将下载数十 GB 的模型文件——请确保您有足够的磁盘空间、良好的网络连接或充足的耐心！

```bash
$ RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./tests/
```

我们也完全支持 `unittest`，运行方式如下：

```bash
$ python -m unittest discover -s tests -t . -v
$ python -m unittest discover -s examples -t examples -v
```

### 将派生仓库的 main 分支与上游（HuggingFace）main 分支同步

为避免向上游仓库发送引用通知（这会给相关 PR 添加注释并向开发者发送不必要的通知），在同步派生仓库的 main 分支时，请遵循以下步骤：
1. 尽可能避免通过派生仓库的分支和 PR 来同步上游，而是直接合并到派生仓库的 main 分支
2. 如果必须使用 PR，请在检出分支后执行以下操作：
```bash
$ git checkout -b 您的同步分支名称
$ git pull --squash --no-commit upstream main
$ git commit -m '提交信息（不要包含 GitHub 引用）'
$ git push --set-upstream origin 您的分支名称
```

### 风格指南

对于文档字符串，🧨 Diffusers 遵循 [Google 风格指南](https://google.github.io/styleguide/pyguide.html)。
