<!---
Copyright 2022 - The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
</p>
<p align="center">
    <a href="https://github.com/huggingface/diffusers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue"></a>
    <a href="https://github.com/huggingface/diffusers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/diffusers.svg"></a>
    <a href="https://pepy.tech/project/diffusers"><img alt="GitHub release" src="https://static.pepy.tech/badge/diffusers/month"></a>
    <a href="CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg"></a>
    <a href="https://twitter.com/diffuserslib"><img alt="X account" src="https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

🤗 **Diffusers** 是当前最前沿预训练扩散模型（Diffusion Models）的工业级标准库，广泛用于生成高质量图像、音频乃至分子的 3D 空间结构。无论您需要极简的高性能开箱即用推理方案，还是正在从零研究训练自有的扩散模型，🤗 Diffusers 提供的模块化工具箱都能提供全流程支持。

本代码库的核心设计哲学：
- [**易用性优于极致性能** (Usability over performance)](https://huggingface.co/docs/diffusers/conceptual/philosophy#usability-over-performance)
- [**直观简单优于隐晦捷径** (Simple over easy)](https://huggingface.co/docs/diffusers/conceptual/philosophy#simple-over-easy)
- [**灵活可定制优于过度抽象** (Customizability over abstractions)](https://huggingface.co/docs/diffusers/conceptual/philosophy#tweakable-contributorfriendly-over-abstraction)

🤗 Diffusers 提供了三大核心组件：

- **前沿扩散管线 ([Diffusion Pipelines](https://huggingface.co/docs/diffusers/api/pipelines/overview))**：只需数行代码即可运行各种 SOTA 扩散模型推理；
- **可插拔噪声调度器 ([Schedulers](https://huggingface.co/docs/diffusers/api/schedulers/overview))**：支持在生成速度与生成质量之间灵活权衡与切换；
- **基础网络模块 ([Models](https://huggingface.co/docs/diffusers/api/models/overview))**：提供 UNet、Transformer Backbone 等丰富的基础预训练网络模块，可与调度器自由组合构建专属的端到端扩散系统。

---

## 📦 安装说明 (Installation)

推荐在独立的 Python 虚拟环境中使用 PyPI 或 Conda 安装 🤗 Diffusers。关于 [PyTorch](https://pytorch.org/get-started/locally/) 的平台特定安装指南，请参考 PyTorch 官方文档。

### PyTorch 环境安装

使用 `pip` 安装（官方标准包）：

```bash
pip install --upgrade diffusers[torch]
```

使用 `conda` 安装（由开源社区维护）：

```sh
conda install -c conda-forge diffusers
```

### Apple Silicon (M1/M2/M3/M4) MPS 加速支持

针对搭载 Apple Silicon 芯片的 Mac 设备，请参阅 [在 Apple Silicon 上运行 Stable Diffusion 指南](https://huggingface.co/docs/diffusers/optimization/mps)。

---

## 🚀 快速上手 (Quickstart)

使用 🤗 Diffusers 生成内容极其简单。要通过文本生成图像，只需使用 `from_pretrained` 方法加载任意预训练扩散模型（欢迎浏览 [Hugging Face Hub](https://huggingface.co/models?library=diffusers&sort=downloads) 探索 30,000+ 个模型权重）：

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

您也可以直接深入调用底层的模型与噪声调度器工具箱，自主组装完整的扩散推理系统：

```python
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import torch

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
scheduler.set_timesteps(50)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
input = noise

for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(input, t).sample
        prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = prev_noisy_sample

image = (input / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
image = Image.fromarray((image * 255).round().astype("uint8"))
image
```

欢迎查阅 [官方快速入门指南 (Quickstart Tour)](https://huggingface.co/docs/diffusers/quicktour)，开启您的扩散模型创作之旅！

---

## 🧭 文档结构导航 (How to navigate the documentation)

| 核心文档专区 | 您能学到什么？ |
| :--- | :--- |
| [**快速上手 (Quickstart)**](https://huggingface.co/docs/diffusers/quicktour) | 快速掌握管线加载、结果生成以及常见推理加速优化的极简速成课程。 |
| [**加载指南 (Loading)**](https://huggingface.co/docs/diffusers/using-diffusers/loading) | 详尽说明如何加载并配置库中所有核心组件（Pipelines、Models 和 Schedulers），以及如何切换不同调度器。 |
| [**模块化 Diffusers (Modular Diffusers)**](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/overview) | 基于高度解耦的模块化管线组件，灵活搭建专属扩散系统。 |
| [**显存与推理优化 (Optimization)**](https://huggingface.co/docs/diffusers/optimization/fp16) | 深入讲解如何优化扩散模型，以实现更快推理速度和极低显存占用（FP16/BF16/FlashAttention/CPU Offload 等）。 |
| [**模型微调与训练 (Training)**](https://huggingface.co/docs/diffusers/training/overview) | 针对多样化任务（Text-to-Image、DreamBooth、LoRA、ControlNet 等）与不同训练技巧的全方位教程。 |

---

## 🤝 参与贡献 (Contribution)

我们 ❤️ 来自开源社区的每一份贡献！
如果您希望为本库贡献代码或文档，请查阅我们的 [贡献者指南 (Contribution Guide)](https://huggingface.co/docs/diffusers/main/en/conceptual/contribution)。

如果您正在使用 AI 智能体（AI Coding Agent）进行协作，请让其首先参考项目在 [`.ai/`](https://github.com/huggingface/diffusers/tree/main/.ai) 目录中定义的工程规范（支持通过 `claude plugin marketplace add huggingface/diffusers` 添加插件，或通过 `diffusers-cli skills add <name>` 单独安装技能）—— 详见 [与 AI 智能体协同编码指南](https://huggingface.co/docs/diffusers/main/en/conceptual/contribution#coding-with-ai-agents)。

您可以浏览 [GitHub Issues](https://github.com/huggingface/diffusers/issues) 寻找感兴趣的任务：
- 查看 [新手友好任务 (Good first issues)](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
- 查看 [新模型与新管线 (New model/pipeline)](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)，贡献前沿扩散架构
- 查看 [新噪声调度器 (New scheduler)](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

也欢迎加入我们的官方 Discord 公共频道：<a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>。我们在那里讨论前沿扩散趋势、交流技术方案或共同交流探索 ☕。

---

## 🎨 热门任务与核心官方管线全览 (Popular Tasks & Pipelines)

<table>
  <tr>
    <th>任务类别</th>
    <th>推理管线 (Pipeline)</th>
    <th>🤗 Hub 推荐模型</th>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>无条件图像生成 (Unconditional Image Generation)</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/ddpm"> DDPM </a></td>
    <td><a href="https://huggingface.co/google/ddpm-ema-church-256"> google/ddpm-ema-church-256 </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>文本生成图像 (Text-to-Image)</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img">Stable Diffusion Text-to-Image</a></td>
      <td><a href="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5"> stable-diffusion-v1-5/stable-diffusion-v1-5 </a></td>
  </tr>
  <tr>
    <td>文本生成图像 (Text-to-Image)</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_unclip">unCLIP</a></td>
      <td><a href="https://huggingface.co/kakaobrain/karlo-v1-alpha"> kakaobrain/karlo-v1-alpha </a></td>
  </tr>
  <tr>
    <td>文本生成图像 (Text-to-Image)</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if">DeepFloyd IF</a></td>
      <td><a href="https://huggingface.co/DeepFloyd/IF-I-XL-v1.0"> DeepFloyd/IF-I-XL-v1.0 </a></td>
  </tr>
  <tr>
    <td>文本生成图像 (Text-to-Image)</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/kandinsky">Kandinsky</a></td>
      <td><a href="https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder"> kandinsky-community/kandinsky-2-2-decoder </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>文本引导图生图 (Text-guided Image-to-Image)</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/controlnet">ControlNet</a></td>
      <td><a href="https://huggingface.co/lllyasviel/sd-controlnet-canny"> lllyasviel/sd-controlnet-canny </a></td>
  </tr>
  <tr>
    <td>文本引导图生图 (Text-guided Image-to-Image)</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/pix2pix">InstructPix2Pix</a></td>
      <td><a href="https://huggingface.co/timbrooks/instruct-pix2pix"> timbrooks/instruct-pix2pix </a></td>
  </tr>
  <tr>
    <td>文本引导图生图 (Text-guided Image-to-Image)</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img">Stable Diffusion Image-to-Image</a></td>
      <td><a href="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5"> stable-diffusion-v1-5/stable-diffusion-v1-5 </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>文本引导局部重绘 (Text-guided Image Inpainting)</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint">Stable Diffusion Inpainting</a></td>
      <td><a href="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting"> stable-diffusion-v1-5/stable-diffusion-inpainting </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>图像多变体生成 (Image Variation)</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation">Stable Diffusion Image Variation</a></td>
      <td><a href="https://huggingface.co/lambdalabs/sd-image-variations-diffusers"> lambdalabs/sd-image-variations-diffusers </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>超分辨率重建 (Super Resolution)</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale">Stable Diffusion Upscale</a></td>
      <td><a href="https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler"> stabilityai/stable-diffusion-x4-upscaler </a></td>
  </tr>
  <tr>
    <td>超分辨率重建 (Super Resolution)</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale">Stable Diffusion Latent Upscale</a></td>
      <td><a href="https://huggingface.co/stabilityai/sd-x2-latent-upscaler"> stabilityai/sd-x2-latent-upscaler </a></td>
  </tr>
</table>

---

## 🌐 基于 🧨 Diffusers 构建的知名代表性项目

- [Microsoft TaskMatrix](https://github.com/microsoft/TaskMatrix)
- [InvokeAI](https://github.com/invoke-ai/InvokeAI)
- [InstantID](https://github.com/InstantID/InstantID)
- [Apple ML Stable Diffusion](https://github.com/apple/ml-stable-diffusion)
- [Lama Cleaner (Sanster)](https://github.com/Sanster/lama-cleaner)
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [Stable-Dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
- [DeepFloyd IF](https://github.com/deep-floyd/IF)
- [BentoML](https://github.com/bentoml/BentoML)
- [Kohya_ss](https://github.com/bmaltais/kohya_ss)
- 以及 GitHub 上超过 **14,000+** 个优秀的开源代码库 💪

衷心感谢大家的信赖与使用 ❤️。

---

## 🏆 致谢 (Credits)

本代码库凝结了众多学者与开源作者的早期研究成果与灵感。特别鸣谢以下项目对我们设计演进的巨大启发：

- @CompVis 团队的 [Latent Diffusion Models 代码库](https://github.com/CompVis/latent-diffusion)
- @hojonathanho 的 [原始 DDPM 实现](https://github.com/hojonathanho/diffusion)，以及 @pesser 极其优雅的 [PyTorch 移植版](https://github.com/pesser/pytorch_diffusion)
- @ermongroup 的 [DDIM 实现](https://github.com/ermongroup/ddim)
- @yang-song 的 [Score-VE 与 Score-VP 实现](https://github.com/yang-song/score_sde_pytorch)

同时感谢 @heejkoo 整理的详实论文与扩散模型生态综述项目 [Awesome-Diffusion-Models](https://github.com/heejkoo/Awesome-Diffusion-Models)，以及 @crowsonkb 和 @rromb 提供的诸多建设性讨论与深刻洞见。

---

## 📑 论文与学术引用 (Citation)

如果您在学术成果或生产实践中使用了 🤗 Diffusers，请引用以下 BibTeX 条目：

```bibtex
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Dhruv Nair and Sayak Paul and William Berman and Yiyi Xu and Steven Liu and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年9月5日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
