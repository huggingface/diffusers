# Diffusers 简化版 - 学习专用

这是一个精简版的 🤗 Diffusers 代码库，专门用于学习和理解扩散模型的核心概念。

## 📝 简化说明

本代码库已经移除了所有高级功能，只保留了最基础、最经典的组件，方便初学者理解扩散模型的核心原理。

### ✅ 保留的核心功能

#### 模型 (Models)
- **UNet2DConditionModel** - Stable Diffusion 使用的条件 UNet
- **UNet2DModel** - 基础的 2D UNet
- **AutoencoderKL** - 变分自编码器 (VAE)
- **Transformer2DModel** - 基础 Transformer 模型
- **VQModel** - Vector Quantized VAE

#### 调度器 (Schedulers)
- **DDPMScheduler** - 去噪扩散概率模型 (基础)
- **DDIMScheduler** - 去噪扩散隐式模型 (快速采样)
- **EulerDiscreteScheduler** - Euler 方法 (常用)
- **PNDMScheduler** - 伪数值方法 (经典)

#### 管道 (Pipelines)
- **Stable Diffusion** 系列
  - StableDiffusionPipeline (文生图)
  - StableDiffusionImg2ImgPipeline (图生图)
  - StableDiffusionInpaintPipeline (图像修复)
- **DDPM** - DDPMPipeline (基础扩散模型)
- **Latent Diffusion** - 潜在扩散模型

### ❌ 已删除的高级功能

为了简化学习曲线，以下功能已被移除：

- **LoRA** - 低秩适配微调
- **量化** (Quantization) - 8-bit/4-bit 量化
- **ControlNet** - 精细控制生成
- **IP-Adapter** - 图像提示适配器
- **PEFT** - 参数高效微调
- **Textual Inversion** - 文本反演
- **高级优化** (Hooks, Guiders) - 内存优化、缓存等
- **模块化管道** (Modular Pipelines)
- **实验性功能** (Experimental)
- **90+ 个高级 pipeline** (Flux, CogVideoX, Kandinsky 等)
- **48+ 个高级 scheduler**

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone <your-repo-url>
cd diffusers

# 安装核心依赖
pip install -e .
```

最小依赖：
- `torch`
- `transformers`
- `accelerate`
- `safetensors`
- `Pillow`
- `numpy`

### 基础示例

#### 1. Stable Diffusion 文生图

```python
from diffusers import StableDiffusionPipeline
import torch

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 生成图像
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image.save("astronaut.png")
```

#### 2. DDPM 基础扩散模型

```python
from diffusers import DDPMPipeline

# 加载预训练模型
pipeline = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")

# 生成图像
image = pipeline().images[0]
image.save("generated.png")
```

#### 3. 理解调度器

```python
from diffusers import DDPMScheduler, DDIMScheduler

# DDPM - 需要更多步数，质量更好
ddpm_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

# DDIM - 更快的采样
ddim_scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

# 在 pipeline 中切换调度器
pipe.scheduler = ddim_scheduler
```

## 📚 代码结构

简化后的目录结构：

```
src/diffusers/
├── models/
│   ├── autoencoders/
│   │   ├── autoencoder_kl.py      # VAE 编码器
│   │   └── vq_model.py             # VQ-VAE
│   ├── transformers/
│   │   └── transformer_2d.py       # 基础 Transformer
│   ├── unets/
│   │   ├── unet_2d.py              # 基础 UNet
│   │   ├── unet_2d_blocks.py       # UNet 构建块
│   │   └── unet_2d_condition.py    # 条件 UNet
│   ├── attention.py                # 注意力机制
│   ├── embeddings.py               # 嵌入层
│   ├── resnet.py                   # ResNet 块
│   └── normalization.py            # 归一化层
│
├── schedulers/
│   ├── scheduling_ddpm.py          # DDPM 调度器
│   ├── scheduling_ddim.py          # DDIM 调度器
│   ├── scheduling_euler_discrete.py # Euler 调度器
│   └── scheduling_pndm.py          # PNDM 调度器
│
├── pipelines/
│   ├── stable_diffusion/           # Stable Diffusion 管道
│   ├── ddpm/                       # DDPM 管道
│   └── latent_diffusion/           # 潜在扩散管道
│
├── configuration_utils.py          # 配置管理
├── image_processor.py              # 图像处理
└── utils/                          # 工具函数
```

## 🎓 学习路径

推荐按以下顺序学习：

1. **理解 DDPM** (`pipelines/ddpm/`)
   - 最基础的扩散模型
   - 理解前向扩散和反向去噪过程

2. **学习调度器** (`schedulers/`)
   - 比较 DDPM vs DDIM
   - 理解采样步骤和噪声调度

3. **研究 UNet** (`models/unets/`)
   - 扩散模型的核心网络架构
   - 理解时间步嵌入和条件注入

4. **探索 VAE** (`models/autoencoders/`)
   - 理解潜在空间压缩
   - Stable Diffusion 如何使用 VAE

5. **分析 Stable Diffusion** (`pipelines/stable_diffusion/`)
   - 完整的文生图流程
   - 文本编码、潜在扩散、VAE 解码

## 🔍 核心概念

### 扩散模型原理

1. **前向扩散** (Forward Diffusion)
   ```
   干净图像 → 逐步添加噪声 → 纯噪声
   ```

2. **反向去噪** (Reverse Denoising)
   ```
   纯噪声 → UNet 预测并移除噪声 → 干净图像
   ```

### Stable Diffusion 流程

```
文本提示 → 文本编码器 (CLIP) → 文本嵌入
                ↓
随机噪声 → UNet 去噪 (潜在空间) → 潜在表示
                ↓
        VAE 解码器 → 最终图像
```

## 📖 推荐阅读

- [DDPM 论文](https://arxiv.org/abs/2006.11239) - Denoising Diffusion Probabilistic Models
- [DDIM 论文](https://arxiv.org/abs/2010.02502) - Denoising Diffusion Implicit Models
- [Stable Diffusion 论文](https://arxiv.org/abs/2112.10752) - High-Resolution Image Synthesis with Latent Diffusion Models

## ⚠️ 注意事项

- 这是一个**学习专用**的简化版本，不适合生产环境
- 缺少高级功能如 LoRA、ControlNet 等
- 如需完整功能，请使用官方版本：https://github.com/huggingface/diffusers

## 📄 许可证

遵循原始 🤗 Diffusers 项目的 Apache 2.0 许可证。

## 🙏 致谢

本简化版基于 [HuggingFace Diffusers](https://github.com/huggingface/diffusers) 项目。

---

**简化版本说明**：此代码库从完整的 Diffusers 库中精简而来，专注于核心扩散模型概念，适合学习和研究使用。
