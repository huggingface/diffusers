<!-- Copyright 2026 The AnyFlow Team, NVIDIA Corp., and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
the License for the specific language governing permissions and limitations under the License.
-->

# AnyFlow

[AnyFlow](https://huggingface.co/papers/2605.13724) 是一个视频扩散**蒸馏**框架，把预训练的 Wan2.1 教师
模型蒸馏成在标准 Euler 采样下支持*任意步数 (any-step)* 的学生模型。同一个蒸馏出来的 checkpoint 可以
在 1、2、4、8、16... NFE 下推理，**质量随步数单调提升** —— 这一点和 consistency models 不同，后者
NFE 增加反而经常掉点。

核心思路是学习 **flow map** $\Phi_{r\leftarrow t}: \mathbf{z}_t \to \mathbf{z}_r$（任意 $1 \ge t \ge r \ge 0$），
而不是 consistency models 学的固定端点映射 $\mathbf{z}_t \to \mathbf{z}_0$。Flow map 的可组合性消除了
采样步之间的 re-noising；on-policy 蒸馏阶段额外用 **DMD 反向散度监督** + **Flow-Map backward simulation**
（3 段 shortcut）补上 consistency 蒸馏遗留的 exposure-bias 缺口。

AnyFlow 由 Yuchao Gu、Guian Fang 等人在 [NUS ShowLab](https://sites.google.com/view/showlab) 与 NVIDIA 合作完成。原始训练代码在 [`NVlabs/AnyFlow`](https://github.com/NVlabs/AnyFlow)，项目主页是 [nvlabs.github.io/AnyFlow](https://nvlabs.github.io/AnyFlow)。4 个发布 checkpoint 归在 [`nvidia/anyflow`](https://huggingface.co/collections/nvidia/anyflow) Hugging Face collection 里。

本文档梳理实战要点：怎么选 pipeline、怎么用 any-step 采样、怎么把 AnyFlow 嵌进 T2V / I2V / V2V 工作流。

## Bidirectional 还是 Causal —— 怎么选 pipeline

AnyFlow 提供两个 pipeline 形态，scheduler 和蒸馏方法相同，区别在于**怎么对帧采样**：

- [`AnyFlowPipeline`](../api/pipelines/anyflow#anyflowpipeline) —— **bidirectional** T2V。一次性对整个
  视频张量去噪，全局自注意力。**纯 prompt 输入、不要流式输出**时选这个。
- [`AnyFlowFARPipeline`](../api/pipelines/anyflow#anyflowfarpipeline) —— **causal (FAR)**。
  按 chunk 分段去噪，块稀疏因果注意力 + 跨 chunk 复用 KV cache。**图生视频 (I2V)**、**视频续写 (V2V)**、
  或任何受益于逐帧自回归采样的场景选这个。同一个模型通过传入 `context_sequence` 来切换三种任务模式。

简化对照表：

| 场景 | Pipeline | 调用方式 |
|------|----------|----------|
| 纯文生视频，固定 NFE 求最大质量 | `AnyFlowPipeline` | `pipe(prompt, ...)` |
| 图生视频（首帧给定） | `AnyFlowFARPipeline` | `pipe(prompt, context_sequence={"raw": <单帧 tensor>}, ...)` |
| 视频续写 / V2V | `AnyFlowFARPipeline` | `pipe(prompt, context_sequence={"raw": <多帧 tensor>}, ...)` |
| 流式 / 渐进式生成 | `AnyFlowFARPipeline` | — |

高分辨率下 bidirectional 单 token 更快；causal 牺牲一点单步速度，换来在所有 latent 帧分配前就能开始
采样的能力，对超长序列尤其有用。

## 加载 checkpoint

NVIDIA 发布了 4 个 AnyFlow checkpoint，pipeline × 规模各一份：

```py
import torch
from diffusers import AnyFlowPipeline, AnyFlowFARPipeline

# Bidirectional, 轻量
pipe = AnyFlowPipeline.from_pretrained(
    "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")

# Bidirectional, 满血
pipe = AnyFlowPipeline.from_pretrained(
    "nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")

# Causal (FAR), 1.3B
pipe = AnyFlowFARPipeline.from_pretrained(
    "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")

# Causal (FAR), 14B
pipe = AnyFlowFARPipeline.from_pretrained(
    "nvidia/AnyFlow-FAR-Wan2.1-14B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")
```

四个 checkpoint 共用同一份 [`FlowMapEulerDiscreteScheduler`](../api/schedulers/flow_map_euler_discrete)，
默认 `shift=5.0`。

## Any-step 采样

AnyFlow 最关键的特性是同一个 checkpoint **不需重新调度**，NFE 越大质量越高。固定 prompt、扫一下步数
就能看出模型怎么在延迟和保真度之间权衡：

```py
import torch
from diffusers import AnyFlowPipeline
from diffusers.utils import export_to_video

pipe = AnyFlowPipeline.from_pretrained(
    "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "森林里一只小熊猫在啃竹子，电影感光照"

for nfe in [1, 2, 4, 8, 16, 32]:
    # 每轮重建 generator —— 这样跨步数对比时唯一变量是 NFE。
    generator = torch.Generator("cuda").manual_seed(0)
    video = pipe(prompt, num_inference_steps=nfe, num_frames=33, generator=generator).frames[0]
    export_to_video(video, f"out_nfe{nfe}.mp4", fps=16)
```

paper 的 Tab 3 / Fig 1 表明：每个 AnyFlow checkpoint 在 4 → 32 NFE 范围 VBench Quality 都单调上升，而
consistency 类基线（rCM、Self-Forcing）在同区间反而掉点。

> [!TIP]
> Classifier-free guidance (CFG) 已经在蒸馏阶段融进权重 (`fuse_guidance_scale = 3.0`)。pipeline 推理
> 时**不会**再跑一次 unconditional 前向 —— guidance 直接由蒸馏后的权重带出。release 出来的 checkpoint
> 都用默认的 `guidance_scale=1.0` 即可。

## 图生视频 与 视频续写

Causal pipeline 用同一个蒸馏模型支持三种任务模式，**通过 `context_sequence` 隐式选择**（dict，含
`"raw"` 视频张量或 `"latent"` 已编码 latent）。Context tensor 的帧数必须满足 `T = 4n + 1`，跟 VAE
时间步长对齐。

> [!IMPORTANT]
> FAR pipeline 是分块 (chunk) rollout，`num_frames` 必须配合 chunk 调度。默认
> `chunk_partition=[1, 3, 3, 3, 3, 3, 3, 2]`（求和 21）对应发布 checkpoint 的标准 `num_frames=81`
> （21 = (81 − 1) // 4 + 1）。改 `num_frames` 时**必须**显式传匹配的 `chunk_partition`，使其求和等于
> `(num_frames - 1) // 4 + 1`，否则 pipeline 会抛 `AssertionError`。比如 `num_frames=33` 对应 9 个 latent
> 帧，可用 `chunk_partition=[1, 4, 4]`。

```py
import numpy as np
import torch
from diffusers import AnyFlowFARPipeline
from diffusers.utils import export_to_video, load_image, load_video

pipe = AnyFlowFARPipeline.from_pretrained(
    "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")


def to_video_tensor(images, height=480, width=832):
    """把 PIL 列表转成 FAR pipeline 需要的 (B, C, T, H, W) [0, 1] 张量。"""
    frames = np.stack([np.asarray(img.resize((width, height))) for img in images]).astype("float32") / 255.0
    return torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)


# 1) 文生视频（无 context）。81 帧匹配默认 chunk_partition。
video = pipe(prompt="一只猫在夕阳下冲浪", num_inference_steps=4, num_frames=81).frames[0]
export_to_video(video, "t2v.mp4", fps=16)

# 2) 图生视频 —— 单帧 context 经过 VAE 是 1 个 latent，正好对上默认 chunk_partition 的第一项 (`[1, ...]`)。
first_frame = load_image("path/to/first_frame.png")
context_tensor = to_video_tensor([first_frame]).to("cuda")  # (1, 3, 1, 480, 832), [0, 1]
video = pipe(
    prompt="一只猫走过阳光下的草坪",
    context_sequence={"raw": context_tensor},
    num_inference_steps=4,
    num_frames=81,
).frames[0]
export_to_video(video, "i2v.mp4", fps=16)

# 3) 视频续写。9 帧 raw context → 3 个 latent context；显式覆盖 chunk_partition，让第一块正好覆盖 context。
context_frames = load_video("path/to/context.mp4")[:9]  # 9 = 4·2 + 1
context_tensor = to_video_tensor(context_frames).to("cuda")  # (1, 3, 9, 480, 832)
video = pipe(
    prompt="继续这个故事",
    context_sequence={"raw": context_tensor},
    num_inference_steps=4,
    num_frames=81,
    chunk_partition=[3, 3, 3, 3, 3, 3, 3],  # 7 个 chunk × 3 = 21 latent；首块就是 context
).frames[0]
export_to_video(video, "v2v.mp4", fps=16)
```

底层 patchify chunk 调度根据 `context_sequence` 自动调整：纯文生用 kernel 2 (full) 和 4 (compressed)；
有 context 时第一个 chunk 改成 kernel 1，让条件帧保留全分辨率。

如果你已经有 VAE 编码过的 latent，可以直接传 `context_sequence={"latent": ...}` 跳过 `vae_encode` 步骤。

## 显存与推理速度

14B 的 AnyFlow 模型用 group offload + VAE slicing 单卡 40 GB 能跑：

```py
import torch
from diffusers import AnyFlowPipeline
from diffusers.hooks import apply_group_offloading

pipe = AnyFlowPipeline.from_pretrained(
    "nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers", torch_dtype=torch.bfloat16
)
apply_group_offloading(pipe.transformer, onload_device="cuda", offload_type="leaf_level")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

延迟方面，`torch.compile` 对 transformer（最重的模块）效果很好：

```py
pipe = pipe.to("cuda")
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
```

编译开销跑几步就摊销掉；配合 AnyFlow 的低 NFE（4-8 步），`torch.compile` 在 14B 上相比 eager
模式有明显加速。

## LoRA 微调

两个 pipeline 都复用 [`WanLoraLoaderMixin`](../api/loaders/lora)，因此为对应 Wan2.1 backbone 训练的
LoRA adapter 直接加载即可：

```py
pipe.load_lora_weights("path/or/repo/with/wan_lora")
```

如果要做**继续 on-policy 蒸馏微调**（用论文里相同的 DMD 反向散度监督配方训新 LoRA），请参考原始
AnyFlow 训练框架 [`NVlabs/AnyFlow`](https://github.com/NVlabs/AnyFlow)，这套训练流程不在
diffusers 范围内。

## 常见坑

- **永远 `guidance_scale=1.0`。** 蒸馏后的 checkpoint 已经把 CFG 融进权重。设 `> 1` 会多跑一遍
  unconditional 前向、延迟翻倍、质量微降。
- **Bidirectional pipeline 不支持流式。** 所有 `num_frames` 一起去噪。需要边采边播请用 causal pipeline。
- **Causal pipeline KV cache 假设 chunk 调度跨调用一致。** 中途重建 cache 不被 release 模型支持。
- **`num_frames` 必须满足 VAE 时间步长。** release checkpoint 用 `(N - 1) % 4 == 0` 的值（如 9、17、33、81）。

## 引用

```bibtex
@misc{gu2026anyflowanystepvideodiffusion,
  title={AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation},
  author={Yuchao Gu and Guian Fang and Yuxin Jiang and Weijia Mao and Song Han and Han Cai and Mike Zheng Shou},
  year={2026},
  eprint={2605.13724},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.13724},
}

@article{gu2025long,
  title={Long-Context Autoregressive Video Modeling with Next-Frame Prediction},
  author={Gu, Yuchao and Mao, Weijia and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2503.19325},
  year={2025}
}
```
