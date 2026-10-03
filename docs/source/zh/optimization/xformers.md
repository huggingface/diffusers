<!--版权归2025年HuggingFace团队所有。保留所有权利。

根据Apache许可证2.0版（"许可证"）授权；除非符合许可证要求，否则不得使用本文件。您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，本软件按"原样"分发，不附带任何明示或暗示的担保或条件。详见许可证中规定的特定语言及限制条款。
-->

# xFormers

我们推荐在推理和训练过程中使用[xFormers](https://github.com/facebookresearch/xformers)。在我们的测试中，其对注意力模块的优化能同时提升运行速度并降低内存消耗。

通过`pip`安装xFormers：

```bash
pip install xformers
```

> [!TIP]
> xFormers的`pip`安装包需要最新版本的PyTorch。如需使用旧版PyTorch，建议[从源码安装xFormers](https://github.com/facebookresearch/xformers#installing-xformers)。

安装完成后，您可调用`enable_xformers_memory_efficient_attention()`来实现更快的推理速度和更低的内存占用，具体用法参见[此章节](memory#memory-efficient-attention)。

> [!WARNING]
> 根据[此问题](https://github.com/huggingface/diffusers/issues/2234#issuecomment-1416931212)反馈，xFormers `v0.0.16`版本在某些GPU上无法用于训练（微调或DreamBooth）。如遇此问题，请按照该issue评论区指引安装开发版本。