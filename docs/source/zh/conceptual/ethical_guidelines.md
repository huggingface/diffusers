<!--版权归2025年HuggingFace团队所有。保留所有权利。

根据Apache许可证2.0版（"许可证"）授权；除非符合许可证要求，否则不得使用此文件。您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，本软件按"原样"分发，不附带任何明示或暗示的担保或条件。详见许可证中规定的特定语言权限和限制。
-->

# 🧨 Diffusers伦理准则

## 前言

[Diffusers](https://huggingface.co/docs/diffusers/index)不仅提供预训练的diffusion模型，还是一个模块化工具箱，支持推理和训练功能。

鉴于该技术在实际场景中的应用及其可能对社会产生的负面影响，我们认为有必要制定项目伦理准则，以指导Diffusers库的开发、用户贡献和使用规范。

该技术涉及的风险仍在持续评估中，主要包括但不限于：艺术家版权问题、深度伪造滥用、不当情境下的色情内容生成、非自愿的人物模仿、以及加剧边缘群体压迫的有害社会偏见。我们将持续追踪风险，并根据社区反馈动态调整本准则。

## 适用范围

Diffusers社区将在项目开发中贯彻以下伦理准则，并协调社区贡献的整合方式，特别是在涉及伦理敏感议题的技术决策时。

## 伦理准则

以下准则具有普遍适用性，但我们主要在处理涉及伦理敏感问题的技术决策时实施。同时，我们承诺将根据技术发展带来的新兴风险持续调整这些原则：

- **透明度**：我们承诺以透明方式管理PR（拉取请求），向用户解释决策依据，并公开技术选择过程。

- **一致性**：我们承诺为用户提供统一标准的项目管理，保持技术稳定性和连贯性。

- **简洁性**：为了让Diffusers库更易使用和开发，我们承诺保持项目目标精简且逻辑自洽。

- **可及性**：本项目致力于降低贡献门槛，即使非技术人员也能参与运营，从而使研究资源更广泛地服务于社区。

- **可复现性**：对于通过Diffusers库发布的上游代码、模型和数据集，我们将明确说明其可复现性。

- **责任性**：作为社区和团队，我们共同承担用户责任，通过风险预判和缓解措施来应对技术潜在危害。

## 实施案例：安全功能与机制

团队持续开发技术和非技术工具，以应对diffusion技术相关的伦理与社会风险。社区反馈对于功能实施和风险意识提升具有不可替代的价值：

- [**社区讨论区**](https://huggingface.co/docs/hub/repositories-pull-requests-discussions)：促进社区成员就项目开展协作讨论。

- **偏见探索与评估**：Hugging Face团队提供[交互空间](https://huggingface.co/spaces/society-ethics/DiffusionBiasExplorer)展示Stable Diffusion中的偏见。我们支持并鼓励此类偏见探索与评估工作。

- **部署安全强化**：
  
  - [**Safe Stable Diffusion**](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_safe)：解决Stable Diffusion等基于未过滤网络爬取数据训练的模型容易产生不当内容的问题。相关论文：[Safe Latent Diffusion：缓解diffusion模型中的不当退化](https://huggingface.co/papers/2211.05105)。

  - [**安全检测器**](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py)：通过比对图像生成后嵌入空间中硬编码有害概念集的类别概率进行检测。有害概念列表经特殊处理以防逆向工程。

- **分阶段模型发布**：对于高度敏感的仓库，采用分级访问控制。这种阶段性发布机制让作者能更好地管控使用场景。

- **许可证制度**：采用新型[OpenRAILs](https://huggingface.co/blog/open_rail)许可协议，在保障开放访问的同时设置使用限制以确保更负责任的应用。
