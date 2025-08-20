# 入门：使用混合推理进行 VAE 编码

VAE 编码用于训练、图像到图像和图像到视频——将图像或视频转换为潜在表示。

## 内存

这些表格展示了在不同 GPU 上使用 SD v1 和 SD XL 进行 VAE 编码的 VRAM 需求。

对于这些 GPU 中的大多数，内存使用百分比决定了其他模型（文本编码器、UNet/Transformer）必须被卸载，或者必须使用分块编码，这会增加时间并影响质量。

<details><summary>SD v1.5</summary>

| GPU                           | 分辨率   |   时间（秒） |   内存（%） |   分块时间（秒） |   分块内存（%） |
|:------------------------------|:-------------|-----------------:|-------------:|--------------------:|-------------------:|
| NVIDIA GeForce RTX 4090       | 512x512      |            0.015 |      3.51901 |               0.015 |            3.51901 |
| NVIDIA GeForce RTX 4090       | 256x256      |            0.004 |      1.3154  |               0.005 |            1.3154  |
| NVIDIA GeForce RTX 4090       | 2048x2048    |            0.402 |     47.1852  |               0.496 |            3.51901 |
| NVIDIA GeForce RTX 4090       | 1024x1024    |            0.078 |     12.2658  |               0.094 |            3.51901 |
| NVIDIA GeForce RTX 4080 SUPER | 512x512      |            0.023 |      5.30105 |               0.023 |            5.30105 |
| NVIDIA GeForce RTX 4080 SUPER | 256x256      |            0.006 |      1.98152 |               0.006 |            1.98152 |
| NVIDIA GeForce RTX 4080 SUPER | 2048x2048    |            0.574 |     71.08    |               0.656 |            5.30105 |
| NVIDIA GeForce RTX 4080 SUPER | 1024x1024    |            0.111 |     18.4772  |               0.14  |            5.30105 |
| NVIDIA GeForce RTX 3090       | 512x512      |            0.032 |      3.52782 |               0.032 |            3.52782 |
| NVIDIA GeForce RTX 3090       | 256x256      |            0.01  |      1.31869 |               0.009 |            1.31869 |
| NVIDIA GeForce RTX 3090       | 2048x2048    |            0.742 |     47.3033  |               0.954 |            3.52782 |
| NVIDIA GeForce RTX 3090       | 1024x1024    |            0.136 |     12.2965  |               0.207 |            3.52782 |
| NVIDIA GeForce RTX 3080       | 512x512      |            0.036 |      8.51761 |               0.036 |            8.51761 |
| NVIDIA GeForce RTX 3080       | 256x256      |            0.01  |      3.18387 |               0.01  |            3.18387 |
| NVIDIA GeForce RTX 3080       | 2048x2048    |            0.863 |     86.7424  |               1.191 |            8.51761 |
| NVIDIA GeForce RTX 3080       | 1024x1024    |            0.157 |     29.6888  |               0.227 |            8.51761 |
| NVIDIA GeForce RTX 3070       | 512x512      |            0.051 |     10.6941  |               0.051 |           10.6941  |
| NVIDIA GeForce RTX 3070       | 256x256      |            0.015 |
|      3.99743 |               0.015 |            3.99743 |
| NVIDIA GeForce RTX 3070       | 2048x2048    |            1.217 |     96.054   |               1.482 |           10.6941  |
| NVIDIA GeForce RTX 3070       | 1024x1024    |            0.223 |     37.2751  |               0.327 |           10.6941  |

</details>

<details><summary>SDXL</summary>

| GPU                           | Resolution   |   Time (seconds) |   Memory Consumed (%) |   Tiled Time (seconds) |   Tiled Memory (%) |
|:------------------------------|:-------------|-----------------:|----------------------:|-----------------------:|-------------------:|
| NVIDIA GeForce RTX 4090       | 512x512      |            0.029 |               4.95707 |                  0.029 |            4.95707 |
| NVIDIA GeForce RTX 4090       | 256x256      |            0.007 |               2.29666 |                  0.007 |            2.29666 |
| NVIDIA GeForce RTX 4090       | 2048x2048    |            0.873 |              66.3452  |                  0.863 |           15.5649  |
| NVIDIA GeForce RTX 4090       | 1024x1024    |            0.142 |              15.5479  |                  0.143 |           15.5479  |
| NVIDIA GeForce RTX 4080 SUPER | 512x512      |            0.044 |               7.46735 |                  0.044 |            7.46735 |
| NVIDIA GeForce RTX 4080 SUPER | 256x256      |            0.01  |               3.4597  |                  0.01  |            3.4597  |
| NVIDIA GeForce RTX 4080 SUPER | 2048x2048    |            1.317 |              87.1615  |                  1.291 |           23.447   |
| NVIDIA GeForce RTX 4080 SUPER | 1024x1024    |            0.213 |              23.4215  |                  0.214 |           23.4215  |
| NVIDIA GeForce RTX 3090       | 512x512      |            0.058 |               5.65638 |                  0.058 |            5.65638 |
| NVIDIA GeForce RTX 3090       | 256x256      |            0.016 |               2.45081 |                  0.016 |            2.45081 |
| NVIDIA GeForce RTX 3090       | 2048x2048    |            1.755 |              77.8239  |                  1.614 |           18.4193  |
| NVIDIA GeForce RTX 3090       | 1024x1024    |            0.265 |              18.4023  |                  0.265 |           18.4023  |
| NVIDIA GeForce RTX 3080       | 512x512      |            0.064 |              13.6568  |                  0.064 |           13.6568  |
| NVIDIA GeForce RTX 3080       | 256x256      |            0.018 |               5.91728 |                  0.018 |            5.91728 |
| NVIDIA GeForce RTX 3080       | 2048x2048    |          内存不足 (OOM) |             内存不足 (OOM) |                  1.866 |           44.4717  |
| NVIDIA GeForce RTX 3080       | 1024x1024    |            0.302 |              44.4308  |                  0.302 |           44.4308  |
| NVIDIA GeForce RTX 3070       | 512x512      |            0.093 |              17.1465  |                  0.093 |           17.1465  |
| NVIDIA GeForce R
| NVIDIA GeForce RTX 3070       | 256x256      |            0.025 |               7.42931 |                  0.026 |            7.42931 |
| NVIDIA GeForce RTX 3070       | 2048x2048    |          OOM     |             OOM       |                  2.674 |           55.8355  |
| NVIDIA GeForce RTX 3070       | 1024x1024    |            0.443 |              55.7841  |                  0.443 |           55.7841  |

</details>

## 可用 VAE

|   | **端点** | **模型** |
|:-:|:-----------:|:--------:|
| **Stable Diffusion v1** | [https://qc6479g0aac6qwy9.us-east-1.aws.endpoints.huggingface.cloud](https://qc6479g0aac6qwy9.us-east-1.aws.endpoints.huggingface.cloud) | [`stabilityai/sd-vae-ft-mse`](https://hf.co/stabilityai/sd-vae-ft-mse) |
| **Stable Diffusion XL** | [https://xjqqhmyn62rog84g.us-east-1.aws.endpoints.huggingface.cloud](https://xjqqhmyn62rog84g.us-east-1.aws.endpoints.huggingface.cloud) | [`madebyollin/sdxl-vae-fp16-fix`](https://hf.co/madebyollin/sdxl-vae-fp16-fix) |
| **Flux** | [https://ptccx55jz97f9zgo.us-east-1.aws.endpoints.huggingface.cloud](https://ptccx55jz97f9zgo.us-east-1.aws.endpoints.huggingface.cloud) | [`black-forest-labs/FLUX.1-schnell`](https://hf.co/black-forest-labs/FLUX.1-schnell) |


> [!TIP]
> 模型支持可以在此处请求：[这里](https://github.com/huggingface/diffusers/issues/new?template=remote-vae-pilot-feedback.yml)。


## 代码

> [!TIP]
> 从 `main` 安装 `diffusers` 以运行代码：`pip install git+https://github.com/huggingface/diffusers@main`


一个辅助方法简化了与混合推理的交互。

```python
from diffusers.utils.remote_utils import remote_encode
```

### 基本示例

让我们编码一张图像，然后解码以演示。

<figure class="image flex flex-col items-center justify-center text-center m-0 w-full">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"/>
</figure>

<details><summary>代码</summary>

```python
from diffusers.utils import load_image
from diffusers.utils.remote_utils import remote_decode

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg?download=true")

latent = remote_encode(
    endpoint="https://ptccx55jz97f9zgo.us-east-1.aws.endpoints.huggingface.cloud/",
    scaling_factor=0.3611,
    shift_factor=0.1159,
)

decoded = remote_decode(
    endpoint="https://whhx50ex1aryqvw6.us-east-1.aws.endpoints.huggingface.cloud/",
    tensor=latent,
    scaling_factor=0.3611,
    shift_factor=0.1159,
)
```

</details>

<figure class="image flex flex-col items-center justify-center text-center m-0 w-full">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/remote_vae/decoded.png"/>
</figure>


### 生成

现在让我们看一个生成示例，我们将编码图像，生成，然后远程解码！

<details><summary>代码</summary>

```python
import torch
from diffusers import StableDiffusionImg2ImgPip
from diffusers.utils import load_image
from diffusers.utils.remote_utils import remote_decode, remote_encode

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
    vae=None,
).to("cuda")

init_image = load_image(
    "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
)
init_image = init_image.resize((768, 512))

init_latent = remote_encode(
    endpoint="https://qc6479g0aac6qwy9.us-east-1.aws.endpoints.huggingface.cloud/",
    image=init_image,
    scaling_factor=0.18215,
)

prompt = "A fantasy landscape, trending on artstation"
latent = pipe(
    prompt=prompt,
    image=init_latent,
    strength=0.75,
    output_type="latent",
).images

image = remote_decode(
    endpoint="https://q1bj3bpq6kzilnsu.us-east-1.aws.endpoints.huggingface.cloud/",
    tensor=latent,
    scaling_factor=0.18215,
)
image.save("fantasy_landscape.jpg")
```

</details>

<figure class="image flex flex-col items-center justify-center text-center m-0 w-full">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/remote_vae/fantasy_landscape.png"/>
</figure>

## 集成

* **[SD.Next](https://github.com/vladmandic/sdnext):** 具有直接支持混合推理功能的一体化用户界面。
* **[ComfyUI-HFRemoteVae](https://github.com/kijai/ComfyUI-HFRemoteVae):** 用于混合推理的 ComfyUI 节点。