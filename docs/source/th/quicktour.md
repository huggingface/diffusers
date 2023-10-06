<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# บทความฉบับย่อ

Diffusion models ได้ถูกเทรนเพื่อลด random Gaussian noise เป็นขั้นเล็กๆ (เรียกว่า timesteps) เพื่อสร้าง sample ที่น่าสนใจ เช่น รูปภาพหรือเสียง ซึ่งได้นำไปสู่การจุดประกายความสนใจอย่างมากในหัวข้อ generative AI และคุณอาจเคยเห็นตัวอย่างของภาพที่เกิดจาก Diffusion models เหล่านี้บนแหล่งต่างๆ 🧨 Diffusers เป็นไลบรารี่ที่มีจุดมุ่งหมายเพื่อทำให้ทุกคนสามารถเข้าถึง Diffusion modedls ได้อย่างง่ายดาย

ไม่ว่าคุณจะเป็นนักพัฒนาหรือเป็นผู้ใช้ทั่วไปก็ตาม บทความฉบับย่อนี่จะช่วยให้คุณเริ่มต้นใช้งาน 🧨 Diffusers อย่างรวดเร็ว โดยมีเพียง 3 ส่วนประกอบของไลบรารี่ที่คุณควรรู้จัก
* [`DiffusionPipeline`] เป็น high-level end-to-end class ที่ถูกออกแบบมาเพื่อการใช้งานอย่างง่ายในการเจน samples จาก pretrained diffusion models ด้วยการรัน inference
* Pretrained [model](./api/models) architectures และ modules ที่สามารถใช้เป็นจุดเริ่มต้นในการสร้าง diffusion systems ต่างๆ
* [Schedulers](./api/schedulers/overview) คืออัลกอริทึมที่ควบคุมการเติม noise ระหว่างการเทรนโมเดลและการรัน inference

 
<Tip>

บทความนี้ถูกย่อมาจาก 🧨 Diffusers [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb) เพื่อช่วยให้คุณเริ่มใช้งานไลบรารี่ได้ หากคุณต้องการเรียนรู้เพิ่มเติมเกี่ยวกับ 🧨 Diffusers ลองเช็ค notebook ดูสิ!

</Tip>

ก่อนเริ่ม เช็คก่อนว่าคุณลงไลบรารี่ที่จำเป็นต้องใช้แล้วหรือยัง:

```py
# uncomment to install the necessary libraries in Colab
#!pip install --upgrade diffusers accelerate transformers
```

- [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) ใช้ในการเพิ่มประสิทธิภาพการโหลดโมเดลต่างๆ และทำ distributed computing
- [🤗 Transformers](https://huggingface.co/docs/transformers/index) ถูกใช้ใน diffusion models ต่างๆ อย่าง [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview).

## DiffusionPipeline

[`DiffusionPipeline`] เป็นวิธีที่ง่ายที่สุดในการใช้งาน pretrained diffusion system สำหรับ inference มันคือ end-to-end system ที่ประกอบไปด้วย model และ scheduler โดยคุณสามารถใช้ [`DiffusionPipeline`] สำหรับหลายๆการใช้งาน ตารางนี้เป็นเพียงส่วนหนึ่งของการใช้งานที่ pipeline ต่างๆสามารถนำไปประกอบได้ และสำหรับข้อมูลเพิ่มเติมสามารถเข้าดูที่ [🧨 Diffusers Summary](./api/pipelines/overview#diffusers-summary)

| **Task**                     | **Description**                                                                                              | **Pipeline**
|------------------------------|--------------------------------------------------------------------------------------------------------------|-----------------|
| Unconditional Image Generation          | generate an image from Gaussian noise | [unconditional_image_generation](./using-diffusers/unconditional_image_generation) |
| Text-Guided Image Generation | generate an image given a text prompt | [conditional_image_generation](./using-diffusers/conditional_image_generation) |
| Text-Guided Image-to-Image Translation     | adapt an image guided by a text prompt | [img2img](./using-diffusers/img2img) |
| Text-Guided Image-Inpainting          | fill the masked part of an image given the image, the mask and a text prompt | [inpaint](./using-diffusers/inpaint) |
| Text-Guided Depth-to-Image Translation | adapt parts of an image guided by a text prompt while preserving structure via depth estimation | [depth2img](./using-diffusers/depth2img) |

เริ่มจากการสร้าง instance ของ [`DiffusionPipeline`] และระบุ checkpoint ที่ต้องการใช้งาน
คุณสามารถใช้ [`DiffusionPipeline`] กับ [checkpoint](https://huggingface.co/models?library=diffusers&sort=downloads) ที่อยู่บน Hugging Face Hub ในที่นี้เราจะใช้ [`stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) สำหรับการเจนรูปภาพจาก text.

<Tip warning={true}>

สำหรับ [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion) ควรอ่าน [license](https://huggingface.co/spaces/CompVis/stable-diffusion-license) ก่อนใช้งาน นอกจากนั้น 🧨 Diffusers มีการใช้งาน [`safety_checker`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py) สำหรับการป้องกันเนื้อหาที่ไม่เหมาะสม คุณสามารถปิดโมดูลนี้ได้ แต่ควรระวังกับผลลัพธ์ที่ไม่เหมาะสมที่ได้จาก inference

</Tip>

โหลดโมเดลด้วย [`~DiffusionPipeline.from_pretrained`]:

```python
>>> from diffusers import DiffusionPipeline

>>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
```

[`DiffusionPipeline`] จะดาวน์โหลดและเก็บโมดูล modeling, tokenization, และ scheduling เห็นได้ว่า Stable Diffusion pipeline ประกอบไปด้วย [`UNet2DConditionModel`] และ [`PNDMScheduler`] รวมทั้งโมดูลอื่นๆ:

```py
>>> pipeline
StableDiffusionPipeline {
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.13.1",
  ...,
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  ...,
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```

เราแนะนำให้รัน pipeline บน GPU เพราะโมเดลประกอบไปด้วย 1.4 พันล้านพารามิเตอร์
คุณสามารถย้าย generator object ไปอยู่บน GPU เหมือนกับโมดูลต่างๆใน PyTorch:

```python
>>> pipeline.to("cuda")
```

หลังจากน้ันก็ให้ text prompt กับ `pipeline` เพื่อเจนรูปภาพในฟอร์แมตของ [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=image#the-image-class)

```python
>>> image = pipeline("An image of a squirrel in Picasso style").images[0]
>>> image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/image_of_squirrel_painting.png"/>
</div>

เซฟรูปภาพเป็นไฟล์ด้วย `save`:

```python
>>> image.save("image_of_squirrel_painting.png")
```

### Local pipeline

คุณสามารถรัน pipeline ด้วยโมเดลที่เซฟไว้ก่อนแล้วได้ เริ่มจากการโหลดโมเดลลงไว้ที่ local storage

```bash
!git lfs install
!git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

และโหลดโมเดลเข้า pipeline

```python
>>> pipeline = DiffusionPipeline.from_pretrained("./stable-diffusion-v1-5", use_safetensors=True)
```

และคุณสามารถรัน pipeline ได้เหมือนวิธีก่อนหน้า

### สลับ schedulers

Schedulers ต่างๆมีจุดเด่นและด้อยที่แตกต่างกัน การสลับ schedulers ใน 🧨 Diffusers ทำได้ไม่ยาก เช่นการสลับ [`PNDMScheduler`] เป็น [`EulerDiscreteScheduler`] ทำได้โดยการเรียก [`~diffusers.ConfigMixin.from_config`]:

```py
>>> from diffusers import EulerDiscreteScheduler

>>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
>>> pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
```

ลองเจนรูปด้วย scheduler ใหม่ดูและสังเกตความแตกต่างสิ!

ในส่วนต่อไปเราจะมาดู model และ scheduler ซึ่งเป็นส่วนประกอบของ [`DiffusionPipeline`] และมาดูวิธีการใช้งานมันในการเจนรูปภาพของน้องแมวกัน

## Models

Models ส่วนมากจะรับ noisy sample และในแต่ละ timestep มันจะคาดเดา *noise residual* ซี่งคือส่วนต่างระหว่างรูปที่มี noise น้อยกว่า sample ปัจจุบัน ณ timestep นั้นๆ (models อื่นๆอาจเรียนรู้วิธีการเดา latent image sample โดยตรงหรืออาจเดา velocity หรือ [`v-prediction`](https://github.com/huggingface/diffusers/blob/5e5ce13e2f89ac45a0066cb3f369462a3cf1d9ef/src/diffusers/schedulers/scheduling_ddim.py#L110))

Models ถูกสร้างด้วย [`~ModelMixin.from_pretrained`] ซึ่งจะโหลดโมเดลจาก Huggingface Hub ไว้ที่ local storage ด้วย โดยสำหรับส่วนนี้เราจะใช้ [`UNet2DModel`] ที่เป็น basic unconditional image generation model ด้วย checkpoint ที่เทรนให้เจนรูปน้องแมวโดยเฉพาะ:

```py
>>> from diffusers import UNet2DModel

>>> repo_id = "google/ddpm-cat-256"
>>> model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True)
```

หากต้องการทราบ model parameters ให้เรียก `model.config`:

```py
>>> model.config
```

Model configuration นั้นเป็น 🧊 frozen 🧊 dictionary หมายถึง parameters เหล่านั้นไม่สามารถถูกเปลี่ยนแปลงหลัง model ได้ถูกสร้างขึ้น แต่ parameters อื่นๆ นั้นสามารถเปลี่ยนแปลงได้ เช่น inference parameters

Model configuration parameters ที่สำคัญอาจมี:

* `sample_size`: the height and width dimension of the input sample.
* `in_channels`: the number of input channels of the input sample.
* `down_block_types` and `up_block_types`: the type of down- and upsampling blocks used to create the UNet architecture.
* `block_out_channels`: the number of output channels of the downsampling blocks; also used in reverse order for the number of input channels of the upsampling blocks.
* `layers_per_block`: the number of ResNet blocks present in each UNet block.

สำหรับการใช้งานโมเดลเพื่อรัน inference ให้สร้าง random Gaussian noise ที่มี dimensions ตรงกับ image shape ที่ต้องการ โดย tensor เป็น 4 มิติประกอบไปด้วย `batch` axis, `channel` axis, และ `sample_size` axis (แตกออกเป็น `height` และ `width` axis):

```py
>>> import torch

>>> torch.manual_seed(0)

>>> noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
>>> noisy_sample.shape
torch.Size([1, 3, 256, 256])
```

หลังจากนั้นให้เอา tensor และ `timestep` ส่งไปให้ model ซึ่ง `timestep` นี้จะบอกถึงระดับ noise ของ input image ที่จะช่วยให้ model รู้ถึงระดับ noise และตำแหน่งของมันใน SDE trajectory ระหว่างทำ diffusion ว่ามันอยู่ใกล้จุดเริ่มต้นหรือปลายทาง หลังจากนั้นใช้ `sample` ในการเข้าถึง output ของโมเดล:

```py
>>> with torch.no_grad():
...     noisy_residual = model(sample=noisy_sample, timestep=2).sample
```

แต่หากต้องการเจนรูปจริงๆ ควรใช้ scheduler เป็นตัวควบคุมแต่ละ timesteps และจำนวนของ noise ที่ใส่เข้าไปใน sample

## Schedulers

Schedulers ควบคุม `noisy_residual` ที่บอกถึงระดับ noise ในแต่ละ sample

<Tip>

🧨 Diffusers สามารถทำงานร่วมกับ custom model หรือ custom schedulers ของคุณเองได้ แม้ [`DiffusionPipeline`] จะเป็นวิธีที่รวดเร็วที่สุดในการเริ่มใช้งาน prebuilt pipelines ก็ตาม

</Tip>

ในที่นี้ เราจะใช้ [`DDPMScheduler`] โดยสร้างมันโดยการเรียก [`~diffusers.ConfigMixin.from_config`]:

```py
>>> from diffusers import DDPMScheduler

>>> scheduler = DDPMScheduler.from_config(repo_id)
>>> scheduler
DDPMScheduler {
  "_class_name": "DDPMScheduler",
  "_diffusers_version": "0.13.1",
  "beta_end": 0.02,
  "beta_schedule": "linear",
  "beta_start": 0.0001,
  "clip_sample": true,
  "clip_sample_range": 1.0,
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  "trained_betas": null,
  "variance_type": "fixed_small"
}
```

<Tip>

💡 สังเกตว่า scheduler ถูกสร้างขึ้นจาก configuration เนื่องจาก schedulers ไม่จำเป็นต้องใช้ checkpoints เหมือนส่วนอื่นๆของ pipeline

</Tip>

Parameters ที่สำคัญประกอบไปด้วย:

* `num_train_timesteps`: the length of the denoising process or in other words, the number of timesteps required to process random Gaussian noise into a data sample.
* `beta_schedule`: the type of noise schedule to use for inference and training.
* `beta_start` and `beta_end`: the start and end noise values for the noise schedule.

เพื่อได้ sample ถัดไปให้เรียก [`~diffusers.DDPMScheduler.step`] ด้วย `timestep` และ `sample` ปัจจุบัน.

```py
>>> less_noisy_sample = scheduler.step(model_output=noisy_residual, timestep=2, sample=noisy_sample).prev_sample
>>> less_noisy_sample.shape
```

`less_noisy_sample` สามารถนำไปควบกับ `timestep` และส่งต่อให้โมเดลเพื่อคาดเดา noisy residual และวนขั้นตอนกลับ จนได้รูปภาพที่ไม่มี noise ในที่สุด 

ก่อนหน้านั้น ควรสร้าง function ในการแสดง `PIL.Image` ที่ได้:

```py
>>> import PIL.Image
>>> import numpy as np


>>> def display_sample(sample, i):
...     image_processed = sample.cpu().permute(0, 2, 3, 1)
...     image_processed = (image_processed + 1.0) * 127.5
...     image_processed = image_processed.numpy().astype(np.uint8)

...     image_pil = PIL.Image.fromarray(image_processed[0])
...     display(f"Image at step {i}")
...     display(image_pil)
```

หลังจากนั้นย้ายทั้ง input tensor และ model ไป GPU:

```py
>>> model.to("cuda")
>>> noisy_sample = noisy_sample.to("cuda")
```

และสร้าง denoising loop เพื่อให้ model เดา residual ของ noisy sample ก่อนหน้าและส่งต่อให้ scheduler นำผลลัพธ์ไปรวมกันเพื่อได้ sample ที่มีระดับ noise ลดลง:

```py
>>> import tqdm

>>> sample = noisy_sample

>>> for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
...     # 1. predict noise residual
...     with torch.no_grad():
...         residual = model(sample, t).sample

...     # 2. compute less noisy image and set x_t -> x_t-1
...     sample = scheduler.step(residual, t, sample).prev_sample

...     # 3. optionally look at image
...     if (i + 1) % 50 == 0:
...         display_sample(sample, i + 1)
```

เรามาดูผลลัพธ์กัน 😻

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/diffusion-quicktour.png"/>
</div>

## Next steps

สำหรับเส้นทางถัดๆไป คุณอาจลองสิ่งเหล่านี้ดู:

* Train or finetune a model to generate your own images in the [training](./tutorials/basic_training) tutorial.
* See example official and community [training or finetuning scripts](https://github.com/huggingface/diffusers/tree/main/examples#-diffusers-examples) for a variety of use cases.
* Learn more about loading, accessing, changing and comparing schedulers in the [Using different Schedulers](./using-diffusers/schedulers) guide.
* Explore prompt engineering, speed and memory optimizations, and tips and tricks for generating higher quality images with the [Stable Diffusion](./stable_diffusion) guide.
* Dive deeper into speeding up 🧨 Diffusers with guides on [optimized PyTorch on a GPU](./optimization/fp16), and inference guides for running [Stable Diffusion on Apple Silicon (M1/M2)](./optimization/mps) and [ONNX Runtime](./optimization/onnx).
