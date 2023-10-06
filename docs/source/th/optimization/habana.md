<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Habana Gaudi

🤗 **Diffusers** เข้ากันได้กับ Habana Gaudi ผ่าน 🤗 [Optimum](https://huggingface.co/docs/optimum/habana/usage_guides/stable_diffusion) ทำตาม [คู่มือการติดตั้ง](https://docs.habana.ai/en/latest/Installation_Guide/index.html) เพื่อติดตั้ง SynapseAI และ Gaudi drivers จากนั้นติดตั้ง Optimum Habana:

```bash
python -m pip install --upgrade-strategy eager optimum[habana]
```

เพื่อสร้างรูปภาพด้วย Stable Diffusion 1 และ 2 บน Gaudi คุณต้องสร้าง instances สองตัว:

- [`~optimum.habana.diffusers.GaudiStableDiffusionPipeline`], pipeline การสร้างข้อความเป็นภาพ
- [`~optimum.habana.diffusers.GaudiDDIMScheduler`], ตัวตั้งเวลาที่ถูกปรับแต่งเพื่อให้เหมาะสมกับ Gaudi

เมื่อคุณเริ่มต้นทำการสร้าง pipeline, คุณต้องระบุ `use_habana=True` เพื่อนำไปใช้บน HPUs และเพื่อให้ได้การสร้างที่เร็วที่สุดคุณควรเปิดใช้ **HPU graphs** ด้วย `use_hpu_graphs=True`.

ในท้ายที่สุด, ระบุ [`~optimum.habana.GaudiConfig`] ซึ่งสามารถดาวน์โหลดได้จาก [Habana](https://huggingface.co/Habana) organization บน Hub.

```python
from optimum.habana import GaudiConfig
from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionPipeline

model_name = "stabilityai/stable-diffusion-2-base"
scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
pipeline = GaudiStableDiffusionPipeline.from_pretrained(
    model_name,
    scheduler=scheduler,
    use_habana=True,
    use_hpu_graphs=True,
    gaudi_config="Habana/stable-diffusion-2",
)
```

ตอนนี้คุณสามารถเรียกใช้ pipeline เพื่อสร้างรูปภาพทีละชุดจาก prompt หนึ่งหรือหลาย prompt:

```python
outputs = pipeline(
    prompt=[
        "High quality photo of an astronaut riding a horse in space",
        "Face of a yellow cat, high resolution, sitting on a park bench",
    ],
    num_images_per_prompt=10,
    batch_size=4,
)
```

สำหรับข้อมูลเพิ่มเติม ดูที่ 🤗 Optimum Habana's [documentation](https://huggingface.co/docs/optimum/habana/usage_guides/stable_diffusion) และ [ตัวอย่าง](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion) ที่ให้ไว้ใน official Github repository.


## การทดสอบผล

เราได้นำ Habana Gaudi รุ่น 1 และ Gaudi2 ไปทดสอบประสิทธิภาพด้วย [Habana/stable-diffusion](https://huggingface.co/Habana/stable-diffusion) และ [Habana/stable-diffusion-2](https://huggingface.co/Habana/stable-diffusion-2) Gaudi configurations (mixed precision bf16/fp32) เพื่อแสดงประสิทธิภาพของพวกเขา.

สำหรับ [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) บนรูปภาพขนาด 512x512:

|                        | Latency (batch size = 1) | Throughput  |
| ---------------------- |:------------------------:|:---------------------------:|
| first-generation Gaudi | 3.80s                    | 0.308 images/s (batch size = 8)             |
| Gaudi2                 | 1.33s                    | 1.081 images/s (batch size = 8)             |

สำหรับ [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) บนรูปภาพขนาด 768x768:

|                        | Latency (batch size = 1) | Throughput                      |
| ---------------------- |:------------------------:|:-------------------------------:|
| first-generation Gaudi | 10.2s                    | 0.108 images/s (batch size = 4) |
| Gaudi2                 | 3.17s                    | 0.379 images/s (batch size = 8) |
