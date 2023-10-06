<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->


# ระบบ ONNX

🤗 [Optimum](https://github.com/huggingface/optimum) นำเสนอกระบวนการการ Diffusion ที่เสถียรภาพสู่ระบบ ONNX ที่เข้ากันได้ คุณจะต้องติดตั้ง 🤗 Optimum ด้วยคำสั่งต่อไปนี้เพื่อรองรับ ONNX Runtime:

```bash
pip install optimum["onnxruntime"]
```

คู่มือนี้จะแสดงวิธีการใช้ Stable Diffusion และ Stable Diffusion XL (SDXL) กับ ONNX Runtime.

## Stable Diffusion

เพื่อโหลดและเรียกใช้การนำไปใช้งานใช้ [`~optimum.onnxruntime.ORTStableDiffusionPipeline`]. หากต้องการโหลดโมเดล PyTorch และแปลงเป็นรูปแบบ ONNX โดยตรง ให้ตั้งค่า `export=True`:

```python
from optimum.onnxruntime import ORTStableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipeline = ORTStableDiffusionPipeline.from_pretrained(model_id, export=True)
prompt = "sailing ship in storm by Leonardo da Vinci"
image = pipeline(prompt).images[0]
pipeline.save_pretrained("./onnx-stable-diffusion-v1-5")
```

<Tip warning={true}>

การสร้างพร้อมทีในชุดของพรอมป์แบบหลายตัวอาจใช้หน่วยความจำมากเกินไป ในขณะที่เรากำลังศึกษาปัญหานี้ คุณอาจต้องทำการวนลูปแทนที่จะใช้แบทช์.

</Tip>

เพื่อส่งออกกระบวนการในรูปแบบ ONNX แบบออฟไลน์และใช้ในภายหลังสำหรับการนำไปใช้, ให้ใช้คำสั่ง [`optimum-cli export`](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#exporting-a-model-to-onnx-using-the-cli) นี้:

```bash
optimum-cli export onnx --model runwayml/stable-diffusion-v1-5 sd_v15_onnx/
```

จากนั้นเพื่อทำการนำไปใช้ (คุณไม่จำเป็นต้องระบุ `export=True` อีกครั้ง):

```python 
from optimum.onnxruntime import ORTStableDiffusionPipeline

model_id = "sd_v15_onnx"
pipeline = ORTStableDiffusionPipeline.from_pretrained(model_id)
prompt = "sailing ship in storm by Leonardo da Vinci"
image = pipeline(prompt).images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/optimum/documentation-images/resolve/main/onnxruntime/stable_diffusion_v1_5_ort_sail_boat.png">
</div>

คุณสามารถค้นหาตัวอย่างเพิ่มเติมใน [เอกสาร](https://huggingface.co/docs/optimum/) ของ 🤗 Optimum, และ Stable Diffusion ได้รับการสนับสนุนสำหรับการแปลงข้อความเป็นภาพ, ภาพเป็นภาพ, และการทำซ่อมภาพ.

## Stable Diffusion XL

เพื่อโหลดและเรียกใช้การนำไปใช้กับ SDXL ใช้ [`~optimum.onnxruntime.ORTStableDiffusionXLPipeline`] ดังนี้:

```python
from optimum.onnxruntime import ORTStableDiffusionXLPipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = ORTStableDiffusionXLPipeline.from_pretrained(model_id)
prompt = "sailing ship in storm by Leonardo da Vinci"
image = pipeline(prompt).images[0]
```

เพื่อส่งออกกระบวนการในรูปแบบ ONNX และใช้ในภายหลังสำหรับการนำไปใช้, ให้ใช้คำสั่ง [`optimum-cli export`](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#exporting-a-model-to-onnx-using-the-cli) นี้:

```bash
optimum-cli export onnx --model stabilityai/stable-diffusion-xl-base-1.0 --task stable-diffusion-xl sd_xl_onnx/
```

SDXL ในรูปแบบ ONNX ได้รับการสนับสนุนสำหรับการแปลงข้อความเป็นภาพและภาพเป็นภาพ.