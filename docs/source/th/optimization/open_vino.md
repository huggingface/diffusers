<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->


# OpenVINO

🤗 [Optimum](https://github.com/huggingface/optimum-intel) ให้ทางเลือกที่ดีที่สุด ที่เข้ากันได้กับ Diffusion Model ที่เสถียรที่สุด ที่เข้ากันได้กับ OpenVINO เพื่อทำการ inference บนหลายๆ ประเภทของ Intel processors (ดู [รายการเต็ม](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html) ของอุปกรณ์ที่รองรับ).

คุณต้องติดตั้ง 🤗 Optimum Intel ด้วยตัวเลือก `--upgrade-strategy eager` เพื่อให้แน่ใจว่า [`optimum-intel`](https://github.com/huggingface/optimum-intel) ใช้เวอร์ชันล่าสุด:

```
pip install --upgrade-strategy eager optimum["openvino"]
```

คู่มือนี้จะแสดงวิธีการใช้ Stable Diffusion และ Stable Diffusion XL (SDXL) กับ OpenVINO.

## Stable Diffusion

เพื่อโหลดและเรียกใช้ inference ใช้ [`~optimum.intel.OVStableDiffusionPipeline`]. หากคุณต้องการโหลดโมเดล PyTorch และแปลงรูปแบบเป็น OpenVINO โดยแบบง่าย ๆ ตั้งค่า `export=True`:

```python
from optimum.intel import OVStableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipeline = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)
prompt = "sailing ship in storm by Rembrandt"
image = pipeline(prompt).images[0]

# อย่าลืมบันทึกโมเดลที่แปลงรูปแบบไว้
pipeline.save_pretrained("openvino-sd-v1-5")
```

เพื่อเร่งความเร็วของการ inference ต่อไป ทำการเปลี่ยนรูปแบบของโมเดลเป็นแบบสถิต (statically reshape) โมเดล. หากคุณเปลี่ยนพารามิเตอร์ใด ๆ เช่น ความสูงหรือความกว้างของผลลัพธ์ คุณจะต้องเปลี่ยนรูปแบบของโมเดลอีกครั้ง.

```python
# กำหนดรูปแบบที่เกี่ยวข้องกับข้อมูลนำเข้าและผลลัพธ์ที่ต้องการ
batch_size, num_images, height, width = 1, 1, 512, 512

# เปลี่ยนรูปแบบของโมเดลเป็นแบบสถิต
pipeline.reshape(batch_size, height, width, num_images)
# คอมไพล์โมเดลก่อนการ inference
pipeline.compile()

image = pipeline(
    prompt,
    height=height,
    width=width,
    num_images_per_prompt=num_images,
).images[0]
```
<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/optimum/documentation-images/resolve/main/intel/openvino/stable_diffusion_v1_5_sail_boat_rembrandt.png">
</div>

คุณสามารถดูตัวอย่างเพิ่มเติมใน [เอกสาร](https://huggingface.co/docs/optimum/intel/inference#stable-diffusion) ของ 🤗 Optimum และ Stable Diffusion รองรับการทำงานสำหรับการแปลงข้อความเป็นภาพ (text-to-image), ภาพเป็นภาพ (image-to-image), และการทำซ่อมแซม (inpainting).

## Stable Diffusion XL

เพื่อโหลดและเรียกใช้ inference ด้วย SDXL, ให้ใช้ [`~optimum.intel.OVStableDiffusionXLPipeline`]:

```python
from optimum.intel import OVStableDiffusionXLPipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = OVStableDiffusionXLPipeline.from_pretrained(model_id)
prompt = "sailing ship in storm by Rembrandt"
image = pipeline(prompt).images[0]
```

เพื่อเร่งความเร็วของการ inference ต่อไป ให้ [เปลี่ยนรูปแบบเป็นแบบสถิต](#stable-diffusion) โมเดลตามที่แสดงในส่วนของ Stable Diffusion.

คุณสามารถดูตัวอย่างเพิ่มเติมใน [เอกสาร](https://huggingface.co/docs/optimum/intel/inference#stable-diffusion-xl) ของ 🤗 Optimum และการใช้งาน SDXL ใน OpenVINO รองรับการแปลงข้อความเป็นภาพและภาพเป็นภาพ.