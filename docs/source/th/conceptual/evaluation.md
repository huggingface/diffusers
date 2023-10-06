<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Evaluating Diffusion Models

<a target="_blank" href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/evaluation.ipynb">                                                                                                                                                                                                                                                                                                                                                            
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="เปิดใน Colab"/>                                                                                                                                                 
</a>   

การประเมินของโมเดลที่สร้างเนื้อหาเช่น [การ Diffusion ที่มีความเสถียร](https://huggingface.co/docs/diffusers/stable_diffusion) เป็นเรื่องจำเพาะธรรมเนียม แต่เป็นที่พึ่งพิงต่ออาชีพและนักวิจัย เราต้องทำการเลือกอย่างรอบคอบในหลายๆ ทางเลือกที่แตกต่างกัน ดังนั้น เมื่อทำงานกับโมเดลที่สร้างเนื้อหาที่แตกต่างกัน (เช่น GAN, Diffusion, เป็นต้น) เราควรเลือกรูปแบบไหนดี?

การประเมินคุณภาพของโมเดลเช่นนี้สามารถทำให้เกิดข้อผิดพลาดได้ และอาจทำให้มีผลต่อการตัดสินใจอย่างไม่ถูกต้อง อย่างไรก็ตาม การวัดค่าปริมาณไม่จำเป็นต้องสอดคล้องกับคุณภาพของภาพ ดังนั้น โดยทั่วไป การผสมผสานระหว่างการประเมินทางคุณภาพและปริมาณให้สัญญาณที่แข็งแกร่งมากขึ้นเมื่อต้องเลือกโมเดลที่ดีกว่าอันหนึ่งกับอีกอันหนึ่ง

ในเอกสารนี้ เราจะให้ภาพรวมที่ไม่ครอบคลุมอย่างไม่สมบูรณ์ของวิธีการ qualititative และ quantitiative เพื่อประเมินโมเดลการ Diffusion นอกจากนี้เราจะเน้นเฉพาะวิธีการ quantitiative โดยเฉพาะวิธีการที่ใช้ในการปรับใช้พร้อมกับ `diffusers`.

วิธีที่แสดงในเอกสารนี้สามารถนำไปใช้ในการประเมินตัวควบคุม [ตัวจัดการเสียง](https://huggingface.co/docs/diffusers/main/en/api/schedulers/overview) ที่เก็บรวมโมเดลการสร้างที่อยู่ด้านล่าง

## สถานการณ์

เราครอบคลุมโมเดล Diffusion ด้วยท่อ:

- การสร้างภาพที่มีการนำทางด้วยข้อความ (เช่น [`StableDiffusionPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/text2img)).
- การสร้างภาพที่มีการนำทางด้วยข้อความเพิ่มเติมที่ขึ้นอยู่กับภาพที่นำเข้า (เช่น [`StableDiffusionImg2ImgPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/img2img), และ [`StableDiffusionInstructPix2PixPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/pix2pix)).
- โมเดลการสร้างภาพที่ขึ้นอยู่กับชั้น (เช่น [`DiTPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/dit)).

## การประเมินทางคุณภาพ

การประเมินทางคุณภาพมักนั้นเป็นการประเมินโดยมนุษย์ของภาพที่สร้างขึ้น คุณภาพถูกวัดตามด้านเช่น ความเป็นสมรรถนะ การจับคู่ข้อความและภาพ และความสัมพันธ์ทางพื้นที่ ป้อนเสนอทั่วไปจะให้ความหมายทางคุณภาพที่ถูกต้อง
ชุดข้อความที่สามารถนำมาใช้ในการประเมินทางคุณภาพสามารถดึงมาจากการทำการวิจัยด้านการทดสอบคุณภาพของภาพเจนเนอเรทิฟแบบไดนามิกที่เหมาะสม

### ประเมินทางปริมาณ

การประเมินทางปริมาณมุ่งเน้นที่ความสัมพันธ์ระหว่างความยาวของข้อความกับความละเอียดของภาพที่สร้างขึ้น
การใช้มาตรการเชิงปริมาณสามารถช่วยให้เราเข้าใจว่าการปรับพารามิเตอร์ทำให้มีผลกับความแม่นยำของโมเดลอย่างไร

## ข้อสำคัญ

การประเมินควรพิจารณาข้อแม้บางประการที่สำคัญ:

1. **สำหรับการประเมินทางคุณภาพ**:
   - การมีผู้ทดสอบมนุษย์: การมีผู้ทดสอบมนุษย์ในการประเมินความถูกต้องและความน่าพึงพอใจของภาพที่สร้างขึ้น
   - ความหลากหลายของการทดสอบ: การใช้ชุดข้อมูลทดสอบที่หลากหลายเพื่อให้ได้ความคืบหน้าที่ครอบคลุม
   - การทดสอบความแปลกปลอม: การทดสอบโมเดลด้วยข้อมูลที่ไม่เคยเห็นมาก่อน เพื่อดูถึงความสามารถในการสร้างภาพใหม่

2. **สำหรับการประเมินทางปริมาณ**:
   - การใช้เครื่องมือการประเมิน: การใช้เครื่องมือทางคณิตศาสตร์เพื่อวัดความสัมพันธ์ระหว่างข้อความและภาพ
   - การใช้ค่าสถิติ: การวิเคราะห์ข้อมูลที่ได้ด้วยการใช้ค่าสถิติทางคณิตศาสตร์
   - การปรับใช้: การปรับพารามิเตอร์ของโมเดลและการทดสอบเพื่อเพิ่มประสิทธิภาพของการสร้าง

การประเมินความสามารถของโมเดลการ Diffusion เป็นขั้นตอนสำคัญเพื่อทำให้โมเดลนี้มีประสิทธิภาพมากขึ้น และเป็นประโยชน์ต่อผู้ใช้ที่สนใจในการสร้างภาพด้วยการ Diffusion.

[แดชบอร์ด Hugging Face](https://huggingface.co/) มีเครื่องมือและเนื้อหาที่สามารถใช้ช่วยในการประเมินนี้
