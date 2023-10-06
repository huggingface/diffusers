<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# xFormers

เราแนะนำให้ใช้ [xFormers](https://github.com/facebookresearch/xformers) ทั้งสำหรับการอินเฟอเรนซ์และการฝึก. ในการทดสอบของเรา, การปรับปรุงที่ทำในบล็อกการให้ความสนใจช่วยเพิ่มความเร็วและลดการใช้หน่วยความจำ.

ติดตั้ง xFormers จาก `pip`:

```bash
pip install xformers
```

<Tip>

แพ็กเกจ xFormers จาก `pip` ต้องการเวอร์ชันล่าสุดของ PyTorch. หากคุณต้องการใช้เวอร์ชันก่อนหน้าของ PyTorch แนะนำให้ [ติดตั้ง xFormers จาก source](https://github.com/facebookresearch/xformers#installing-xformers).

</Tip>

หลังจากที่ xFormers ถูกติดตั้งแล้ว, คุณสามารถใช้ `enable_xformers_memory_efficient_attention()` เพื่อทำให้การทำนายเร็วขึ้นและลดการใช้หน่วยความจำดังที่แสดงใน [ส่วนนี้](memory#memory-efficient-attention).

<Tip warning={true}>

ตามข้อมูลจาก [ปัญหานี้](https://github.com/huggingface/diffusers/issues/2234#issuecomment-1416931212), xFormers `v0.0.16` ไม่สามารถใช้สำหรับการฝึก (fine-tune หรือ DreamBooth) บาง GPU ได้. หากคุณพบปัญหานี้, โปรดติดตั้งเวอร์ชันทดสอบตามที่ได้ระบุในความเห็นของปัญหานี้.

</Tip>