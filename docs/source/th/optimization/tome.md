<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# การผสานโทเคน

[การผสานโทเคน (Token merging)](https://huggingface.co/papers/2303.17604) (ToMe) ผสานโทเคน/พัทช์ที่ไม่จำเป็นเรื่อย ๆ ใน forward pass ของเครือข่ายทรานส์ฟอร์เมอร์ (Transformer-based network) ซึ่งสามารถเพิ่มความเร็วในการทำนายของ [`StableDiffusionPipeline`].

คุณสามารถใช้ ToMe จากไลบรารี [`tomesd`](https://github.com/dbolya/tomesd) ด้วยฟังก์ชัน [`apply_patch`](https://github.com/dbolya/tomesd?tab=readme-ov-file#usage):

```diff
from diffusers import StableDiffusionPipeline
import tomesd

pipeline = StableDiffusionPipeline.from_pretrained(
      "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True,
).to("cuda")
+ tomesd.apply_patch(pipeline, ratio=0.5)

image = pipeline("a photo of an astronaut riding a horse on mars").images[0]
```

ฟังก์ชัน `apply_patch` จะเปิดเผยจำนวน [อาร์กิวเมนต์](https://github.com/dbolya/tomesd#usage) เพื่อช่วยให้สมดุลระหว่างความเร็วในการทำนายของ pipeline และคุณภาพของโทเคนที่สร้างขึ้นได้ อาร์กิวเมนต์ที่สำคัญที่สุดคือ `ratio` ซึ่งควบคุมจำนวนโทเคนที่ถูกผสานรวมระหว่าง forward pass.

ตามที่รายงานไว้ใน [เอกสารวิจัย](https://huggingface.co/papers/2303.17604) โทเคนที่ผ่านการผสานรวมโดย ToMe สามารถรักษาคุณภาพของรูปภาพที่สร้างขึ้นได้ พร้อมทั้งเพิ่มความเร็วในการทำนายเช่นกัน โดยเพิ่มค่าของ `ratio` คุณสามารถเพิ่มความเร็วในการทำนายได้อีกมาก แต่จะเสียคุณภาพของรูปภาพบ้างเล็กน้อย.

เพื่อทดสอบคุณภาพของรูปภาพที่สร้างขึ้น เราได้สุ่มเลือก prompt จาก [Parti Prompts](https://parti.research.google/) และทำการทำนายด้วย [`StableDiffusionPipeline`] ด้วยการตั้งค่าต่อไปนี้:

<div class="flex justify-center">
      <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/tome/tome_samples.png">
</div>

เราไม่รู้สึกถึงการลดคุณภาพของตัวอย่างที่สร้างขึ้นอย่างมีนัยสำคัญ และคุณสามารถดูตัวอย่างที่สร้างขึ้นได้ใน [รายงาน WandB](https://wandb.ai/sayakpaul/tomesd-results/runs/23j4bj3i?workspace=) หากคุณสนใจที่จะทำการทดลองทำซ้ำการทดลองนี้ ให้ใช้ [สคริปต์นี้](https://gist.github.com/sayakpaul/8cac98d7f22399085a060992f411ecbd).

## การทดสอบผล

เราได้ทำการทดสอบประสิทธิผลของ `tomesd` กับ [`StableDiffusionPipeline`] ที่เปิดใช้ [xFormers](https://huggingface.co/docs/diffusers/optimization/xformers) ที่มีการเปิดใช้ในหลายขนาดของภาพ ผลลัพธ์ได้มาจาก GPU A100 และ V100 ในสภาพแวดล้อมการพัฒนาต่อไปนี้:

```bash
- เวอร์ชัน `diffusers`: 0.15.1
- เวอร์ชัน Python: 3.8.16
- เวอร์ชัน PyTorch (GPU?): 1.13.1+cu116 (True)
- เวอร์ชัน Huggingface Hub: 0.13.2
- เวอร์ชัน Transformers: 4.27.2
- เวอร์ชัน Accelerate: 0.18.0
- เวอร์ชัน xFormers: 0.0.16
- เวอร์ชัน tomesd: 0.1.2
```

หากคุณต้องการทำการทดสอบประสิทธิผลนี้อีกครั้ง คุณสามารถใช้ [สคริปต์นี้](https://gist.github.com/sayakpaul/27aec6bca7eb7b0e0aa4112205850335) ผลลัพธ์ถูกรายงานเป็นวินาที และเมื่อเกี่ยวข้องเรารายงานเปอร์เซ็นต์ความเร็วเพิ่มขึ้นเมื่อใช้ ToMe และ ToMe + xFormers เมื่อเทียบกับ pipeline ปกติ.

| **GPU**  | **ขนาดภาพ** | **ขนาดชุด** | **ปกติ** | **ToMe**       | **ToMe + xFormers** |
|----------|----------------|----------------|-------------|----------------|---------------------|
| **A100** |            512 |             10 |        6.88 | 5.26 (+23.55%) |      4.69 (+31.83%) |
|          |            768 |             10 |         OOM |          14.71 |                  11 |
|          |                |              8 |         OOM |          11.56 |                8.84 |
|          |                |              4 |         OOM |           5.98 |                4.66 |
|          |                |              2 |        4.99 | 3.24 (+35.07%) |       2.1 (+37.88%) |
|          |                |              1 |        3.29 | 2.24 (+31.91%) |       2.03 (+38.3%) |
|          |           1024 |             10 |         OOM |            OOM |                 OOM |
|          |                |              8 |         OOM |            OOM |                 OOM |
|          |                |              4 |         OOM |          12.51 |                9.09 |
|          |                |              2 |         OOM |           6.52 |                4.96 |
|          |                |              1 |         6.4 | 3.61 (+43.59%) |      2.81 (+56.09%) |
| **V100** |            512 |             10 |         OOM |          10.03 |                9.29 |
|          |                |              8 |         OOM |           8.05 |                7.47 |
|          |                |              4 |         5.7 |  4.3 (+24.56%) |      3.98 (+30.18%) |
|          |                |              2 |        3.14 | 2.43 (+22.61%) |      2.27 (+27.71%) |
|          |                |              1 |        1.88 | 1.57 (+16.49%) |      1.57 (+16.49%) |
|          |            768 |             10 |         OOM |            OOM |               23.67 |
|          |                |              8 |         OOM |            OOM |               18.81 |
|          |                |              4 |         OOM |          11.81 |                 9.7 |
|          |                |              2 |         OOM |           6.27 |                 5.2 |
|          |                |              1 |        5.43 | 3.38 (+37.75%) |      2.82 (+48.07%) |
|          |           1024 |             10 |         OOM |            OOM |                 OOM |
|          |                |              8 |         OOM |            OOM |                 OOM |
|          |                |              4 |         OOM |            OOM |               19.35 |
|          |                |              2 |         OOM |             13 |               10.78 |
|          |                |              1 |         OOM |           6.66 |                5.54 |

จากตารางข้างต้นเราสามารถเห็นว่า การเพิ่มความเร็วด้วย `tomesd` มีผลมากขึ้นเมื่อขนาดภาพมีขนาดใหญ่ขึ้น นอกจากนี้ยังสังเกตได้ว่า ด้วย `tomesd` เราสามารถให้ pipeline ทำงานกับขนาดภาพที่ใหญ่ขึ้นได้อีก คุณสามารถเพิ่มความเร็วในการทำนายมากขึ้นได้ด้วย [`torch.compile`](torch2.0).
