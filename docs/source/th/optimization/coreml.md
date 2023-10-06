<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# วิธีการรัน Stable Diffusion ด้วย Core ML

[Core ML](https://developer.apple.com/documentation/coreml) เป็นรูปแบบโมเดลและไลบรารีเรียนรู้เชิงปัญญาเทียบกับ Apple frameworks หากคุณสนใจที่จะรันโมเดล Stable Diffusion ในแอป macOS หรือ iOS/iPadOS ของคุณ คู่มือนี้จะแสดงวิธีการแปลงจุดตรวจสอบ PyTorch ที่มีอยู่เป็นรูปแบบ Core ML และใช้ในการทำนายด้วย Python หรือ Swift

Core ML สามารถใช้ประโยชน์จากเครื่องมือคำนวณทั้งหมดที่มีอยู่ในอุปกรณ์ Apple: CPU, GPU, และ Apple Neural Engine (หรือ ANE, ตัวเร่งที่ถูกปรับเพื่อเส้นทางเฉพาะที่พร้อมใช้งานใน Macs และ iPhones/iPads รุ่นใหม่). ขึ้นอยู่กับโมเดลและอุปกรณ์ที่โมเดลกำลังทำงานอยู่ Core ML สามารถผสานและจับคู่เครื่องมือคำนวณได้เช่นกัน ดังนั้นบางส่วนของโมเดลอาจทำงานบน CPU ในขณะที่อื่นๆ ทำงานบน GPU, เป็นต้น

<Tip>

คุณยังสามารถรันโค้ด Python `diffusers` บน Macs ที่ใช้ชิพ Apple Silicon โดยใช้ตัวเร่ง `mps` ที่มีอยู่ใน PyTorch การเข้าใช้วิธีนี้อธิบายอย่างละเอียดใน [คู่มือ mps](mps) แต่มันไม่สามารถทำงานได้ร่วมกับแอปหลัก

</Tip>

## จุดตรวจสอบ Core ML ของ Stable Diffusion

Weights ของ Stable Diffusion ถูกเก็บไว้ในรูปแบบ PyTorch ดังนั้นคุณต้องแปลงเป็นรูปแบบ Core ML ก่อนที่เราจะสามารถใช้กับแอป

โดยดีที่มีนักวิศวกร Apple พัฒนา [เครื่องมือแปลง](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml) ที่พึ่งได้จาก `diffusers` เพื่อแปลงจุดตรวจสอบ PyTorch เป็น Core ML

ก่อนที่คุณจะแปลงโมเดล ให้สำรวจ Hugging Face Hub ก่อนสิ่งที่คุณสนใจอาจจะมีให้ในรูปแบบ Core ML อยู่แล้ว:

- องค์กร [Apple](https://huggingface.co/apple) รวมถึง Stable Diffusion เวอร์ชัน 1.4, 1.5, 2.0 base และ 2.1 base
- องค์กร [coreml](https://huggingface.co/coreml) รวมถึง DreamBoothed และโมเดลที่ปรับปรุงเอง
- ใช้ [ตัวกรองนี้](https://huggingface.co/models?pipeline_tag=text-to-image&library=coreml&p=2&sort=likes) เพื่อดึงคืนจุดตรวจสอบ Core ML ทั้งหมดที่มีอยู่

หากคุณไม่พบโมเดลที่คุณสนใจ เราขอแนะนำให้ทำตามคำแนะนำสำหรับการ [แปลงโมเดลเป็น Core ML](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml) โดย Apple

## เลือก Core ML Variant ที่จะใช้

โมเดล Stable Diffusion สามารถแปลงเป็น Core ML variants ที่แตกต่างกันสำหรับวัตถุประสงค์ที่แตกต่างกัน:

- ประเภทของบล็อกการให้ความสนใจที่ใช้ การดำเนินการให้ความสนใจถูกใช้เพื่อ "ให้ความสนใจ" ในความสัมพันธ์ระหว่างพื้นที่ต่าง ๆ ในการแสดงภาพ และเครื่องมือในการเข้าใจว่า

ภาพและข้อความมีความเกี่ยวข้องกัน การให้ความสนใจเป็นการคำนวณและใช้หน่วยความจำเป็นมาก เพื่อสร้างการให้ความสนใจที่แตกต่างกันตามลักษณะของอุปกรณ์ต่าง ๆ Core ML Stable Diffusion มี variants การให้ความสนใจสองประเภท:
    * `split_einsum` ([นำเสนอโดย Apple](https://machinelearning.apple.com/research/neural-engine-transformers)) ถูกปรับให้เหมาะกับอุปกรณ์ ANE ซึ่งมีให้ใน iPhones, iPads รุ่นใหม่ และคอมพิวเตอร์ M-series
    * การให้ความสนใจ "เดิม" (การปฏิบัติการฐานที่ใช้ใน `diffusers`) เพียงเพียงเข้ากันได้กับ CPU/GPU และไม่ใช่ ANE การรันโมเดลบน CPU + GPU โดยใช้ `การให้ความสนใจเดิม` อาจจะเร็วกว่าการใช้ ANE ดู [เบนช์มาร์คเพิร์ฟอร์มันภายใน](https://huggingface.co/blog/fast-mac-diffusers#performance-benchmarks) รวมถึงบางการวัดเพิ่มเติมที่มีคนในชุมชนได้รับการให้มากขึ้น [โดยทางชุมชน](https://github.com/huggingface/swift-coreml-diffusers/issues/31) สำหรับรายละเอียดเพิ่มเติม

- กรอบการทำนายที่รองรับ
    * แพ็คเกจ (packages) เหมาะสำหรับการทำนายใน Python นี้สามารถใช้เพื่อทดสอบโมเดล Core ML ที่แปลงก่อนที่จะพยายามรวมเข้ากับแอปหลัก หรือหากคุณต้องการสำรวจประสิทธิภาพ Core ML แต่ไม่จำเป็นต้องรองรับแอปหลัก ตัวอย่างเช่น แอปพลิเคชันที่มี UI บนเว็บสามารถใช้แบ็กเอนด์ Core ML ใน Python
    * โมเดลที่คอมไพล์ (compiled) จำเป็นสำหรับโค้ด Swift โมเดลที่ได้รับการคอมไพล์ใน Hub แบ่งน้ำหนักของโมเดล UNet ใหญ่เป็นไฟล์หลายๆ ไฟล์เพื่อให้เข้ากันได้กับอุปกรณ์ iOS และ iPadOS นี้สอดคล้องกับ [`--chunk-unet` ตัวเลือกการแปลง](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml) หากคุณต้องการรองรับแอปหลัก คุณจำเป็นต้องเลือกตัวแปร `compiled`

[โมเดล Core ML Stable Diffusion](https://huggingface.co/apple/coreml-stable-diffusion-v1-4/tree/main) อยู่ในรูปแบบนี้ แต่อาจมีความแตกต่างกันขึ้นอยู่กับชุมชน:

```
coreml-stable-diffusion-v1-4
├── README.md
├── original
│   ├── compiled
│   └── packages
└── split_einsum
    ├── compiled
    └── packages
```

คุณสามารถดาวน์โหลดและใช้ตัวแปรที่คุณต้องการได้ตามที่แสดงด้านล่าง

## Core ML Inference ใน Python

ติดตั้งไลบรารีต่อไปนี้เพื่อรันการทำนาย Core ML ใน Python:

```bash
pip install huggingface_hub
pip install git+https://github.com/apple/ml-stable-diffusion
```

### ดาวน์โหลด checkpoints โมเดล

เพื่อรันการทำนายใน Python ให้ใช้หนึ่งในเวอร์ชันที่เก็บไว้ในโฟลเดอร์ `packages` เนื่องจากเวอร์ชัน `compiled` เหมาะกับ Swift เท่านั้น คุณสามารถเลือกที่จะใช้การให้ความสนใจ `original` หรือ `split_einsum`

นี่คือวิธีการดาวน์โหลดตัวแปรการให้ความสนใจ `original` จาก Hub ไปยังไดเรกทอรีที่ชื่อ `models`:

```Python
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "apple/coreml-stable-diffusion-v1-4"
variant = "original/packages"

model_path = Path("./models") / (repo_id.split("/")[-1] + "_" + variant.replace("/", "_"))
snapshot_download(repo_id, allow_patterns=f"{variant}/*", local_dir=model_path, local_dir_use_symlinks=False)
print(f"Model downloaded at {model_path}")
```

### การทำนาย[[python-inference]]

เมื่อคุณดาวน์โหลดสแนปช็อตของโมเดลแล้ว คุณสามารถทดสอบได้โดยใช้สคริปต์ Python ของ Apple

```shell
python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" -i models/coreml-stable-diffusion-v1-4_original_packages -o </path/to/output/image> --compute-unit CPU_AND_GPU --seed 93
```

`<output-mlpackages-directory>` ควรชี้ไปที่จุดตรวจสอบที่คุณดาวน์โหลดในขั้นตอนก่อนหน้า และ `--compute-unit` ระบุฮาร์ดแวร์ที่คุณต้องการให้ทำนาย ต้องเป็นหนึ่งในตัวเลือกเหล่านี้: `ALL`, `CPU_AND_GPU`, `CPU_ONLY`, `CPU_AND_NE` คุณยังสามารถระบุเส้นทางเอาท์พุททางเลือกได้ตามต้องการ และเลือกเมล็ดพันธุ์สำหรับการทำให้เล่นซ้ำได้

สคริปต์การทำนายถูกแปลว่าคุณกำลังใช้เวอร์ชันเดิมของโมเดล Stable Diffusion คือ `CompVis/stable-diffusion-v1-4` หากคุณใช้โมเดลอื่น เคาะ *ต้อง* ระบุ id Hub ของโมเดลในบรรทัดคำสั่งการทำนาย โดยใช้ตัวเลือก `--model-version` นี้ทำงานสำหรับโมเดลที่ได้รับการสนับสนุนและโมเดลที่ปรับปรุงเอง

ตัวอย่างเช่น หากคุณต้องการใช้ [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5):

```shell
python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" --compute-unit ALL -o output --seed 93 -i models/coreml-stable-diffusion-v1-5_original_packages --model-version runwayml/stable-diffusion-v1-5
```

## Core ML inference ใน Swift

การรันการทำนายใน Swift จะเร็วกว่าน้อยเนื่องจากโมเดลได้ถูกคอมไพล์ไปแล้วในรูปแบบ `mlmodelc` ซึ่งสังเหตุได้ในการเริ่มต้นแอปเมื่อโมเดลถูกโหลด แต่ไม่ควรสังเกตเห็นถึงเมื่อคุณรันการสร้างหลายรุ่นหลังจากนั้น

### การดาวน์โหลด

เพื่อรันการทำนายใน Swift บน Mac ของคุณ คุณต้องใช้หนึ่งในตัวแปรที่ถูกคอมไพล์เช็คพอยท์เวอร์ชัน `compiled` เราขอแนะนำให้คุณดาวน์โหลดไฟล์เหล่านี้ไปยังเครื่องโดยใช้โค้ด Python ที่คล้ายกับตัวอย่างก่อนหน้า แต่ใช้ตัวแปร `compiled` หนึ่งในตัวเลือก:

```Python
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "apple/coreml-stable-diffusion-v1-4"
variant = "original/compiled"

model_path = Path("./models") / (repo_id.split("/")[-1] + "_" + variant.replace("/", "_"))
snapshot_download(repo_id, allow_patterns=f"{variant}/*", local_dir=model_path, local_dir_use_symlinks=False)
print(f"Model downloaded at {model_path}")
```

### การทำนาย[[swift-inference]]

เพื่อรันการทำนาย โปรดคล๊อนเรพอ Apple's repo:

```bash
git clone https://github.com/apple/ml-stable-diffusion
cd ml-stable-diffusion
```

แล้วใช้เครื่องมือของ Apple คือ [Swift Package Manager](https://www.swift.org/package-manager/#):

```bash
swift run StableDiffusionSample --resource-path models/coreml-stable-diffusion-v1-4_original_compiled --compute-units all "a photo of an astronaut riding a horse on mars"
```

คุณต้องระบุเส้นทางไปยังโมเดล Stable Diffusion ที่คุณดาวน์โหลดในขั้นตอนก่อนหน้า

เปลี่ยน `all` เป็น `cpu` หรือ `gpu` หากคุณต้องการให้ทำนายบนฮาร์ดแวร์เฉพาะ ในกรณีนี้ Core ML จะโปร่งใสในการเลือกเครื่องมือคำนวณและจะรวดเร็วขึ้นในกรณีที่อุปกรณ์มี GPU ที่ทำงานดี

สำหรับอ้างอิงเพิ่มเติมเกี่ยวกับวิธีการรันการทำนายใน Swift ดูที่ [`swift-coreml-diffusers`](https://github.com/huggingface/swift-coreml-diffusers)

## สรุป

นี้คือวิธีการแปลงจุดตรวจสอบ Stable Diffusion จาก PyTorch เป็น Core ML และวิธีการทำนายใน Python และ Swift โดยใช้ Core ML ของ Apple หากคุณต้องการให้แอปของคุณสามารถทำนายด้วย Stable Diffusion ทั้งหมดนี้ ติดต่อคู่ค้าหรือนักพัฒนาของ Apple เพื่อความช่วยเหลือเพิ่มเติมและรายละเอียดเกี่ยวกับการปรับปรุงหรือปรับปรุงที่เป็นไปได้

หมายเหตุ: บทคู่มือนี้อธิบายการใช้ Stable Diffusion ตัวย่อในแอป Core ML และการทำนาย Python/Swift สำหรับความสะดวกในการใช้งาน อย่าลืมว่าความสามารถและประสิทธิภาพของโมเดล Core ML ที่ได้จะขึ้นอยู่กับการใช้งานและอุปกรณ์ของคุณ