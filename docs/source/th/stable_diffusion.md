<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
                                                               
# เพิ่มประสิทธิภาพให้กับ Diffusion

[[open-in-colab]]

ในบทนี้ เราจะมาดูวิธีการเจน samples ให้เร็วขึ้นระหว่างการใช้งาน [`DiffusionPipeline`].

เริ่มจากการโหลด [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) model:

```python
from diffusers import DiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
```

เราจะใช้ prompt ตัวอย่างต่อไปนี้ แต่คุณอาจลอง prompt อื่นๆ ได้เช่นกัน:

```python
prompt = "portrait photo of a old warrior chief"
```

## Speed

<Tip>

💡 หากคุณไม่มี GPU ลองใช้งานบริการของ [Colab](https://colab.research.google.com/) ดูสิ!

</Tip>

วิธีที่ง่ายที่สุดเพื่อเร่ง inference คือการย้าย pipeline ไปบน GPU:

```python
pipeline = pipeline.to("cuda")
```

หากต้องการเจนรูปเดียวกันหลายครั้งๆ ให้ตั้ง seed กับ [`Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) (อ่านเพิ่มเติมได้ที่ [reproducibility](./using-diffusers/reproducibility)):

```python
import torch

generator = torch.Generator("cuda").manual_seed(0)
```

และเจนรูปได้โดยการเรียก:

```python
image = pipeline(prompt, generator=generator).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_1.png">
</div>

ขั้นตอนทั้งหมดนี้ใช้เวลา ~30 วินาทีบน T4 GPU ซึ่ง [`DiffusionPipeline`] จะรัน inference ด้วย `float32` precision เป็นทั้งหมด 50 inference steps หมายความว่าเราสามารถลดเวลาที่ใช้โดยการสลับไปใช้ `float16` หรือลด inference steps 

เรามาลองสลับไปใช้ `float16` กันดู:

```python
import torch

pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
pipeline = pipeline.to("cuda")
generator = torch.Generator("cuda").manual_seed(0)
image = pipeline(prompt, generator=generator).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_2.png">
</div>

คราวนี้ใช้เวลาเพียง ~11 วินาที ซึ่งเร็วกว่ารอบก่อนหน้าประมาณตั้ง 3 เท่า!

<Tip>

💡 เราแนะนำให้ใช้ `float16` สำหรับการรัน inference เพราะไม่ค่อยส่งผลต่อคุณภาพของผลลัพธ์ซักเท่าไหร่

</Tip>

อีกวิธีหนึ่งในการเร่งความเร็วคือการลด inference steps ซึ่งจะส่งผลต่อคุณภาพของผลลัพธ์ แต่คุณสามารถลองเปลี่ยน schedulers เป็นอันที่ใหม่กว่าเพื่อไม่ให้คุณภาพถูกกระทบมากนัก สามารถเช็ค schedulers ที่ใช้งานร่วมด้วยกับ [`DiffusionPipeline`] โดยการเรียก `compatibles`:

```python
pipeline.scheduler.compatibles
[
    diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler,
    diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler,
    diffusers.schedulers.scheduling_k_dpm_2_discrete.KDPM2DiscreteScheduler,
    diffusers.schedulers.scheduling_deis_multistep.DEISMultistepScheduler,
    diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler,
    diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler,
    diffusers.schedulers.scheduling_ddpm.DDPMScheduler,
    diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler,
    diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteScheduler,
    diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler,
    diffusers.schedulers.scheduling_pndm.PNDMScheduler,
    diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler,
    diffusers.schedulers.scheduling_ddim.DDIMScheduler,
]
```

[`PNDMScheduler`] เป็นค่าเริ่มต้นของ Stable Diffusion pipeline ซึ่งจำเป็นต้องใช้ประมาณ 50 inference steps แต่ schedulers ที่ใหม่กว่าอย่าง [`DPMSolverMultistepScheduler`] จำเป็นต้องใช้เพียง ~20 หรือ 25 inference steps ให้ใช้ [`ConfigMixin.from_config`] ในการโหลด scheduler ใหม่:

```python
from diffusers import DPMSolverMultistepScheduler

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
```

เปลี่ยน `num_inference_steps` เป็น 20:

```python
generator = torch.Generator("cuda").manual_seed(0)
image = pipeline(prompt, generator=generator, num_inference_steps=20).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_3.png">
</div>

ยอดไปเลย! เราตัดเวลาที่ใช้เจนภาพเหลือเพียง 4 วินาทีได้แล้ว ⚡️

## Memory

อีกส่วนหนึ่งที่สำคัญในการใช้งาน pipeline คือปริมาณของ memory ที่ใช้งาน ในส่วนนี้เราจะมาลองลด memory ที่จำเป็นต้องใช้กัน

สร้าง function ที่เจนรูปเป็น batch จาก prompts หลายๆตัว และอย่าลืมป้อน seed ให้กับ `Generator` เพื่อสร้างผลลัพธ์ซ้ำในอนาคตได้ด้วยล่ะ

```python
def get_inputs(batch_size=1):
    generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
    prompts = batch_size * [prompt]
    num_inference_steps = 20

    return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}
```

เริ่มจาก `batch_size=4` และสังเกตปริมาณ memory ที่ใช้งานดู:

```python
from diffusers.utils import make_image_grid 

images = pipeline(**get_inputs(batch_size=4)).images
make_image_grid(images, 2, 2)
```

เว้นว่าคุณมี GPU ที่ปริมาณ VRAM มาก คุณควรเจอกับ `OOM` error เพราะ memory ส่วนมากถูกกินไปด้วย cross-attention layers ซึ่งเติบโตเป็นแบบเอ็กซ์โปเนนเชียล คุณสามารถแก้ปัญหานี้ได้โดยการใช้งาน [`~DiffusionPipeline.enable_attention_slicing`] function:

```python
pipeline.enable_attention_slicing()
```

หลังจากนั้นลองเปลี่ยน `batch_size` เป็น 8 ดูสิ!

```python
images = pipeline(**get_inputs(batch_size=8)).images
make_image_grid(images, rows=2, cols=4)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_5.png">
</div>

ตอนนี้คุณสามารถเจน 8 รูปพร้อมๆกันในเพียง ~3.5 วินาทีต่อรูปบน T4 GPU ได้แล้ว

## Quality

หลังจากที่เราเร่งความเร็วของ pipeline ของเราได้แล้ว เรามาดูวิธีการเพิ่มคุณภาพของผลลัพธ์กัน

### Better checkpoints

ขั้นตอนที่ชัดเจนที่สุดคือการใช้ตรวจสอบข้อมูลเช็คพ้อยท์ที่ดีกว่า โมเดล Stable Diffusion เป็นจุดเริ่มต้นที่ดี และตั้งแต่การเปิดตัวอย่างเป็นทางการได้มีการปล่อยเวอร์ชันที่ดีขึ้นมาหลายเวอร์ชัน อย่างไรก็ตาม การใช้เวอร์ชันใหม่ไม่ได้หมายความว่าคุณจะได้ผลลัพธ์ที่ดีขึ้นโดยอัตโนมัติ คุณจะต้องทดลองกับเช็คพ้อยท์ที่แตกต่างกันด้วยตนเอง และทำการค้นคว้าเล็กน้อย (เช่นใช้ [negative prompts](https://minimaxir.com/2022/11/stable-diffusion-negative-prompt/)) เพื่อให้ได้ผลลัพธ์ที่ดีที่สุด

เนื่องจากฟิลด์นี้กำลังเติบโตมีเช็คพ้อยท์ที่มีคุณภาพสูงมากขึ้นที่ถูกปรับแต่งเพื่อสร้างสไตล์ที่แน่นอน ลองสำรวจ [Hub](https://huggingface.co/models?library=diffusers&sort=downloads) และ [Diffusers Gallery](https://huggingface.co/spaces/huggingface-projects/diffusers-gallery) เพื่อหาตัวที่คุณสนใจ!

### Better pipeline components

คุณยังสามารถลองแทนที่คอมโพเนนต์ของไพป์ไลน์ปัจจุบันด้วยเวอร์ชันใหม่ ลองโหลด [autodecoder](https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main/vae) ล่าสุดจาก Stability AI เข้าสู่ไพป์ไลน์ และสร้างภาพบางภาพ:

```python
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
pipeline.vae = vae
images = pipeline(**get_inputs(batch_size=8)).images
make_image_grid(images, rows=2, cols=4)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_6.png">
</div>

### Better prompt engineering

ข้อความที่คุณใช้เพื่อสร้างภาพมีความสำคัญมาก ถึงขนาดที่มีชื่อเรียกว่า *prompt engineering* (วิศวกรรมพรอมป์) บางข้อพิจารณาในขณะที่ทำ *prompt engineering* คือ:

- ภาพหรือภาพที่คล้ายกับภาพที่ฉันต้องการสร้างถูกเก็บไว้บนอินเทอร์เน็ตอย่างไร?
- ฉันสามารถให้รายละเอียดเพิ่มเติมได้อย่างไรเพื่อนำทางโมเดลให้เป็นสไตล์ที่ฉันต้องการ?

โดยมีความคิดเหล่านี้ในเวลาเดียวกัน เพื่อปรับปรุง *prompt* เพื่อรวมสีและรายละเอียดที่มีคุณภาพสูง:

```python
prompt += ", tribal panther make up, blue on red, side profile, looking away, serious eyes"
prompt += " 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta"
```

สร้างชุดภาพด้วย *prompt* ใหม่:

```python
images = pipeline(**get_inputs(batch_size=8)).images
make_image_grid(images, rows=2, cols=4)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_7.png">
</div>

น่าประทับใจมาก! ขอปรับปรุงภาพที่สอง - ซึ่งสอดคล้องกับ `Generator` ด้วย `seed` เลข `1` - อีกนิดหน่อยโดยการเพิ่มข้อความเกี่ยวกับอายุของเรื่องราว:

```python
prompts = [
    "portrait photo of the oldest warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    "portrait photo of a old warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    "portrait photo of a warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    "portrait photo of a young warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
]

generator = [torch.Generator("cuda").manual_seed(1) for _ in range(len(prompts))]
images = pipeline(prompt=prompts, generator=generator, num_inference_steps=25).images
make_image_grid(images, 2, 2)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_8.png">
</div>

## Next steps

ในบทแนะนำนี้ คุณได้เรียนรู้วิธีการปรับแต่ง [`DiffusionPipeline`] เพื่อเพิ่มประสิทธิภาพทางคำนวณและหน่วยความจำ รวมถึงการปรับปรุงคุณภาพของผลลัพธ์ที่สร้างขึ้นมา หากคุณสนใจที่จะทำให้ไพป์ไลน์ของคุณเร็วขึ้นได้อีกต่อไป ลองดูทรัพยากรต่อไปนี้:

- เรียนรู้ว่า [PyTorch 2.0](./optimization/torch2.0) และ [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) สามารถทำให้การทำนายเร็วขึ้น 5 - 300% ได้ บน GPU A100 การทำนายสามารถเร็วขึ้นได้ถึง 50%!
- หากคุณไม่สามารถใช้ PyTorch 2, เราขอแนะนำให้คุณติดตั้ง [xFormers](./optimization/xformers) หลักการทำงานของ memory-efficient attention ที่ประหยัดหน่วยความจำทำงานได้ดีกับ PyTorch 1.13.1 เพื่อความเร็วที่เร็วขึ้นและการใช้หน่วยความจำลดลง
- เทคนิคการปรับปรุงอื่น ๆ เช่น การโอนโมเดลถูกนำเสนอใน [คู่มือนี้](./optimization/fp16)