# AutoPipeline

🤗 Diffusers สามารถทำหลายงานที่แตกต่างกันได้ และคุณสามารถใช้ weights ที่ถูกเทรนล่วงหน้าเดียวกันสำหรับหลายงาน เช่น การแปลงข้อความเป็นรูปภาพ, การแปลงรูปภาพเป็นรูปภาพ, และการทำซ่อมแซมภาพ หากคุณเป็นมือใหม่ใน library และโมเดลอาจจะยากที่จะรู้ว่าควรใช้ไพป์ไลน์ใดสำหรับงาน ตัวอย่างเช่นถ้าคุณใช้ checkpoint [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) สำหรับการแปลงข้อความเป็นรูปภาพ คุณอาจจะไม่รู้ว่าคุณสามารถใช้ก็ได้สำหรับการแปลงรูปภาพเป็นรูปภาพและการทำซ่อมแซมภาพโดยโหลดจุดตรวจสอบด้วยคลาส [`StableDiffusionImg2ImgPipeline`] และ [`StableDiffusionInpaintPipeline`] ตามลำดับ

คลาส `AutoPipeline` ถูกออกแบบมาเพื่อทำให้การใช้ไพป์ไลน์ต่างๆ ใน 🤗 Diffusers ง่ายขึ้น มันเป็นไพป์ไลน์ทั่วไปที่ให้ความสำคัญกับ *งาน* ไพป์ไลน์ `AutoPipeline` จะตรวจจับโดยอัตโนมัติคลาสไพป์ไลน์ที่ถูกต้องที่จะใช้ ซึ่งทำให้ง่ายต่อการโหลด checkpoint สำหรับงานโดยไม่ต้องทราบชื่อคลาสไพป์ไลน์ที่เฉพาะเจาะจง

<Tip>

ดูที่ [AutoPipeline](./pipelines/auto_pipeline) เพื่อดูงานที่รองรับ ในปัจจุบันรองรับการแปลงข้อความเป็นรูปภาพ, การแปลงรูปภาพเป็นรูปภาพ และการทำซ่อมแซม

</Tip>

บทนี้จะแสดงให้คุณเห็นวิธีใช้ `AutoPipeline` เพื่อคาดการณ์คลาสไพป์ไลน์ที่จะโหลดสำหรับงานที่เฉพาะเจาะจงโดยใช้ weights ที่ถูกเทรนล่วงหน้า

## เลือกอโต้ไพป์ไลน์สำหรับงานของคุณ

เริ่มต้นด้วยการเลือก checkpoint เช่นถ้าคุณสนใจการแปลงข้อความเป็นรูปภาพด้วยจุดควบคุม [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) ใช้ [`AutoPipelineForText2Image`]:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
prompt = "peasant and dragon combat, wood cutting style, viking era, bevel with rune"

image = pipeline(prompt, num_inference_steps=25).images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-text2img.png" alt="generated image of peasant fighting dragon in wood cutting style"/>
</div>

ภายใน, [`AutoPipelineForText2Image`]:

1. ตรวจจับคลาส `"stable-diffusion"` จากไฟล์ [`model_index.json`](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/model_index.json)
2. โหลดคลาส [`StableDiffusionPipline`] ที่เป็นการแปลงข้อความเป็นรูปภาพที่เชื่อมโยงกับคลาส `"stable-diffusion"` ที่ระบุ

เช่นเดียวกันสำหรับการแปลงรูปภาพเป็นรูปภาพ, [`AutoPipelineForImage2Image`] ตรวจจับจุดตรวจสอบ `"stable-diffusion"` จากไฟล์ `model_index.json` และจะโหลดคลาส [`StableDiffusionImg2ImgPipeline`] ที่เกี่ยวข้องอัตโนมัติ เพื่อระบุได้เพิ่มเติมให้กับคลาสไพป์ไลน์ เช่น `strength` ซึ่งกำหนดปริมาณของ noise หรือการแปรผันที่เพิ่มเข้าไปในรูปภาพที่นำเข้า:

```py
from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")
prompt = "a portrait of a dog wearing a pearl earring"

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/1665_Girl_with_a_Pearl_Earring.jpg/800px-1665_Girl_with_a_Pearl_Earring.jpg"

response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")
image.thumbnail((768, 768))

image = pipeline(prompt, image, num_inference_steps=200, strength=0.75, guidance_scale=10.5).images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-img2img.png" alt="generated image of a vermeer portrait of a dog wearing a pearl earring"/>
</div>

และหากคุณต้องการทำซ่อมแซม, แล้ว [`AutoPipelineForInpainting`] จะโหลดคลาส [`StableDiffusionInpaintPipeline`] ที่อยู่ภายในเช่นเดียวกัน:

```py
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

pipeline = AutoPipelineForInpainting.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).convert("RGB")
mask_image = load_image(mask_url).convert("RGB")

prompt = "A majestic tiger sitting on a bench"
image = pipeline(prompt, image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80).images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-inpaint.png" alt="generated image of a tiger sitting on a bench"/>
</div>

หากคุณพยายามโหลด checkpoint ที่ไม่ได้รับการรองรับ จะโยน error:

```py
from diffusers import AutoPipelineForImage2Image
import torch

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "openai/shap-e-img2img", torch_dtype=torch.float16, use_safetensors=True
)
"ValueError: AutoPipeline can't find a pipeline linked to ShapEImg2ImgPipeline for None"
```

## ใช้ไพป์ไลน์หลายรายการ

สำหรับบางกระบวนการหรือหากคุณกำลังโหลดไพป์ไลน์หลายรายการ มันจะเป็นเพิ่มประสิทธิภาพทางหน่วยความจำมากกว่าที่จะใช้ส่วนประกอบเดียวกันจากจุดตรวจสอบแทนที่จะโหลดเพิ่มเติมซึ่งจะกินหน่วยความจำเพิ่มขึ้นโดยไม่จำเป็น เช่นถ้าคุณใช้จุดตรวจสอบสำหรับการแปลงข้อความเป็นรูปภาพและต้องการใช้ซ้ำสำหรับการแปลงรูปภาพเป็นรูปภาพ ใช้ [`~AutoPipelineForImage2Image.from_pipe`] วิธีนี้จะสร้างไพป์ไลน์ใหม่จากส่วนประกอบของไพป์ไลน์ที่โหลดไว้ก่อนหน้านี้โดยไม่มีค่าใช้จ่ายในหน่วยความจำเพิ่มขึ้น

วิธี [`~AutoPipelineForImage2Image.from_pipe`] จะตรวจจับคลาสไพป์ไลน์ต้นฉบับและทำการแมปไปยังคลาสไพป์ไลน์ใหม่ที่เกี่ยวข้องกับงานที่คุณต้องการทำ เช่น หากคุณโหลดคลาสไพป์ไลน์ `"stable-diffusion"` สำหรับการแปลงข้อความเป็นรูปภาพ:

```py
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

pipeline_text2img = AutoPipelineForText2Image.from_pretrained(
    "runwayml/st

able-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
print(type(pipeline_text2img))
"<class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'>"
```

จากนั้น [`~AutoPipelineForImage2Image.from_pipe`] จะทำการแมปคลาสไพป์ไลน์ `"stable-diffusion"` ต้นฉบับไปยัง [`StableDiffusionImg2ImgPipeline`]:

```py
pipeline_img2img = AutoPipelineForImage2Image.from_pipe(pipeline_text2img)
print(type(pipeline_img2img))
"<class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline'>"
```

หากคุณส่งอาร์กิวเมนต์ทางเลือก - เช่น ปิดการใช้ตัวตรวจสอบความปลอดภัย - ไปที่ไพป์ไลน์ต้นฉบับ อาร์กิวเมนต์นี้จะถูกส่งต่อไปยังไพป์ไลน์ใหม่:

```py
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

pipeline_text2img = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
    requires_safety_checker=False,
).to("cuda")

pipeline_img2img = AutoPipelineForImage2Image.from_pipe(pipeline_text2img)
print(pipe.config.requires_safety_checker)
"False"
```

คุณสามารถเขียนทับอาร์กิวเมนต์ใด ๆ และการกำหนดค่าจากไพป์ไลน์ต้นฉบับหากคุณต้องการเปลี่ยนพฤติกรรมของไพป์ไลน์ใหม่ เช่น เปิดตัวตรวจสอบความปลอดภัยอีกครั้งและเพิ่มอาร์กิวเมนต์ `strength`:

```py
pipeline_img2img = AutoPipelineForImage2Image.from_pipe(pipeline_text2img, requires_safety_checker=True, strength=0.3)
```
