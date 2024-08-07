# إنشاء الصور باستخدام النص والعمق 

[[open-in-colab]]

تتيح وحدة [`StableDiffusionDepth2ImgPipeline`] تمرير موجه نصي وصورة أولية لتوجيه إنشاء الصور الجديدة. بالإضافة إلى ذلك، يمكنك أيضًا تمرير `depth_map` للحفاظ على بنية الصورة. إذا لم يتم توفير `depth_map`، فإن الأنبوب يتوقع العمق تلقائيًا عبر نموذج تقدير العمق المدمج [depth-estimation model](https://github.com/isl-org/MiDaS).

ابدأ بإنشاء مثيل من [`StableDiffusionDepth2ImgPipeline`]:

```python
import torch
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers.utils import load_image, make_image_grid

pipeline = StableDiffusionDepth2ImgPipeline.from_pretrained(
"stabilityai/stable-diffusion-2-depth",
torch_dtype=torch.float16,
use_safetensors=True,
).to("cuda")
```

الآن، قم بتمرير موجهك إلى الأنبوب. يمكنك أيضًا تمرير `negative_prompt` لمنع كلمات معينة من توجيه كيفية إنشاء الصورة:

```python
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
init_image = load_image(url)
prompt = "نمرين"
negative_prompt = "سيء، مشوه، قبيح، تشريح سيء"
image = pipeline(prompt=prompt, image=init_image, negative_prompt=negative_prompt, strength=0.7).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

| المدخلات                                                                           | المخرجات                                                                                                                                |
|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/coco-cats.png" width="500"/> | <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/depth2img-tigers.png" width="500"/> |