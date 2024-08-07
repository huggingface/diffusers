# Shap-E

[[open-in-colab]]

Shap-E هو نموذج شرطي لتوليد أصول ثلاثية الأبعاد يمكن استخدامها في تطوير ألعاب الفيديو وتصميم الديكور الداخلي والهندسة المعمارية. تم تدريبه على مجموعة بيانات كبيرة من الأصول ثلاثية الأبعاد، وتمت معالجتها بعد ذلك لإنشاء المزيد من وجهات النظر لكل كائن وإنتاج سحب نقاط 16K بدلاً من 4K. يتم تدريب نموذج Shap-E في خطوتين:

1. يقبل مشفر سحب النقاط والصور التي تم عرضها لأصل ثلاثي الأبعاد ويخرج معلمات الدوال الضمنية التي تمثل الأصل.
2. يتم تدريب نموذج الانتشار على المخفيات التي ينتجها المشفر لتوليد حقول الإشعاع العصبية (NeRFs) أو شبكة ثلاثية الأبعاد ذات نسيج، مما يجعل من السهل تصيير الأصل ثلاثي الأبعاد واستخدامه في التطبيقات اللاحقة.

سيوضح هذا الدليل كيفية استخدام Shap-E لبدء إنشاء أصول ثلاثية الأبعاد الخاصة بك!

قبل أن تبدأ، تأكد من تثبيت المكتبات التالية:

```py
# قم بإلغاء التعليق لتثبيت المكتبات الضرورية في Colab
#! pip install -q diffusers transformers accelerate trimesh
```

## نص إلى 3D

لإنشاء صورة متحركة لكائن ثلاثي الأبعاد، قم بتمرير موجه نصي إلى [`ShapEPipeline`]. يقوم الأنبوب بتوليد قائمة من إطارات الصور التي تستخدم لإنشاء الكائن ثلاثي الأبعاد.

```py
import torch
from diffusers import ShapEPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to(device)

guidance_scale = 15.0
prompt = ["A firecracker", "A birthday cupcake"]

images = pipe(
    prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=64,
    frame_size=256,
).images
```

الآن استخدم وظيفة [`~utils.export_to_gif`] لتحويل قائمة إطارات الصور إلى صورة GIF للكائن ثلاثي الأبعاد.

```py
from diffusers.utils import export_to_gif

export_to_gif(images[0], "firecracker_3d.gif")
export_to_gif(images[1], "cake_3d.gif")
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/shap_e/firecracker_out.gif"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">prompt = "A firecracker"</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/shap_e/cake_out.gif"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">prompt = "A birthday cupcake"</figcaption>
  </div>
</div>


## صورة إلى 3D

لإنشاء كائن ثلاثي الأبعاد من صورة أخرى، استخدم [`ShapEImg2ImgPipeline`]. يمكنك استخدام صورة موجودة أو إنشاء صورة جديدة تمامًا. دعنا نستخدم نموذج [Kandinsky 2.1](../api/pipelines/kandinsky) لإنشاء صورة جديدة.

```py
from diffusers import DiffusionPipeline
import torch

prior_pipeline = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
pipeline = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16, use_safetensors=True).to("cuda")

prompt = "A cheeseburger, white background"

image_embeds, negative_image_embeds = prior_pipeline(prompt, guidance_scale=1.0).to_tuple()
image = pipeline(
    prompt,
    image_embeds=image_embeds,
    negative_image_embeds=negative_image_embeds,
).images[0]

image.save("burger.png")
```

مرر تشيز برجر إلى [`ShapEImg2ImgPipeline`] لإنشاء تمثيل ثلاثي الأبعاد له.

```py
from PIL import Image
from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_gif

pipe = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e-img2img", torch_dtype=torch.float16, variant="fp16").to("cuda")

guidance_scale = 3.0
image = Image.open("burger.png").resize((256, 256))

images = pipe(
    image,
    guidance_scale=guidance_scale,
    num_inference_steps=64,
    frame_size=256,
).images

gif_path = export_to_gif(images[0], "burger_3d.gif")
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/shap_e/burger_in.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">cheeseburger</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/shap_e/burger_out.gif"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">3D cheeseburger</figcaption>
  </div>
</div>


## إنشاء شبكة

Shap-E هو نموذج مرن يمكنه أيضًا إنشاء مخرجات شبكة منسوجة ليتم تصييرها للتطبيقات اللاحقة. في هذا المثال، ستقوم بتحويل الإخراج إلى ملف `glb` لأن مكتبة مجموعات البيانات 🤗 تدعم عرض شبكة ملفات `glb` التي يمكن عرضها بواسطة [عارض مجموعة البيانات](https://huggingface.co/docs/hub/datasets-viewer#dataset-preview).

يمكنك إنشاء مخرجات شبكة لكل من [`ShapEPipeline`] و [`ShapEImg2ImgPipeline`] عن طريق تحديد `output_type` المعلمة كما `"mesh"`:

```py
import torch
from diffusers import ShapEPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to(device)

guidance_scale = 15.0
prompt = "A birthday cupcake"

images = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=64, frame_size=256, output_type="mesh").images
```

استخدم وظيفة [`~utils.export_to_ply`] لحفظ إخراج الشبكة كملف `ply`:

<Tip>
يمكنك أيضًا حفظ إخراج الشبكة كملف `obj` باستخدام وظيفة [`~utils.export_to_obj`]. إن القدرة على حفظ إخراج الشبكة في مجموعة متنوعة من التنسيقات تجعلها أكثر مرونة للاستخدام اللاحق!
</Tip>

```py
from diffusers.utils import export_to_ply

ply_path = export_to_ply(images[0], "3d_cake.ply")
print(f"Saved to folder: {ply_path}")
```

بعد ذلك، يمكنك تحويل ملف `ply` إلى ملف `glb` باستخدام مكتبة trimesh:

```py
import trimesh

mesh = trimesh.load("3d_cake.ply")
mesh_export = mesh.export("3d_cake.glb", file_type="glb")
```

افتراضيًا، يتم تركيز إخراج الشبكة من منظور الرؤية السفلي، ولكن يمكنك تغيير منظور الرؤية الافتراضي عن طريق تطبيق تحويل الدوران:

```py
import trimesh
import numpy as np

mesh = trimesh.load("3d_cake.ply")
rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
mesh = mesh.apply_transform(rot)
mesh_export = mesh.export("3d_cake.glb", file_type="glb")
```

قم بتحميل ملف الشبكة إلى مستودع مجموعة البيانات الخاصة بك لعرضه باستخدام عارض مجموعة البيانات!

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/3D-cake.gif"/>
</div>