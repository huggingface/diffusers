# النص أو الصورة إلى الفيديو

انطلاقًا من نجاح نماذج النص إلى الصورة القائمة على الانتشار، يمكن لنماذج الفيديو التوليدية إنشاء مقاطع فيديو قصيرة من موجه نصي أو صورة أولية. تُضيف هذه النماذج نوعًا من الطبقات الزمنية و/أو المكانية إلى البنية المعمارية لتوسيع نموذج الانتشار المُدرب مسبقًا لتوليد الفيديو. ويتم استخدام مجموعة بيانات مختلطة من الصور ومقاطع الفيديو لتدريب النموذج الذي يتعلم إخراج سلسلة من لقطات الفيديو بناءً على النص أو الصورة الشرطية.

سيوضح هذا الدليل كيفية إنشاء مقاطع الفيديو، وكيفية تكوين معلمات نموذج الفيديو، وكيفية التحكم في إنشاء الفيديو.

## النماذج الشائعة

> [!TIP]
> اكتشف نماذج توليد الفيديو الرائجة والحديثة الأخرى على [المنصة](https://huggingface.co/models?pipeline_tag=text-to-video&sort=trending)!

تُعد [Stable Video Diffusions (SVD)](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) و [I2VGen-XL](https://huggingface.co/ali-vilab/i2vgen-xl/) و [AnimateDiff](https://huggingface.co/guoyww/animatediff) و [ModelScopeT2V](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b) نماذج شائعة الاستخدام في انتشار الفيديو. يتميز كل نموذج عن الآخر. على سبيل المثال، يقوم نموذج AnimateDiff بإدراج وحدة نمذجة الحركة في نموذج نص إلى صورة مجمد لتوليد صور متحركة مخصصة، في حين أن نموذج SVD مُدرب بالكامل من البداية باستخدام عملية تدريب من ثلاث مراحل لتوليد مقاطع فيديو قصيرة عالية الجودة.

### انتشار الفيديو المستقر

يستند [SVD](../api/pipelines/svd) إلى نموذج Stable Diffusion 2.1 وهو مُدرب على الصور، ثم مقاطع الفيديو منخفضة الدقة، وأخيرًا مجموعة بيانات أصغر من مقاطع الفيديو عالية الدقة. يقوم هذا النموذج بتوليد مقطع فيديو قصير مدته من 2 إلى 4 ثوانٍ من صورة أولية. يمكنك معرفة المزيد من التفاصيل حول النموذج، مثل التكييف الدقيق، في دليل [Stable Video Diffusion](../using-diffusers/svd).

ابدأ بتحميل [`StableVideoDiffusionPipeline`] وتمرير صورة أولية لتوليد مقطع فيديو منها.

```py
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipeline = StableVideoDiffusionPipeline.from_pretrained(
"stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipeline(image, decode_chunk_size=8, generator=generator).frames[0]
export_to_video(frames, "generated.mp4", fps=7)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأولية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/output_rocket.gif"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">مقطع الفيديو المُولد</figcaption>
</div>
</div>

### I2VGen-XL

[I2VGen-XL](../api/pipelines/i2vgenxl) هو نموذج انتشار يمكنه توليد مقاطع فيديو بدقة أعلى من SVD، كما أنه قادر على قبول موجهات نصية بالإضافة إلى الصور. تم تدريب النموذج باستخدام مشفرين هرميين (مشفر التفاصيل ومشفر عالمي) لالتقاط التفاصيل منخفضة المستوى وعالية المستوى في الصور بشكل أفضل. ويتم استخدام هذه التفاصيل المُتعلمة لتدريب نموذج انتشار الفيديو الذي يحسن دقة الفيديو والتفاصيل في الفيديو المُولد.

يمكنك استخدام I2VGen-XL عن طريق تحميل [`I2VGenXLPipeline`]، وتمرير موجه نصي وصورة لتوليد مقطع فيديو.

```py
import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif, load_image

pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
pipeline.enable_model_cpu_offload()

image_url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png"
image = load_image(image_url).convert("RGB")

prompt = "Papers were floating in the air on a table in the library"
negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
generator = torch.manual_seed(8888)

frames = pipeline(
prompt=prompt,
image=image,
num_inference_steps=50,
negative_prompt=negative_prompt,
guidance_scale=9.0,
generator=generator
).frames[0]
export_to_gif(frames, "i2v.gif")
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأولية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/i2vgen-xl-example.gif"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">مقطع الفيديو المُولد</figcaption>
</div>
</div>

### AnimateDiff

[AnimateDiff](../api/pipelines/animatediff) هو نموذج مُكيّف يقوم بإدراج وحدة حركة في نموذج انتشار مُدرب مسبقًا لتحريك صورة. يتم تدريب المُكيّف على مقاطع الفيديو لتعلم الحركة التي تُستخدم لشرط عملية التوليد لإنشاء مقطع فيديو. إنه أسرع وأسهل في تدريب المُكيّف فقط، ويمكن تحميله في معظم نماذج الانتشار، مما يحولها فعليًا إلى "نماذج فيديو".

ابدأ بتحميل [`MotionAdapter`].

```py
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
```

ثم قم بتحميل نموذج Stable Diffusion المُدرب باستخدام [`AnimateDiffPipeline`].

```py
pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
"emilianJR/epiCRealism",
subfolder="scheduler",
clip_sample=False,
timestep_spacing="linspace",
beta_schedule="linear",
steps_offset=1,
)
pipeline.scheduler = scheduler
pipeline.enable_vae_slicing()
pipeline.enable_model_cpu_offload()
```

قم بإنشاء موجه نصي وقم بتوليد مقطع الفيديو.

```py
output = pipeline(
prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
negative_prompt="bad quality, worse quality, low resolution",
num_frames=16,
guidance_scale=7.5,
num_inference_steps=50,
generator=torch.Generator("cpu").manual_seed(49),
)
frames = output.frames[0]
export_to_gif(frames, "animation.gif")
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff.gif"/>
</div>

### ModelscopeT2V

[ModelscopeT2V](../api/pipelines/text_to_video) يضيف عمليات تجميع ومكانية وزمانية وانتباه إلى شبكة UNet، وهو مُدرب على مجموعات بيانات الصور والنصوص ومقاطع الفيديو والنصوص لتحسين ما يتعلمه أثناء التدريب. يأخذ النموذج موجهًا نصيًا، ويقوم بتشفيره وإنشاء تضمينات نصية يتم إزالة تشويشها بواسطة شبكة UNet، ثم فك تشفيرها بواسطة VQGAN إلى مقطع فيديو.

<Tip>
يُنشئ نموذج ModelScopeT2V مقاطع فيديو تحتوي على علامة مائية بسبب مجموعات البيانات التي تم تدريبه عليها. لاستخدام نموذج خالٍ من العلامات المائية، جرّب نموذج [cerspense/zeroscope_v2_76w](https://huggingface.co/cerspense/zeroscope_v2_576w) مع [`TextToVideoSDPipeline`] أولاً، ثم قم بتحسين إخراجها باستخدام نقطة تفتيش [cerspense/zeroscope_v2_XL](https://huggingface.co/cerspense/zeroscope_v2_XL) باستخدام [`VideoToVideoSDPipeline`].
</Tip>

قم بتحميل نقطة تفتيش ModelScopeT2V في [`DiffusionPipeline`] مع موجه نصي لتوليد مقطع فيديو.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipeline = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()

prompt = "Confident teddy bear surfer rides the wave in the tropics"
video_frames = pipeline(prompt).frames[0]
export_to_video(video_frames, "modelscopet2v.mp4", fps=10)
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/modelscopet2v.gif" />
</div>

## تكوين معلمات النموذج

هناك بعض المعلمات المهمة التي يمكنك تكوينها في الأنبوب والتي ستؤثر على عملية إنشاء الفيديو وجودته. دعنا نلقي نظرة فاحصة على ما تفعله هذه المعلمات وكيف يؤثر تغييرها على الإخراج.

### عدد الإطارات

تحدد معلمة `num_frames` عدد إطارات الفيديو المولدة في الثانية. الإطار هو صورة يتم تشغيلها في تسلسل مع إطارات أخرى لإنشاء حركة أو مقطع فيديو. يؤثر هذا على مدة الفيديو لأن الأنبوب يولد عددًا معينًا من الإطارات في الثانية (تحقق من المرجع API للأنبوب للتعرف على القيمة الافتراضية). لزيادة مدة الفيديو، ستحتاج إلى زيادة معلمة `num_frames`.

```py
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipeline = StableVideoDiffusionPipeline.from_pretrained(
"stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipeline(image, decode_chunk_size=8, generator=generator, num_frames=25).frames[0]
export_to_video(frames, "generated.mp4", fps=7)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/num_frames_14.gif"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">num_frames=14</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/num_frames_25.gif"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">num_frames=25</figcaption>
</div>
</div>

### مقياس التوجيه

تتحكم معلمة `guidance_scale` في مدى توافق مقطع الفيديو المُولد مع الموجه النصي أو الصورة الأولية. تشير قيمة `guidance_scale` الأعلى إلى أن مقطع الفيديو المُولد أكثر توافقًا مع الموجه النصي أو الصورة الأولية، في حين أن قيمة `guidance_scale` الأقل تشير إلى أن مقطع الفيديو المُولد أقل توافقًا، مما قد يمنح النموذج "إبداعًا" أكثر لتفسير إدخال الشرط.

<Tip>
يستخدم SVD معلمات `min_guidance_scale` و `max_guidance_scale` لتطبيق التوجيه على الإطارات الأولى والأخيرة على التوالي.
</Tip>

```py
import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif, load_image

pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
pipeline.enable_model_cpu_offload()

image_url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png"
image = load_image(image_url).convert("RGB")

prompt = "Papers were floating in the air on a table in the library"
negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
generator = torch.manual_seed(0)

frames = pipeline(
prompt=prompt,
image=image,
num_inference_steps=50,
negative_prompt=negative_prompt,
guidance_scale=1.0,
generator=generator
).frames[0]
export_to_gif(frames, "i2v.gif")
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/i2vgen-xl-example.gif"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale=9.0</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guidance_scale_1.0.gif"/>
<figcaption class="mt-2 text-center text-sm
### Negative prompt

يمنع المحث السلبي النموذج من توليد أشياء لا تريده أن يقوم بها. ويستخدم هذا المعامل بشكل شائع لتحسين جودة التوليد بشكل عام عن طريق إزالة الخصائص الرديئة أو السيئة مثل "دقة منخفضة" أو "تفاصيل سيئة".

```py
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)

pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
"emilianJR/epiCRealism",
subfolder="scheduler",
clip_sample=False,
timestep_spacing="linspace",
beta_schedule="linear",
steps_offset=1,
)
pipeline.scheduler = scheduler
pipeline.enable_vae_slicing()
pipeline.enable_model_cpu_offload()

output = pipeline(
prompt="360 camera shot of a sushi roll in a restaurant",
negative_prompt="Distorted, discontinuous, ugly, blurry, low resolution, motionless, static",
num_frames=16,
guidance_scale=7.5,
num_inference_steps=50,
generator=torch.Generator("cpu").manual_seed(0),
)
frames = output.frames[0]
export_to_gif(frames, "animation.gif")
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff_no_neg.gif"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">بدون محث سلبي</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff_neg.gif"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">تم تطبيق المحث السلبي</figcaption>
</div>
</div>

### معلمات خاصة بالنموذج

هناك بعض معلمات الأنابيب التي تكون فريدة لكل نموذج، مثل ضبط الحركة في الفيديو أو إضافة ضوضاء إلى الصورة الأولية.

<hfoptions id="special-parameters">
<hfoption id="Stable Video Diffusion">

يوفر Stable Video Diffusion ميكرو-كونديشينينج إضافي لمعدل الإطارات باستخدام معلمة "fps" وللحركة باستخدام معلمة "motion_bucket_id". وتسمح هذه المعلمات معًا بتعديل مقدار الحركة في الفيديو المولد.

هناك أيضًا معلمة "noise_aug_strength" التي تزيد من مقدار الضوضاء المضافة إلى الصورة الأولية. يؤثر تغيير هذه المعلمة على مدى تشابه الفيديو المولد والصورة الأولية. كما تزيد قيمة "noise_aug_strength" الأعلى من مقدار الحركة. لمزيد من المعلومات، اقرأ دليل [Micro-conditioning](../using-diffusers/svd#micro-conditioning).

</hfoption>
<hfoption id="Text2Video-Zero">

يحسب Text2Video-Zero مقدار الحركة المطلوب تطبيقها على كل إطار من اللانتمات التي تم أخذ عينات منها بشكل عشوائي. يمكنك استخدام معلمات "motion_field_strength_x" و "motion_field_strength_y" للتحكم في مقدار الحركة المطلوب تطبيقها على محوري الفيديو x و y. والمعلمات "t0" و "t1" هي خطوات الوقت لتطبيق الحركة على اللانتمات.

</hfoption>
</hfoptions>

## التحكم في توليد الفيديو

يمكن التحكم في توليد الفيديو بطريقة مماثلة للطريقة التي يتم بها التحكم في النص إلى الصورة والصورة إلى الصورة والطلاء باستخدام [`ControlNetModel`]. والفرق الوحيد هو أنك بحاجة إلى استخدام [`~pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.CrossFrameAttnProcessor`] بحيث يحضر كل إطار الإطار الأول.

### Text2Video-Zero

يمكن ضبط توليد الفيديو Text2Video-Zero بناءً على صور الوضع والحافة لمزيد من التحكم في حركة الموضوع في الفيديو المولد أو للحفاظ على هوية موضوع/كائن في الفيديو. يمكنك أيضًا استخدام Text2Video-Zero مع [InstructPix2Pix](../api/pipelines/pix2pix) لتحرير الفيديوهات بالنص.

<hfoptions id="t2v-zero">
<hfoption id="pose control">

ابدأ بتنزيل فيديو واستخراج صور الوضع منه.

```py
from huggingface_hub import hf_hub_download
from PIL import Image
import imageio

filename = "__assets__/poses_skeleton_gifs/dance1_corr.mp4"
repo_id = "PAIR/Text2Video-Zero"
video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)

reader = imageio.get_reader(video_path, "ffmpeg")
frame_count = 8
pose_images = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]
```

قم بتحميل [`ControlNetModel`] لتقدير الوضع ونقطة تفتيش في [`StableDiffusionControlNetPipeline`]. بعد ذلك، ستستخدم [`~pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.CrossFrameAttnProcessor`] لـ UNet و ControlNet.

```py
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

model_id = "runwayml/stable-diffusion-v1-5"
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
model_id, controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

pipeline.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipeline.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
```

ثبّت اللانتمات لجميع الإطارات، ثم مرر محثك وصور الوضع المستخرجة إلى النموذج لتوليد فيديو.

```py
latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(len(pose_images), 1, 1, 1)

prompt = "Darth Vader dancing in a desert"
result = pipeline(prompt=[prompt] * len(pose_images), image=pose_images, latents=latents).images
imageio.mimsave("video.mp4", result, fps=4)
```

</hfoption>
<hfoption id="edge control">

قم بتنزيل فيديو واستخراج الحواف منه.

```py
from huggingface_hub import hf_hub_download
from PIL import Image
import imageio

filename = "__assets__/poses_skeleton_gifs/dance1_corr.mp4"
repo_id = "PAIR/Text2Video-Zero"
video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)

reader = imageio.get_reader(video_path, "ffmpeg")
frame_count = 8
pose_images = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]
```

قم بتحميل [`ControlNetModel`] للحافة القانية ونقطة تفتيش في [`StableDiffusionControlNetPipeline`]. بعد ذلك، ستستخدم [`~pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.CrossFrameAttnProcessor`] لـ UNet و ControlNet.

```py
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

model_id = "runwayml/stable-diffusion-v1-5"
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
model_id, controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

pipeline.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipeline.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
```

ثبّت اللانتمات لجميع الإطارات، ثم مرر محثك وصور الحافة المستخرجة إلى النموذج لتوليد فيديو.

```py
latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(len(pose_images), 1, 1, 1)

prompt = "Darth Vader dancing in a desert"
result = pipeline(prompt=[prompt] * len(pose_images), image=pose_images, latents=latents).images
imageio.mimsave("video.mp4", result, fps=4)
```

</hfoption>
<hfoption id="InstructPix2Pix">

يسمح InstructPix2Pix باستخدام النص لوصف التغييرات التي تريد إجراؤها على الفيديو. ابدأ بتنزيل فيديو وقراءته.

```py
from huggingface_hub import hf_hub_download
from PIL import Image
import imageio

filename = "__assets__/pix2pix video/camel.mp4"
repo_id = "PAIR/Text2Video-Zero"
video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)

reader = imageio.get_reader(video_path, "ffmpeg")
frame_count = 8
video = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]
```

قم بتحميل [`StableDiffusionInstructPix2PixPipeline`] وتعيين [`~pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.CrossFrameAttnProcessor`] لـ UNet.

```py
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16).to("cuda")
pipeline.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))
```

مرر محثًا يصف التغيير الذي تريد تطبيقه على الفيديو.

```py
prompt = "make it Van Gogh Starry Night style"
result = pipeline(prompt=[prompt] * len(video), image=video).images
imageio.mimsave("edited_video.mp4", result, fps=4)
```

</hfoption>
</hfoptions>

## تحسين

يتطلب إنشاء الفيديو الكثير من الذاكرة لأنك تقوم بتوليد العديد من إطارات الفيديو في وقت واحد. يمكنك تقليل متطلبات الذاكرة لديك على حساب بعض سرعة الاستدلال. جرّب ما يلي:

1. قم بتفريغ مكونات الأنبوب التي لم تعد بحاجة إليها إلى وحدة المعالجة المركزية
2. يقوم التقطيع للأمام بتشغيل طبقة التغذية الأمامية في حلقة بدلاً من تشغيلها جميعها مرة واحدة
3. قم بتقسيم عدد الإطارات التي يجب أن يقوم VAE بفك تشفيرها إلى مجموعات بدلاً من فك تشفيرها جميعها مرة واحدة

```diff
- pipeline.enable_model_cpu_offload()
- frames = pipeline(image, decode_chunk_size=8, generator=generator).frames[0]
+ pipeline.enable_model_cpu_offload()
+ pipeline.unet.enable_forward_chunking()
+ frames = pipeline(image, decode_chunk_size=2, generator=generator, num_frames=25).frames[0]
```

إذا لم تكن الذاكرة مشكلة وتريد التحسين للسرعة، فجرّب لف UNet مع [`torch.compile`](../optimization/torch2.0#torchcompile).

```diff
- pipeline.enable_model_cpu_offload()
+ pipeline.to("cuda")
+ pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
```