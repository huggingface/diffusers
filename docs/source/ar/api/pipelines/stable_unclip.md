# Stable unCLIP

تم ضبط نقاط تثبيت unCLIP من نقاط تثبيت [Stable Diffusion 2.1](./stable_diffusion/stable_diffusion_2) لتصبح مشروطة في تضمين صور CLIP. لا يزال Stable unCLIP يعتمد على تضمين النص. نظرًا لوجود شرطين منفصلين، يمكن استخدام Stable unCLIP للتنوع الصوري الموجه بالنص. عند دمجه مع نموذج unCLIP السابق، يمكن استخدامه أيضًا لتوليد الصور بالكامل من النص.

المستخلص من الورقة هو:

*أظهرت النماذج التمييزية مثل CLIP أنها تتعلم تمثيلات قوية للصور التي تلتقط كل من الدلالات والأسلوب. للاستفادة من هذه التمثيلات لتوليد الصور، نقترح نموذجًا مكونًا من مرحلتين: نموذج سابق ينشئ تضمين صورة CLIP معطى تعليق نصي، وفك تشفير ينشئ صورة مشروطة بتضمين الصورة. نُظهر أن إنشاء تمثيلات الصور بشكل صريح يحسن تنوع الصور مع فقدان ضئيل في الواقعية التشابه في التعليق. يمكن لفك تشفيرنا المشروط بتمثيلات الصور أيضًا إنتاج تنويعات لصورة ما مع الحفاظ على كل من الدلالات والأسلوب، مع تغيير التفاصيل غير الأساسية الغائبة عن تمثيل الصورة. علاوة على ذلك، تمكن مساحة التضمين المشتركة لـ CLIP من التلاعب بالصور الموجهة باللغة بطريقة zero-shot. نستخدم نماذج الانتشار للفك والترميز، ونجرب كلًا من النماذج التلقائية والنماذج الانتشارية للنماذج السابقة، ونجد أن هذه الأخيرة أكثر كفاءة من الناحية الحسابية وتنتج عينات ذات جودة أعلى.*

## نصائح

يأخذ Stable unCLIP `noise_level` كإدخال أثناء الاستنتاج والذي يحدد مقدار الضوضاء المضافة إلى تضمين الصورة. يؤدي ارتفاع مستوى الضوضاء إلى زيادة التباين في الصور النهائية غير المزودة بالضوضاء. بشكل افتراضي، لا نضيف أي ضوضاء إضافية إلى تضمين الصورة (`noise_level = 0`).

### توليد الصور من النص

يمكن الاستفادة من Stable unCLIP لتوليد الصور من النص عن طريق توصيله بنموذج سابق من نموذج KakaoBrain مفتوح المصدر DALL-E 2 replication [Karlo](https://huggingface.co/kakaobrain/karlo-v1-alpha):

```python
import torch
from diffusers import UnCLIPScheduler, DDPMScheduler, StableUnCLIPPipeline
from diffusers.models import PriorTransformer
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

prior_model_id = "kakaobrain/karlo-v1-alpha"
data_type = torch.float16
prior = PriorTransformer.from_pretrained(prior_model_id, subfolder="prior", torch_dtype=data_type)

prior_text_model_id = "openai/clip-vit-large-patch14"
prior_tokenizer = CLIPTokenizer.from_pretrained(prior_text_model_id)
prior_text_model = CLIPTextModelWithProjection.from_pretrained(prior_text_model_id, torch_dtype=data_type)
prior_scheduler = UnCLIPScheduler.from_pretrained(prior_model_id, subfolder="prior_scheduler")
prior_scheduler = DDPMScheduler.from_config(prior_scheduler.config)

stable_unclip_model_id = "stabilityai/stable-diffusion-2-1-unclip-small"

pipe = StableUnCLIPPipeline.from_pretrained(
    stable_unclip_model_id,
    torch_dtype=data_type,
    variant="fp16",
    prior_tokenizer=prior_tokenizer,
    prior_text_encoder=prior_text_model,
    prior=prior,
    prior_scheduler=prior_scheduler,
)

pipe = pipe.to("cuda")
wave_prompt = "dramatic wave, the Oceans roar, Strong wave spiral across the oceans as the waves unfurl into roaring crests; perfect wave form; perfect wave shape; dramatic wave shape; wave shape unbelievable; wave; wave shape spectacular"

image = pipe(prompt=wave_prompt).images[0]
image
```

<Tip warning={true}>

للنص إلى صورة نستخدم `stabilityai/stable-diffusion-2-1-unclip-small` لأنه تم تدريبه على تضمين CLIP ViT-L/14، مثل نموذج Karlo السابق. تم تدريب [stabilityai/stable-diffusion-2-1-unclip](https://hf.co/stabilityai/stable-diffusion-2-1-unclip) على OpenCLIP ViT-H، لذلك لا نوصي باستخدامه.

</Tip>

### تنوع الصور الموجهة بالنص

```python
from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers.utils import load_image
import torch

pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
)
pipe = pipe.to("cuda")

url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/tarsila_do_amaral.png"
init_image = load_image(url)

images = pipe(init_image).images
images[0].save("variation_image.png")
```

اختياريًا، يمكنك أيضًا تمرير موجه إلى `pipe` مثل:

```python
prompt = "A fantasy landscape, trending on artstation"

image = pipe(init_image, prompt=prompt).images[0]
image
```

<Tip>

تأكد من مراجعة الدليل [Schedulers](../../using-diffusers/schedulers) لمعرفة كيفية استكشاف التوازن بين سرعة المجدول والجودة، وقسم [إعادة استخدام المكونات عبر الأنابيب](../../using-diffusers/loading#reuse-components-across-pipelines) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في أنابيب متعددة.

</Tip>

## StableUnCLIPPipeline

[[autodoc]] StableUnCLIPPipeline
- all
- __call__
- enable_attention_slicing
- disable_attention_slicing
- enable_vae_slicing
- disable_vae_slicing
- enable_xformers_memory_efficient_attention
- disable_xformers_memory_efficient_attention

## StableUnCLIPImg2ImgPipeline

[[autodoc]] StableUnCLIPImg2ImgPipeline
- all
- __call__
- enable_attention_slicing
- disable_attention_slicing
- enable_vae_slicing
- disable_vae_slicing
- enable_xformers_memory_efficient_attention
- disable_xformers_memory_efficient_attention

## ImagePipelineOutput

[[autodoc]] pipelines.ImagePipelineOutput