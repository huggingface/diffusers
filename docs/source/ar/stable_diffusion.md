## فعالية وكفاءة الانتشار

يمكن أن يكون من الصعب جعل [`DiffusionPipeline`] تنتج صورًا بأسلوب معين أو تضمين ما تريده. في كثير من الأحيان، يتعين عليك تشغيل [`DiffusionPipeline`] عدة مرات قبل أن تحصل على صورة ترضيك. ولكن توليد شيء من لا شيء عملية مكثفة من الناحية الحسابية، خاصة إذا كنت تقوم بالاستنتاج مرارًا وتكرارًا.

لهذا السبب من المهم الحصول على أكبر قدر من الكفاءة *الحسابية* (السرعة) و*الذاكرة* (ذاكرة GPU) من الأنابيب لتقليل الوقت بين دورات الاستدلال بحيث يمكنك التكرار بشكل أسرع.

يوضح هذا البرنامج التعليمي كيفية التوليد بشكل أسرع وأفضل باستخدام [`DiffusionPipeline`].

ابدأ بتحميل نموذج [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5):

```python
from diffusers import DiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
```

سيكون مثال المطالبة الذي ستستخدمه هو صورة شخصية لرئيس محارب قديم، ولكن لا تتردد في استخدام مطالبتك الخاصة:

```python
prompt = "portrait photo of a old warrior chief"
```

## السرعة

<Tip>

💡 إذا لم يكن لديك حق الوصول إلى وحدة معالجة الرسومات (GPU)، فيمكنك استخدام واحدة مجانًا من مزود وحدة معالجة الرسومات مثل [Colab](https://colab.research.google.com/)!

</Tip>

من أبسط الطرق لتسريع الاستنتاج هو وضع الأنابيب على وحدة معالجة الرسومات (GPU) بنفس الطريقة التي تقوم بها مع أي وحدة PyTorch:

```python
pipeline = pipeline.to("cuda")
```

للتأكد من أنه يمكنك استخدام الصورة نفسها وتحسينها، استخدم [`Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) وقم بتعيين بذرة لـ [إمكانية إعادة الإنتاج](./using-diffusers/reusing_seeds):

```python
import torch

generator = torch.Generator("cuda").manual_seed(0)
```

الآن يمكنك إنشاء صورة:

```python
image = pipeline(prompt, generator=generator).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_1.png">
</div>

استغرقت هذه العملية ~30 ثانية على وحدة معالجة الرسومات T4 (قد تكون أسرع إذا كانت وحدة معالجة الرسومات المخصصة لك أفضل من T4). بشكل افتراضي، يقوم [`DiffusionPipeline`] بالاستنتاج بدقة `float32` الكاملة لـ 50 خطوة استدلال. يمكنك تسريع ذلك عن طريق التبديل إلى دقة أقل مثل `float16` أو تشغيل عدد أقل من خطوات الاستدلال.

لنبدأ بتحميل النموذج في `float16` وإنشاء صورة:

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

هذه المرة، استغرق الأمر ~11 ثانية فقط لتوليد الصورة، وهو أسرع 3 مرات تقريبًا من السابق!

<Tip>

💡 نوصي بشدة بتشغيل أنابيبك دائمًا في `float16`، وحتى الآن، نادرًا ما رأينا أي تدهور في جودة الإخراج.

</Tip>

الخيار الآخر هو تقليل عدد خطوات الاستدلال. قد يساعد اختيار جدول زمني أكثر كفاءة في تقليل عدد الخطوات دون التضحية بجودة الإخراج. يمكنك العثور على الجداول الزمنية المتوافقة مع النموذج الحالي في [`DiffusionPipeline`] عن طريق استدعاء طريقة `compatibles`:

```python
pipeline.scheduler.compatibles
[
diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler،
diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler،
diffusers.schedulers.scheduling_k_dpm_2_discrete.KDPM2DiscreteScheduler،
diffusers.schedulers.scheduling_deis_multistep.DEISMultistepScheduler،
diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler،
diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler،
diffusers.schedulers.scheduling_ddpm.DDPMScheduler،
diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler،
diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteScheduler،
diffusers.utils.dummy_torch_and_torchsde_objects.DPMSolverSDEScheduler،
diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler،
diffusers.schedulers.scheduling_pndm.PNDMScheduler،
diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler،
diffusers.schedulers.scheduling_ddim.DDIMScheduler،
]
```

يستخدم نموذج Stable Diffusion جدول [`PNDMScheduler`] بشكل افتراضي والذي يتطلب عادةً ~50 خطوة استدلال، ولكن الجداول الزمنية الأكثر كفاءة مثل [`DPMSolverMultistepScheduler`]، تتطلب فقط ~20 أو 25 خطوة استدلال. استخدم طريقة [`~ConfigMixin.from_config`] لتحميل جدول زمني جديد:

```python
from diffusers import DPMSolverMultistepScheduler

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
```

الآن قم بتعيين `num_inference_steps` إلى 20:

```python
generator = torch.Generator("cuda").manual_seed(0)
image = pipeline(prompt, generator=generator, num_inference_steps=20).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_3.png">
</div>

رائع، لقد نجحت في تقليص وقت الاستدلال إلى 4 ثوانٍ فقط! ⚡️

## الذاكرة

المفتاح الآخر لتحسين أداء الأنابيب هو استهلاك ذاكرة أقل، مما يعني ضمنيًا المزيد من السرعة، حيث تحاول غالبًا زيادة عدد الصور المولدة في الثانية. أسهل طريقة لمعرفة عدد الصور التي يمكنك إنشاؤها في نفس الوقت هي تجربة أحجام دفعات مختلفة حتى تحصل على `OutOfMemoryError` (OOM).

قم بإنشاء دالة ستولد دفعة من الصور من قائمة المطالبات و`Generators`. تأكد من تعيين بذرة لكل `Generator` حتى تتمكن من إعادة استخدامها إذا أنتجت نتيجة جيدة.

```python
def get_inputs(batch_size=1):
generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
prompts = batch_size * [prompt]
num_inference_steps = 20

return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}
```

ابدأ بـ `batch_size=4` وشاهد مقدار الذاكرة التي استهلكتها:

```python
from diffusers.utils import make_image_grid

images = pipeline(**get_inputs(batch_size=4)).images
make_image_grid(images, 2, 2)
```

من المحتمل أن تكون التعليمات البرمجية أعلاه قد أعادت خطأ `OOM` ما لم يكن لديك وحدة معالجة رسومات (GPU) بها ذاكرة وصول عشوائي (VRAM) أكبر! تشغل طبقات الاهتمام المتقاطع معظم الذاكرة. بدلاً من تشغيل هذه العملية في دفعة، يمكنك تشغيلها بالتتابع لتوفير كمية كبيرة من الذاكرة. كل ما عليك فعله هو تكوين الأنابيب لاستخدام وظيفة [`~DiffusionPipeline.enable_attention_slicing`]:

```python
pipeline.enable_attention_slicing()
```

الآن جرب زيادة `batch_size` إلى 8!

```python
images = pipeline(**get_inputs(batch_size=8)).images
make_image_grid(images, rows=2, cols=4)
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_5.png">
</div>

في حين أنك لم تتمكن من إنشاء دفعة من 4 صور، يمكنك الآن إنشاء دفعة من 8 صور عند ~3.5 ثانية لكل صورة! هذا هو على الأرجح أسرع ما يمكنك الذهاب إليه على وحدة معالجة الرسومات T4 دون التضحية بالجودة.

## الجودة

في القسمين الأخيرين، تعلمت كيفية تحسين سرعة أنابيبك باستخدام `fp16`، وتقليل عدد خطوات الاستدلال عن طريق استخدام جدول زمني أكثر كفاءة، وتمكين تقطيع الاهتمام لتقليل استهلاك الذاكرة. الآن ستركز على كيفية تحسين جودة الصور المولدة.

### نقاط مرجعية أفضل

الخطوة الأكثر وضوحًا هي استخدام نقاط مرجعية أفضل. يعد نموذج Stable Diffusion نقطة انطلاق جيدة، ومنذ إطلاقه الرسمي، تم إصدار العديد من الإصدارات المحسنة أيضًا. ومع ذلك، فإن استخدام إصدار أحدث لا يعني تلقائيًا أنك ستحصل على نتائج أفضل. لا يزال يتعين عليك تجربة نقاط مرجعية مختلفة بنفسك، وإجراء بعض الأبحاث (مثل استخدام [المطالبات السلبية](https://minimaxir.com/2022/11/stable-diffusion-negative-prompt/)) للحصول على أفضل النتائج.

مع نمو المجال، هناك المزيد والمزيد من نقاط المراقبة عالية الجودة التي تمت معاينتها لإنتاج أساليب معينة. جرب استكشاف [Hub](https://huggingface.co/models?library=diffusers&sort=downloads) و[معرض Diffusers](https://huggingface.co/spaces/huggingface-projects/diffusers-gallery) للعثور على ما يثير اهتمامك!

### مكونات الأنابيب الأفضل

يمكنك أيضًا تجربة استبدال مكونات الأنابيب الحالية بإصدار أحدث. دعنا نحاول تحميل أحدث [autoencoder](https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main/vae) من Stability AI في الأنابيب، وإنشاء بعض الصور:

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
### تحسين تصميم المحثات 
نص المحث الذي تستخدمه لتوليد صورة ما مهم للغاية، لدرجة أنه يُطلق عليه اسم *هندسة المحثات*. هناك بعض الاعتبارات التي يجب مراعاتها أثناء هندسة المحثات: 

- كيف يتم تخزين الصورة أو الصور المشابهة للصورة التي أريد توليدها على الإنترنت؟ 
- ما هي التفاصيل الإضافية التي يمكنني تقديمها والتي توجه النموذج نحو الأسلوب الذي أريده؟ 

مع أخذ ذلك في الاعتبار، دعونا نحسن المحث لإدراج اللون وتفاصيل الجودة العالية: 

```python
prompt += ", tribal panther make up, blue on red, side profile, looking away, serious eyes"
prompt += " 50mm portrait photography, hard rim lighting photography--beta --ar 2:3 --beta --upbeta"
```

قم بتوليد مجموعة من الصور باستخدام المحث الجديد: 

```python
images = pipeline(**get_inputs(batch_size=8)).images
make_image_grid(images, rows=2, cols=4)
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_7.png">
</div> 

مثير للإعجاب حقًا! دعونا نعدل الصورة الثانية - المقابلة لـ `Generator` مع بذرة `1` - بشكل أكبر من خلال إضافة بعض النص حول عمر الموضوع: 

```python
prompts = [
"portrait photo of the oldest warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3 --beta --upbeta",
"portrait photo of a old warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3 --beta --upbeta",
"portrait photo of a warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3 --beta --upbeta",
"portrait photo of a young warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3 --beta --upbeta",
]

generator = [torch.Generator("cuda").manual_seed(1) for _ in range(len(prompts))]
images = pipeline(prompt=prompts, generator=generator, num_inference_steps=25).images
make_image_grid(images, 2, 2)
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_8.png">
</div> 

## الخطوات التالية 

في هذا البرنامج التعليمي، تعلمت كيفية تحسين [`DiffusionPipeline`] لكفاءة الحساب والذاكرة، بالإضافة إلى تحسين جودة المخرجات المولدة. إذا كنت مهتمًا بجعل خط أنابيبك أسرع، فراجع الموارد التالية: 

- تعرف على كيفية [PyTorch 2.0](./optimization/torch2.0) و [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) يمكن أن يحقق سرعة استدلال أسرع بنسبة 5-300%. على GPU A100، يمكن أن يكون الاستدلال أسرع بنسبة تصل إلى 50%! 

- إذا لم تتمكن من استخدام PyTorch 2، نوصي بتثبيت [xFormers](./optimization/xformers). تعمل آلية الاهتمام بكفاءة الذاكرة بشكل رائع مع PyTorch 1.13.1 لسرعة أكبر واستهلاك ذاكرة أقل. 

- يتم تغطية تقنيات التحسين الأخرى، مثل إزالة تحميل النموذج، في [هذا الدليل](./optimization/fp16).