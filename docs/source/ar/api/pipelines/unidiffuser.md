# UniDiffuser

اقترح نموذج UniDiffuser في [One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale](https://huggingface.co/papers/2303.06555) بواسطة Fan Bao, Shen Nie, Kaiwen Xue, Chongxuan Li, Shi Pu, Yaole Wang, Gang Yue, Yue Cao, Hang Su, Jun Zhu.

الملخص من الورقة هو:

*تقترح هذه الورقة إطار عمل موحد للانتشار (يُطلق عليه UniDiffuser) ليتناسب مع جميع التوزيعات ذات الصلة بمجموعة من البيانات متعددة الوسائط في نموذج واحد. الفكرة الأساسية لدينا هي - يمكن توحيد تعلم نماذج الانتشار للتوزيعات الهامشية والشرطية والمشتركة على أنها التنبؤ بالضوضاء في البيانات المضطربة، حيث يمكن أن تكون مستويات الاضطراب (أي الخطوات الزمنية) مختلفة لوسائط مختلفة. مستوحاة من وجهة النظر الموحدة، يتعلم UniDiffuser جميع التوزيعات في وقت واحد مع تعديل طفيف على نموذج الانتشار الأصلي - يضطرب البيانات في جميع الوسائط بدلاً من وسيط واحد، ويدخل خطوات زمنية فردية في وسائط مختلفة، ويتنبأ بضوضاء جميع الوسائط بدلاً من وسيط واحد. يتم معلمه UniDiffuser بواسطة محول للنشر النماذج للتعامل مع أنواع الإدخال المختلفة للوسائط المختلفة. عند تنفيذه على بيانات الصور والنصوص المقترنة واسعة النطاق، يمكن لـ UniDiffuser إجراء الصور والنصوص وتوليد الصور والنصوص وأزواج الصور والنصوص من خلال تعيين خطوات زمنية مناسبة دون أي تكلفة إضافية. على وجه الخصوص، يمكن لـ UniDiffuser إنتاج عينات واقعية إدراكيًا في جميع المهام، ونتائجه الكمية (مثل FID و CLIP score) ليست متفوقة فحسب على النماذج متعددة الأغراض الحالية ولكنها أيضًا قابلة للمقارنة مع النماذج المخصصة (مثل Stable Diffusion و DALL-E 2) في مهام تمثيلية (مثل توليد الصور من النص).*

يمكنك العثور على كود المصدر الأصلي في [thu-ml/unidiffuser](https://github.com/thu-ml/unidiffuser) ونقاط التحقق الإضافية في [thu-ml](https://huggingface.co/thu-ml).

<Tip warning={true}>

يوجد حاليًا مشكلة في PyTorch 1.X حيث تكون الصور الناتجة سوداء تمامًا أو تصبح قيم البكسل `NaNs`. يمكن التخفيف من حدة هذه المشكلة عن طريق التبديل إلى PyTorch 2.X.

</Tip>

تمت المساهمة بهذا الأنبوب بواسطة [dg845](https://github.com/dg845). ❤️

## أمثلة الاستخدام

بما أن نموذج UniDiffuser مدرب على نمذجة التوزيع المشترك لأزواج (الصورة، النص)، فإنه قادر على أداء مجموعة متنوعة من مهام التوليد:

### التوليد غير المشروط للصور والنصوص

سيؤدي التوليد غير المشروط (حيث نبدأ فقط من الكمونات التي تم أخذ عينات منها من توزيع غاوسي القياسي السابق) من [`UniDiffuserPipeline`] إلى إنتاج زوج (صورة، نص):

```python
import torch

from diffusers import UniDiffuserPipeline

device = "cuda"
model_id_or_path = "thu-ml/unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# التوليد غير المشروط للصور والنصوص. يتم استنتاج مهمة التوليد تلقائيًا.
sample = pipe(num_inference_steps=20, guidance_scale=8.0)
image = sample.images[0]
text = sample.text[0]
image.save("unidiffuser_joint_sample_image.png")
print(text)
```

يُطلق على هذا أيضًا "التوليد المشترك" في ورقة UniDiffuser، حيث نقوم بأخذ عينات من التوزيع المشترك للصور والنصوص.

لاحظ أنه يتم استنتاج مهمة التوليد من الإدخالات المستخدمة عند استدعاء الأنبوب.

من الممكن أيضًا تحديد مهمة التوليد غير المشروط ("الوضع") يدويًا باستخدام [`UniDiffuserPipeline.set_joint_mode`]:

```python
# مكافئ لما سبق.
pipe.set_joint_mode()
sample = pipe(num_inference_steps=20, guidance_scale=8.0)
```

عندما يتم تعيين الوضع يدويًا، ستستخدم الاستدعاءات اللاحقة للأنبوب الوضع المحدد دون محاولة استنتاج الوضع.

يمكنك إعادة تعيين الوضع باستخدام [`UniDiffuserPipeline.reset_mode`], بعد ذلك، سيستنتج الأنبوب الوضع مرة أخرى.

يمكنك أيضًا إنشاء صورة فقط أو نص فقط (والذي يطلق عليه ورقة UniDiffuser "التوليد الهامشي" حيث نقوم بأخذ عينات من التوزيع الهامشي للصور والنصوص، على التوالي):

```python
# على عكس مهام التوليد الأخرى، لا يستخدم التوليد الخاص بالصور فقط والنص فقط الإرشاد الخالي من التصنيف
# التوليد الخاص بالصور فقط
pipe.set_image_mode()
sample_image = pipe(num_inference_steps=20).images[0]
# التوليد الخاص بالنص فقط
pipe.set_text_mode()
sample_text = pipe(num_inference_steps=20).text[0]
```

### التوليد من النص إلى الصورة

UniDiffuser قادر أيضًا على أخذ العينات من التوزيعات الشرطية؛ أي توزيع الصور المشروطة بنص موجه أو توزيع النصوص المشروطة بصورة.

فيما يلي مثال على أخذ العينات من التوزيع الشرطي للصورة (التوليد من النص إلى الصورة أو التوليد المشروط بالصور للنص):

```python
import torch

from diffusers import UniDiffuserPipeline

device = "cuda"
model_id_or_path = "thu-ml/unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# التوليد من النص إلى الصورة
prompt = "فيل تحت البحر"

sample = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=8.0)
t2i_image = sample.images[0]
t2i_image
```

يتطلب وضع `text2img` إما إدخال `prompt` أو `prompt_embeds`. يمكنك تعيين وضع `text2img` يدويًا باستخدام [`UniDiffuserPipeline.set_text_to_image_mode`].

### التوليد من الصورة إلى النص

وبالمثل، يمكن لـ UniDiffuser أيضًا إنتاج عينات نصية بناءً على صورة (التوليد من الصورة إلى النص أو التوليد المشروط بالصور للنص):

```python
import torch

from diffusers import UniDiffuserPipeline, load_image

device = "cuda"
model_id_or_path = "thu-ml/unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# التوليد من الصورة إلى النص
image_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/unidiffuser/unidiffuser_example_image.jpg"
init_image = load_image(image_url).resize((512, 512))

sample = pipe(image=init_image, num_inference_steps=20, guidance_scale=8.0)
i2t_text = sample.text[0]
print(i2t_text)
```

يتطلب وضع `img2text` إدخال `image`. يمكنك تعيين وضع `img2text` يدويًا باستخدام [`UniDiffuserPipeline.set_image_to_text_mode`].

### تنويع الصور

يقترح مؤلفو UniDiffuser إجراء تنويع الصور من خلال طريقة "الرحلة ذهابًا وإيابًا"، حيث نقوم، عند إعطاء صورة إدخال، بإجراء توليد من الصورة إلى النص أولاً، ثم إجراء توليد من النص إلى الصورة على مخرجات التوليد الأول.

ينتج عن هذا صورة جديدة تشبه من الناحية الدلالية صورة الإدخال:

```python
import torch

from diffusers import UniDiffuserPipeline, load_image

device = "cuda"
model_id_or_path = "thu-ml/unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# يمكن إجراء تنويع الصور من خلال التوليد من الصورة إلى النص متبوعًا بالتوليد من النص إلى الصورة:
# 1. التوليد من الصورة إلى النص
image_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/unidiffuser/unidiffuser_example_image.jpg"
init_image = load_image(image_url).resize((512, 512))

sample = pipe(image=init_image, num_inference_steps=20, guidance_scale=8.0)
i2t_text = sample.text[0]
print(i2t_text)

# 2. التوليد من النص إلى الصورة
sample = pipe(prompt=i2t_text, num_inference_steps=20, guidance_scale=8.0)
final_image = sample.images[0]
final_image.save("unidiffuser_image_variation_sample.png")
```

### تنويع النص

وبالمثل، يمكن إجراء تنويع النص على موجه إدخال باستخدام التوليد من النص إلى الصورة متبوعًا بالتوليد من الصورة إلى النص:

```python
import torch

from diffusers import UniDiffuserPipeline

device = "cuda"
model_id_or_path = "thu-ml/unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# يمكن إجراء تنويع النص من خلال التوليد من النص إلى الصورة متبوعًا بالتوليد من الصورة إلى النص:
# 1. التوليد من النص إلى الصورة
prompt = "فيل تحت البحر"

sample = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=8.0)
t2i_image = sample.images[0]
t2i_image.save("unidiffuser_text2img_sample_image.png")

# 2. التوليد من الصورة إلى النص
sample = pipe(image=t2i_image, num_inference_steps=20, guidance_scale=8.0)
final_prompt = sample.text[0]
print(final_prompt)
```

<Tip>

تأكد من مراجعة الدليل [Schedulers](../../using-diffusers/schedulers) لمعرفة كيفية استكشاف المقايضة بين سرعة الجدولة والجودة، وقسم [إعادة استخدام المكونات عبر الأنابيب](../../using-diffusers/loading#reuse-components-across-pipelines) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في أنابيب متعددة.

</Tip>

## UniDiffuserPipeline

[[autodoc]] UniDiffuserPipeline

- all
- __call__

## ImageTextPipelineOutput

[[autodoc]] pipelines.ImageTextPipelineOutput