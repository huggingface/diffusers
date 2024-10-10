# DeepCache 

يعمل DeepCache على تسريع StableDiffusionPipeline و StableDiffusionXLPipeline من خلال التخزين الاستراتيجي وإعادة استخدام الميزات عالية المستوى مع تحديث الميزات منخفضة المستوى بكفاءة عن طريق الاستفادة من بنية U-Net. 

ابدأ بتثبيت DeepCache:

```bash
pip install DeepCache
```
ثم قم بتحميل وتمكين DeepCacheSDHelper: 

```diff
  import torch
  from diffusers import StableDiffusionPipeline
  pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16).to("cuda")

+ from DeepCache import DeepCacheSDHelper
+ helper = DeepCacheSDHelper(pipe=pipe)
+ helper.set_params(
+     cache_interval=3,
+     cache_branch_id=0,
+ )
+ helper.enable()

  image = pipe("a photo of an astronaut on a moon").images[0]
```

تقبل طريقة set_params وسيطين: cache_interval وcache_branch_id. يشير cache_interval إلى تكرار تخزين الميزات، ويتم تحديده على أنه عدد الخطوات بين كل عملية تخزين. ويحدد cache_branch_id الفرع المسؤول عن تنفيذ عمليات التخزين (مرتبة من الطبقة الأضحل إلى الأعمق). 

إن اختيار قيمة أقل لـ cache_branch_id أو قيمة أكبر لـ cache_interval يمكن أن يؤدي إلى زيادة سرعة الاستدلال على حساب تقليل جودة الصورة (يمكن العثور على تجارب الحجب الخاصة بهذين المعاملين في الورقة البحثية). بمجرد ضبط هذه الوسائط، استخدم طرق "enable" أو "disable" لتنشيط أو إلغاء تنشيط "DeepCacheSDHelper". 

<div class="flex justify-center">
    <img src="https://github.com/horseee/Diffusion_DeepCache/raw/master/static/images/example.png">
</div>

يمكنك العثور على المزيد من العينات المولدة (خط الأنابيب الأصلي مقابل DeepCache) ووقت الاستدلال المقابل في تقرير WandB. يتم اختيار المطالبات بشكل عشوائي من مجموعة بيانات MS-COCO 2017. 

## المعيار المرجعي 

لقد اختبرنا مدى تسريع DeepCache لـ Stable Diffusion v2.1 مع 50 خطوة استدلال على NVIDIA RTX A5000، باستخدام تكوينات مختلفة للقرار وحجم الدفعة وفاصل التخزين (I) وفرع التخزين (B). 

| القرار | حجم الدفعة | الأصلي | DeepCache(I=3, B=0) | DeepCache(I=5, B=0) | DeepCache(I=5, B=1) | 
| --- | --- | --- | --- | --- | --- | 
| 512 | 8 | 15.96 | 6.88(2.32x) | 5.03(3.18x) | 7.27(2.20x) | 
| | 4 | 8.39 | 3.60(2.33x) | 2.62(3.21x) | 3.75(2.24x) | 
| | 1 | 2.61 | 1.12(2.33x) | 0.81(3.24x) | 1.11(2.35x) | 
| 768 | 8 | 43.58 | 18.99(2.29x) | 13.96(3.12x) | 21.27(2.05x) | 
| | 4 | 22.24 | 9.67(2.30x) | 7.10(3.13x) | 10.74(2.07x) | 
| | 1 | 6.33 | 2.72(2.33x) | 1.97(3.21x) | 2.98(2.12x) | 
| 1024 | 8 | 101.95 | 45.57(2.24x) | 33.72(3.02x) | 53.00(1.92x) | 
| | 4 | 49.25 | 21.86(2.25x) | 16.19(3.04x) | 25.78(1.91x) | 
| | 1 | 13.83 | 6.07(2.28x) | 4.43(3.12x) | 7.15(1.93x) |