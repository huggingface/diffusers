## دمج الرموز 

[دمج الرموز](https://huggingface.co/papers/2303.17604) (ToMe) يقوم بدمج الرموز/الرقعات الزائدة تدريجياً في عملية التقديم لشبكة تعتمد على المحول، والتي يمكن أن تسرع زمن الاستدلال لـ [`StableDiffusionPipeline`].

قم بتثبيت ToMe من `pip`:

```bash
pip install tomesd
```

يمكنك استخدام ToMe من مكتبة [`tomesd`](https://github.com/dbolya/tomesd) مع دالة [`apply_patch`](https://github.com/dbolya/tomesd?tab=readme-ov-file#usage):

```diff
from diffusers import StableDiffusionPipeline
import torch
import tomesd

pipeline = StableDiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True,
).to("cuda")
+ tomesd.apply_patch(pipeline, ratio=0.5)

image = pipeline("a photo of an astronaut riding a horse on mars").images[0]
```

تعرض دالة `apply_patch` عددًا من [الحجج](https://github.com/dbolya/tomesd#usage) للمساعدة في تحقيق التوازن بين سرعة استدلال الأنابيب وجودة الرموز المولدة. أهم حجة هي `ratio` التي تتحكم في عدد الرموز التي يتم دمجها أثناء عملية التقديم.

كما هو مذكور في [الورقة](https://huggingface.co/papers/2303.17604)، يمكن لـ ToMe الحفاظ بشكل كبير على جودة الصور المولدة مع تعزيز سرعة الاستدلال. من خلال زيادة `ratio`، يمكنك تسريع الاستدلال بشكل أكبر، ولكن على حساب بعض تدهور جودة الصورة.

لاختبار جودة الصور المولدة، قمنا باختيار بعض المطالبات من [Parti Prompts](https://parti.research.google/) وأجرينا الاستدلال باستخدام [`StableDiffusionPipeline`] مع الإعدادات التالية:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/tome/tome_samples.png">
</div>

لم نلاحظ أي انخفاض كبير في جودة العينات المولدة، ويمكنك الاطلاع على العينات المولدة في هذا [تقرير WandB](https://wandb.ai/sayakpaul/tomesd-results/runs/23j4bj3i?workspace=). إذا كنت مهتمًا بتكرار هذه التجربة، فاستخدم هذا [النص البرمجي](https://gist.github.com/sayakpaul/8cac98d7f22399085a060992f411ecbd).

## المعايير القياسية

كما قمنا باختبار تأثير `tomesd` على [`StableDiffusionPipeline`] مع [xFormers](https://huggingface.co/docs/diffusers/optimization/xformers) الممكّنة عبر عدة دقات للصور. تم الحصول على النتائج من معالجات الرسوميات A100 و V100 في بيئة التطوير التالية:

```bash
- `diffusers` version: 0.15.1
- Python version: 3.8.16
- PyTorch version (GPU?): 1.13.1+cu116 (True)
- Huggingface_hub version: 0.13.2
- Transformers version: 4.27.2
- Accelerate version: 0.18.0
- xFormers version: 0.0.16
- tomesd version: 0.1.2
```

لإعادة إنتاج هذا المعيار القياسي، لا تتردد في استخدام هذا [النص البرمجي](https://gist.github.com/sayakpaul/27aec6bca7eb7b0e0aa4112205850335). يتم الإبلاغ عن النتائج بالثواني، وحيثما ينطبق، نقوم بالإبلاغ عن النسبة المئوية لزيادة السرعة على الأنبوب الأساسي عند استخدام ToMe و ToMe + xFormers.

| **GPU**  | **Resolution** | **Batch size** | **Vanilla** | **ToMe**       | **ToMe + xFormers** |
|----------|----------------|----------------|-------------|----------------|---------------------|
| **A100** |            512 |             10 |        6.88 | 5.26 (+23.55%) |      4.69 (+31.83%) |
|          |            768 |             10 |         OOM |          14.71 |                  11 |
|          |                |              8 |         OOM |          11.56 |                8.84 |
|          |                |              4 |         OOM |           5.98 |                4.66 |
|          |                |              2 |        4.99 | 3.24 (+35.07%) |       2.1 (+37.88%) |
|          |                |              1 |        3.29 | 2.24 (+31.91%) |       2.03 (+38.3%) |
|          |           1024 |             10 |         OOM |            OOM |                 OOM |
|          |                |              8 |         OOM |            OOM |                 OOM |
|          |                |              4 |         OOM |          12.51 |                9.09 |
|          |                |              2 |         OOM |           6.52 |                4.96 |
|          |                |              1 |         6.4 | 3.61 (+43.59%) |      2.81 (+56.09%) |
| **V100** |            512 |             10 |         OOM |          10.03 |                9.29 |
|          |                |              8 |         OOM |           8.05 |                7.47 |
|          |                |              4 |         5.7 |  4.3 (+24.56%) |      3.98 (+30.18%) |
|          |                |              2 |        3.14 | 2.43 (+22.61%) |      2.27 (+27.71%) |
|          |                |              1 |        1.88 | 1.57 (+16.49%) |      1.57 (+16.49%) |
|          |            768 |             10 |         OOM |            OOM |               23.67 |
|          |                |              8 |         OOM |            OOM |               18.81 |
|          |                |              4 |         OOM |          11.81 |                 9.7 |
|          |                |              2 |         OOM |           6.27 |                 5.2 |
|          |                |              1 |        5.43 | 3.38 (+37.75%) |      2.82 (+48.07%) |
|          |           1024 |             10 |         OOM |            OOM |                 OOM |
|          |                |              8 |         OOM |            OOM |                 OOM |
|          |                |              4 |         OOM |            OOM |               19.35 |
|          |                |              2 |         OOM |             13 |               10.78 |
|          |                |              1 |         OOM |           6.66 |                5.54 |

كما هو موضح في الجداول أعلاه، تصبح زيادة السرعة من `tomesd` أكثر وضوحًا لدقات الصور الأكبر. ومن المثير للاهتمام أيضًا ملاحظة أنه مع `tomesd`، من الممكن تشغيل الأنبوب على دقة أعلى مثل 1024x1024. قد تتمكن من تسريع الاستدلال بشكل أكبر باستخدام [`torch.compile`](torch2.0).