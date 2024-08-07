# T2I-Adapter  

[T2I-Adapter](https://hf.co/papers/2302.08453) هو نموذج خفيف الوزن يوفر إدخال شرطي إضافي للصورة (خط فني، كاني، رسم، عمق، وضع) للتحكم بشكل أفضل في توليد الصور. وهو مشابه لشبكة التحكم ControlNet، ولكنه أصغر بكثير (حوالي 77 مليون معامل وحجم ملف يبلغ حوالي 300 ميجابايت) لأنه يقوم بإدراج الأوزان فقط في شبكة U-Net بدلاً من نسخها وتدريبها.

يتوفر T2I-Adapter للتدريب فقط مع نموذج Stable Diffusion XL (SDXL).

سيتناول هذا الدليل برنامج النص البرمجي للتدريب [train_t2i_adapter_sdxl.py](https://github.com/huggingface/diffusers/blob/main/examples/t2i_adapter/train_t2i_adapter_sdxl.py) لمساعدتك على التعرف عليه، وكيف يمكنك تكييفه مع حالتك الاستخدامية الخاصة.

قبل تشغيل البرنامج النصي، تأكد من تثبيت المكتبة من المصدر:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

بعد ذلك، انتقل إلى مجلد الأمثلة الذي يحتوي على برنامج النص البرمجي للتدريب وقم بتثبيت التبعيات المطلوبة للبرنامج النصي الذي تستخدمه:

```bash
cd examples/t2i_adapter
pip install -r requirements.txt
```

<Tip>
🤗 Accelerate هي مكتبة تساعدك على التدريب على وحدات معالجة الرسوميات (GPU) أو وحدات معالجة الرسوميات (TPU) متعددة أو باستخدام الدقة المختلطة. سيقوم تلقائيًا بتكوين إعداد التدريب الخاص بك بناءً على أجهزتك وبيئتك. اطلع على الجولة السريعة للمكتبة 🤗 Accelerate [Quick tour](https://huggingface.co/docs/accelerate/quicktour) لمعرفة المزيد.
</Tip>

قم بتهيئة بيئة 🤗 Accelerate:

```bash
accelerate config
```

لإعداد بيئة 🤗 Accelerate الافتراضية دون اختيار أي تكوينات:

```bash
accelerate config default
```

أو إذا لم يدعم بيئتك غلافًا تفاعليًا، مثل دفتر الملاحظات، فيمكنك استخدام ما يلي:

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

أخيرًا، إذا كنت تريد تدريب نموذج على مجموعة بياناتك الخاصة، فراجع دليل [إنشاء مجموعة بيانات للتدريب](create_dataset) لمعرفة كيفية إنشاء مجموعة بيانات تعمل مع برنامج النص البرمجي للتدريب.

<Tip>
تسلط الأقسام التالية الضوء على أجزاء من برنامج النص البرمجي للتدريب المهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب البرنامج النصي بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فلا تتردد في قراءة البرنامج النصي [script](https://github.com/huggingface/diffusers/blob/main/examples/t2i_adapter/train_t2i_adapter_sdxl.py) وأخبرنا إذا كان لديك أي أسئلة أو مخاوف.
</Tip>

## معلمات البرنامج النصي

يوفر برنامج النص البرمجي للتدريب العديد من المعلمات لمساعدتك على تخصيص عملية تشغيل التدريب. يمكن العثور على جميع المعلمات ووصفاتها في دالة [`parse_args()`](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L233). يوفر قيم افتراضية لكل معلمة، مثل حجم دفعة التدريب ومعدل التعلم، ولكن يمكنك أيضًا تعيين قيمك الخاصة في أمر التدريب إذا رغبت في ذلك.

على سبيل المثال، لتنشيط تجميع التدرجات، أضف المعلمة `--gradient_accumulation_steps` إلى أمر التدريب:

```bash
accelerate launch train_t2i_adapter_sdxl.py \
----gradient_accumulation_steps=4
```

تم وصف العديد من المعلمات الأساسية والمهمة في دليل التدريب [Text-to-image](text2image#script-parameters)، لذلك يركز هذا الدليل فقط على معلمات T2I-Adapter ذات الصلة:

- `--pretrained_vae_model_name_or_path`: المسار إلى VAE مُدرب مسبقًا؛ من المعروف أن VAE الخاص بـ SDXL يعاني من عدم استقرار رقمي، لذلك تسمح هذه المعلمة بتحديد VAE أفضل [VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)

- `--crops_coords_top_left_h` و `--crops_coords_top_left_w`: إحداثيات الارتفاع والعرض المراد تضمينها في تضمينات إحداثيات القطع الخاصة بـ SDXL

- `--conditioning_image_column`: عمود الصور الشرطية في مجموعة البيانات

- `--proportion_empty_prompts`: نسبة موجهات الصور التي سيتم استبدالها بالسلاسل الفارغة

## برنامج النص البرمجي للتدريب

كما هو الحال مع معلمات البرنامج النصي، يتم توفير دليل تفصيلي لبرنامج النص البرمجي للتدريب في دليل التدريب [Text-to-image](text2image#training-script). بدلاً من ذلك، يلقي هذا الدليل نظرة على أجزاء البرنامج النصي ذات الصلة بـ T2I-Adapter.

يبدأ برنامج النص البرمجي للتدريب عن طريق إعداد مجموعة البيانات. ويشمل ذلك [تحويل النص إلى رموز](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L674) [applying transforms](https://github.com/huggingface/diffusers/blob/aab6de202c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L714) إلى الصور والصور الشرطية.

```py
conditioning_image_transforms = transforms.Compose(
[
transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
transforms.CenterCrop(args.resolution),
transforms.ToTensor(),
]
)
```

داخل دالة [`main()`](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L770)، يتم تحميل T2I-Adapter إما من مهايئ مُدرب مسبقًا أو يتم تهيئته بشكل عشوائي:

```py
if args.adapter_model_name_or_path:
logger.info("Loading existing adapter weights.")
t2iadapter = T2IAdapter.from_pretrained(args.adapter_model_name_or_path)
else:
logger.info("Initializing t2iadapter weights.")
t2iadapter = T2IAdapter(
in_channels=3,
channels=(320, 640, 1280, 1280),
num_res_blocks=2,
downscale_factor=16,
adapter_type="full_adapter_xl",
)
```

يتم تهيئة [المحسن](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L952) لمعلمات T2I-Adapter:

```py
params_to_optimize = t2iadapter.parameters()
optimizer = optimizer_class(
params_to_optimize,
lr=args.learning_rate,
betas=(args.adam_beta1, args.adam_beta2),
weight_decay=args.adam_weight_decay,
eps=args.adam_epsilon,
)
```

أخيرًا، في [حلقة التدريب](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L1086)، يتم تمرير الصورة الشرطية والتضمين النصي إلى شبكة U-Net للتنبؤ ببقايا الضوضاء:

```py
t2iadapter_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
down_block_additional_residuals = t2iadapter(t2iadapter_image)
down_block_additional_residuals = [
sample.to(dtype=weight_dtype) for sample in down_block_additional_residuals
]

model_pred = unet(
inp_noisy_latents,
timesteps,
encoder_hidden_states=batch["prompt_ids"],
added_cond_kwargs=batch["unet_added_conditions"],
down_block_additional_residuals=down_block_additional_residuals,
).sample
```

إذا كنت تريد معرفة المزيد حول كيفية عمل حلقة التدريب، فراجع البرنامج التعليمي [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline) الذي يفكك النمط الأساسي لعملية إزالة التشويش.

## تشغيل البرنامج النصي

الآن أنت مستعد لتشغيل برنامج النص البرمجي للتدريب! 🚀

بالنسبة لهذا المثال التدريبي، ستستخدم مجموعة بيانات [fusing/fill50k](https://huggingface.co/datasets/fusing/fill50k). يمكنك أيضًا إنشاء مجموعة بياناتك الخاصة واستخدامها إذا كنت تريد (راجع دليل [إنشاء مجموعة بيانات للتدريب](https://moon-ci-docs.huggingface.co/docs/diffusers/pr_5512/en/training/create_dataset)).

قم بتعيين متغير البيئة `MODEL_DIR` إلى معرف نموذج على Hub أو مسار إلى نموذج محلي و`OUTPUT_DIR` إلى المكان الذي تريد حفظ النموذج فيه.

قم بتنزيل الصور التالية لشرط تدريبك:

```bash
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

<Tip>
لرصد تقدم التدريب باستخدام Weights & Biases، أضف المعلمة `--report_to=wandb` إلى أمر التدريب. ستحتاج أيضًا إلى إضافة المعلمات `--validation_image`، و`--validation_prompt`، و`--validation_steps` إلى أمر التدريب لتتبع النتائج. يمكن أن يكون هذا مفيدًا جدًا في تصحيح أخطاء النموذج وعرض النتائج الوسيطة.
</Tip>

```bash
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="path to save model"

accelerate launch train_t2i_adapter_sdxl.py \
--pretrained_model_name_or_path=$MODEL_DIR \
--output_dir=$OUTPUT_DIR \
--dataset_name=fusing/fill50k \
--mixed_precision="fp16" \
--resolution=1024 \
--learning_rate=1e-5 \
--max_train_steps=15000 \
--validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
--validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
--validation_steps=100 \
--train_batch_size=1 \
--gradient_accumulation_steps=4 \
--report_to="wandb" \
--seed=42 \
--push_to_hub
```

بمجرد اكتمال التدريب، يمكنك استخدام T2I-Adapter للاستنتاج:

```py
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteSchedulerTest
from diffusers.utils import load_image
import torch

adapter = T2IAdapter.from_pretrained("path/to/adapter", torch_dtype=torch.float16)
pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", adapter=adapter, torch_dtype=torch.float16
)

pipeline.scheduler = EulerAncestralDiscreteSchedulerTest.from_config(pipe.scheduler.config)
pipeline.enable_xformers_memory_efficient_attention()
pipeline.enable_model_cpu_offload()

control_image = load_image("./conditioning_image_1.png")
prompt = "pale golden rod circle with old lace background"

generator = torch.manual_seed(0)
image = pipeline(
prompt, image=control_image, generator=generator
).images[0]
image.save("./output.png")
```

## الخطوات التالية

تهانينا على تدريب نموذج T2I-Adapter! 🎉 لمزيد من المعلومات:

- اقرأ منشور المدونة [Efficient Controllable Generation for SDXL with T2I-Adapters](https://huggingface.co/blog/t2i-sdxl-adapters) لمعرفة المزيد من التفاصيل حول النتائج التجريبية من فريق T2I-Adapter.