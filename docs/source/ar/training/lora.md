# LoRA

<Tip warning={true}>
هذا تجريبي وقد يتغير API في المستقبل.
</Tip>

[LoRA (Low-Rank Adaptation of Large Language Models)](https://hf.co/papers/2106.09685) هي تقنية تدريب خفيفة الوزن وشائعة تقلل بشكل كبير من عدد المعلمات القابلة للتدريب. تعمل عن طريق إدخال عدد أقل من الأوزان الجديدة في النموذج، وتدريب هذه الأوزان فقط. يجعل التدريب باستخدام LoRA أسرع، وأكثر كفاءة في استخدام الذاكرة، وينتج أوزان نموذج أصغر (بضع مئات من الميغابايت)، والتي يسهل تخزينها ومشاركتها. يمكن أيضًا دمج LoRA مع تقنيات التدريب الأخرى مثل DreamBooth لتسريع التدريب.

<Tip>
LoRA متعدد الاستخدامات ومدعوم لـ  [DreamBooth](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py), [Kandinsky 2.2](https://github.com/huggingface/diffusers/blob/main/examples/kandinsky2_2/text_to_image/train_text_to_image_lora_decoder.py),  [Stable Diffusion XL](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py), [text-to-image](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py), [Wuerstchen](https://github.com/huggingface/diffusers/blob/main/examples/wuerstchen/text_to_image/train_text_to_image_lora_prior.py) .
</Tip>

سيتعمق هذا الدليل في النص البرمجي [train_text_to_image_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py) لمساعدتك على التعرف عليه بشكل أفضل، وكيف يمكنك تكييفه مع حالتك الاستخدامية الخاصة.

قبل تشغيل النص البرمجي، تأكد من تثبيت المكتبة من المصدر:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

انتقل إلى مجلد المثال الذي يحتوي على النص البرمجي للتدريب وقم بتثبيت التبعيات المطلوبة للنص البرمجي الذي تستخدمه:

<hfoptions id="installation">
<hfoption id="PyTorch">

```bash
cd examples/text_to_image
pip install -r requirements.txt
```

</hfoption>
<hfoption id="Flax">

```bash
cd examples/text_to-image
pip install -r requirements_flax.txt
```

</hfoption>
</hfoptions>

<Tip>
🤗 Accelerate هي مكتبة تساعدك على التدريب على وحدات GPU/TPUs متعددة أو باستخدام الدقة المختلطة. سيقوم تلقائيًا بتكوين إعداد التدريب الخاص بك بناءً على أجهزتك وبيئتك. الق نظرة على جولة 🤗 Accelerate [سريعة](https://huggingface.co/docs/accelerate/quicktour) لمعرفة المزيد.
</Tip>

قم بتهيئة بيئة 🤗 Accelerate:

```bash
accelerate config
```

لإعداد بيئة 🤗 Accelerate الافتراضية دون اختيار أي تكوينات:

```bash
accelerate config default
```

أو إذا لم يدعم بيئتك غلافًا تفاعليًا، مثل دفتر الملاحظات، فيمكنك استخدام:

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

أخيرًا، إذا كنت تريد تدريب نموذج على مجموعة بياناتك الخاصة، فراجع دليل [إنشاء مجموعة بيانات للتدريب](create_dataset) لمعرفة كيفية إنشاء مجموعة بيانات تعمل مع النص البرمجي للتدريب.

<Tip>
تسلط الأقسام التالية الضوء على أجزاء من النص البرمجي للتدريب المهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب النص البرمجي بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فلا تتردد في قراءة النص البرمجي [هنا](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/text_to_image_lora.py) وأخبرنا إذا كان لديك أي أسئلة أو مخاوف.
</Tip>

## معلمات النص البرمجي

يحتوي النص البرمجي للتدريب على العديد من المعلمات لمساعدتك على تخصيص عملية تشغيل التدريب. يمكن العثور على جميع المعلمات ووصفاتها في دالة [`parse_args()`](https://github.com/huggingface/diffusers/blob/dd9a5caf61f04d11c0fa9f3947b69ab0010c9a0f/examples/text_to_image/train_text_to_image_lora.py#L85). يتم توفير القيم الافتراضية لمعظم المعلمات والتي تعمل بشكل جيد إلى حد ما، ولكن يمكنك أيضًا تعيين قيمك الخاصة في أمر التدريب إذا كنت ترغب في ذلك.

على سبيل المثال، لزيادة عدد حقبات التدريب:

```bash
accelerate launch train_text_to_image_lora.py \
--num_train_epochs=150 \
```

تم وصف العديد من المعلمات الأساسية والمهمة في دليل تدريب [Text-to-image](text2image#script-parameters)، لذلك يركز هذا الدليل فقط على المعلمات ذات الصلة بـ LoRA:

- `--rank`: البعد الداخلي لمصفوفات الرتبة المنخفضة التي سيتم تدريبها؛ يعني الرتبة الأعلى المزيد من المعلمات القابلة للتدريب
- `--learning_rate`: معدل التعلم الافتراضي هو 1e-4، ولكن مع LoRA، يمكنك استخدام معدل تعلم أعلى

## النص البرمجي للتدريب

يمكن العثور على رمز معالجة مجموعة البيانات و حلقة التدريب في دالة [`main()`](https://github.com/huggingface/diffusers/blob/dd9a5caf61f04d11c0fa9f3947b69ab0010c9a0f/examples/text_to_image/train_text_to_image_lora.py#L371)، وإذا كنت بحاجة إلى تكييف النص البرمجي للتدريب، فهذا هو المكان الذي ستجري فيه تغييراتك.

كما هو الحال مع معلمات النص البرمجي، يتم توفير دليل تفصيلي لنص البرنامج النصي للتدريب في دليل تدريب [Text-to-image](text2image#training-script). بدلاً من ذلك، يلقي هذا الدليل نظرة على أجزاء النص البرمجي ذات الصلة بـ LoRA.

<hfoptions id="lora">
<hfoption id="UNet">

يستخدم Diffusers [`~peft.LoraConfig`] من مكتبة [PEFT](https://hf.co/docs/peft) لتهيئة معلمات محول LoRA مثل الرتبة، والألفا، والوحدات النمطية التي سيتم إدراج أوزان LoRA فيها. يتم إضافة المحول إلى UNet، ويتم تصفية طبقات LoRA فقط للتحسين في `lora_layers`.

```py
unet_lora_config = LoraConfig(
r=args.rank,
lora_alpha=args.rank,
init_lora_weights="gaussian",
target_modules=["to_k"، "to_q"، "to_v"، "to_out.0"]،
)

unet.add_adapter(unet_lora_config)
lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
```

</hfoption>
<hfoption id="text encoder">

يدعم Diffusers أيضًا ضبط دقيق لترميز النص باستخدام LoRA من مكتبة [PEFT](https://hf.co/docs/peft) عند الضرورة مثل ضبط دقيق لـ Stable Diffusion XL (SDXL). يتم استخدام [`~peft.LoraConfig`] لتهيئة معلمات محول LoRA والتي يتم إضافتها بعد ذلك إلى ترميز النص، ويتم تصفية طبقات LoRA فقط للتدريب.

```py
text_lora_config = LoraConfig(
r=args.rank,
lora_alpha=args.rank,
init_lora_weights="gaussian",
target_modules=["q_proj"، "k_proj"، "v_proj"، "out_proj"]،
)

text_encoder_one.add_adapter(text_lora_config)
text_encoder_two.add_adapter(text_lora_config)
text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
text_lora_parameters_two = list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))
```

</hfoption>
</hfoptions>

يتم تهيئة [المحسن](https://github.com/huggingface/diffusers/blob/e4b8f173b97731686e290b2eb98e7f5df2b1b322/examples/text_to_image/train_text_to_image_lora.py#L529) مع `lora_layers` لأن هذه هي الأوزان الوحيدة التي سيتم تحسينها:

```py
optimizer = optimizer_cls(
lora_layers،
lr=args.learning_rate،
betas=(args.adam_beta1, args.adam_beta2)،
weight_decay=args.adam_weight_decay،
eps=args.adam_epsilon،
)
```

بصرف النظر عن إعداد طبقات LoRA، فإن نص البرنامج النصي للتدريب هو نفسه تقريبًا مثل train_text_to_image.py!

## إطلاق النص البرمجي

بمجرد إجراء جميع التغييرات أو إذا كنت راضيًا عن التكوين الافتراضي، فأنت جاهز لإطلاق النص البرمجي للتدريب! 🚀

دعونا نتدرب على مجموعة بيانات [Naruto BLIP captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) لإنشاء شخصيات ناروتو الخاصة بك. قم بتعيين متغيرات البيئة `MODEL_NAME` و `DATASET_NAME` إلى النموذج ومجموعة البيانات على التوالي. يجب عليك أيضًا تحديد مكان حفظ النموذج في `OUTPUT_DIR`، واسم النموذج لحفظه على Hub مع `HUB_MODEL_ID`. يقوم النص البرمجي بإنشاء وحفظ الملفات التالية في مستودعك:

- نقاط تفتيش النموذج المحفوظة
- `pytorch_lora_weights.safetensors` (أوزان LoRA المدربة)

إذا كنت تتدرب على أكثر من وحدة GPU واحدة، فأضف المعلمة `--multi_gpu` إلى أمر `accelerate launch`.

<Tip warning={true}>
تستغرق عملية التدريب الكاملة ~5 ساعات على وحدة معالجة الرسوميات 2080 Ti GPU مع 11 جيجابايت من VRAM.
</Tip>

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/sddata/finetune/lora/naruto"
export HUB_MODEL_ID="naruto-lora"
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME \
    --dataloader_num_workers=8 \
    --resolution=512 \
    --center_crop \
    --random_flip \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=15000 \
    --learning_rate=1e-04 \
    --max_grad_norm=1 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=0 \
    --output_dir=${OUTPUT_DIR} \
    --push_to_hub \
    --hub_model_id=${HUB_MODEL_ID} \
    --report_to=wandb \
    --checkpointing_steps=500 \
    --validation_prompt="A naruto with blue eyes." \
    --seed=1337
```

بمجرد اكتمال التدريب، يمكنك استخدام نموذجك للاستنتاج:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5"، torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("path/to/lora/model"، weight_name="pytorch_lora_weights.safetensors")
image = pipeline("A naruto with blue eyes").images[0]
```

## الخطوات التالية

تهانينا على تدريب نموذج جديد باستخدام LoRA! لمعرفة المزيد عن كيفية استخدام نموذجك الجديد، قد تكون الأدلة التالية مفيدة:

- تعرف على كيفية [تحميل تنسيقات LoRA المختلفة](../using-diffusers/loading_adapters#LoRA) المدربة باستخدام مدربين مجتمعيين مثل Kohya و TheLastBen.
- تعرف على كيفية استخدام و [دمج عدة LoRAs](../tutorials/using_peft_for_inference) مع PEFT للاستنتاج.