# نظرة عامة 

🤗 Diffusers توفر مجموعة من النصوص البرمجية للتدريب لتمكينك من تدريب نماذج الانتشار الخاصة بك. يمكنك إيجاد جميع نصوصنا التدريبية في [diffusers/examples](https://github.com/huggingface/diffusers/tree/main/examples).

كل نص تدريبي هو:

- **مكتفي ذاتياً**: النص التدريبي لا يعتمد على أي ملفات محلية، وجميع الحزم المطلوبة لتشغيل النص يتم تثبيتها من ملف `requirements.txt`.

- **سهل التعديل**: نصوص التدريب هي مثال على كيفية تدريب نموذج الانتشار لمهمة محددة ولن تعمل بشكل مباشر لكل سيناريو تدريبي. من المرجح أن تحتاج إلى تكييف نص التدريب مع حالتك الاستخدامية المحددة. لمساعدتك في ذلك، قمنا بالكشف الكامل عن كود معالجة البيانات وحلقة التدريب حتى تتمكن من تعديلها لاستخدامك الخاص.

- **صديق للمبتدئين**: تم تصميم نصوص التدريب لتكون سهلة الفهم للمبتدئين، بدلاً من تضمين أحدث الأساليب المتقدمة للحصول على أفضل النتائج وأكثرها تنافسية. يتم استبعاد أي طرق تدريب نعتبرها معقدة عن قصد.

- **وحيد الغرض**: تم تصميم كل نص تدريبي لغرض واحد صريح للحفاظ على سهولة القراءة والفهم.

تشمل مجموعة نصوص التدريب الحالية لدينا ما يلي:

| التدريب | دعم SDXL | دعم LoRA | دعم Flax |
|---|---|---|---|
| [توليد الصور غير المشروط](https://github.com/huggingface/diffusers/tree/main/examples/unconditional_image_generation) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb) |  |  |  |
| [النص إلى الصورة](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) | 👍 | 👍 | 👍 |
| [الانعكاس النصي](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb) |  |  | 👍 |
| [DreamBooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_training.ipynb) | 👍 | 👍 | 👍 |
| [ControlNet](https://github.com/huggingface/diffusers/tree/main/examples/controlnet) | 👍 |  | 👍 |
| [InstructPix2Pix](https://github.com/huggingface/diffusers/tree/main/examples/instruct_pix2pix) | 👍 |  |  |
| [انتشار مخصص](https://github.com/huggingface/diffusers/tree/main/examples/custom_diffusion) |  |  |  |
| [T2I-Adapters](https://github.com/huggingface/diffusers/tree/main/examples/t2i_adapter) | 👍 |  |  |
| [Kandinsky 2.2](https://github.com/huggingface/diffusers/tree/main/examples/kandinsky2_2/text_to_image) |  | 👍 |  |
| [Wuerstchen](https://github.com/huggingface/diffusers/tree/main/examples/wuerstchen/text_to_image) |  | 👍 |  |

يتم صيانة هذه الأمثلة بشكل **نشط**، لذا لا تتردد في فتح مشكلة إذا لم تكن تعمل كما هو متوقع. إذا شعرت بأنه يجب تضمين مثال تدريبي آخر، فأنت أكثر من مرحب بك لبدء [طلب ميزة](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feature_request.md&title=) لمناقشة فكرتك معنا وما إذا كانت تلبي معاييرنا من حيث الاكتفاء الذاتي وسهولة التعديل وسهولة الاستخدام والغرض الواحد.

## التثبيت

تأكد من أنه يمكنك تشغيل أحدث إصدارات نصوص المثال بنجاح عن طريق تثبيت المكتبة من المصدر في بيئة افتراضية جديدة:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

ثم انتقل إلى مجلد نص التدريب (على سبيل المثال، [DreamBooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)) وقم بتثبيت ملف `requirements.txt`. تمتلك بعض نصوص التدريب ملف متطلبات محدد لـ SDXL أو LoRA أو Flax. إذا كنت تستخدم أحد هذه النصوص، فتأكد من تثبيت ملف المتطلبات المقابل.

```bash
cd examples/dreambooth
pip install -r requirements.txt
# لتدريب SDXL مع DreamBooth
pip install -r requirements_sdxl.txt
```

للتسريع من التدريب وتقليل استخدام الذاكرة، نوصي بما يلي:

- استخدم PyTorch 2.0 أو أعلى لاستخدام [scaled dot product attention](../optimization/torch2.0#scaled-dot-product-attention) تلقائيًا أثناء التدريب (لا يلزم إجراء أي تغييرات على كود التدريب)

- قم بتثبيت [xFormers](../optimization/xformers) لتمكين الاهتمام بكفاءة الذاكرة