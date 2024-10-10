# xFormers

نوصي باستخدام [xFormers](https://github.com/facebookresearch/xformers) لكل من الاستنتاج والتدريب. في اختباراتنا، سمحت التحسينات التي تم إجراؤها في كتل الاهتمام بزيادة السرعة وتقليل استهلاك الذاكرة.

قم بتثبيت xFormers من `pip`:

```bash
pip install xformers
```

> تتطلب حزمة xFormers `pip` أحدث إصدار من PyTorch. إذا كنت بحاجة إلى استخدام إصدار سابق من PyTorch، فإننا نوصي بـ [تثبيت xFormers من المصدر](https://github.com/facebookresearch/xformers#installing-xformers).

بعد تثبيت xFormers، يمكنك استخدام `enable_xformers_memory_efficient_attention()` لزيادة سرعة الاستدلال وتقليل استهلاك الذاكرة كما هو موضح في هذا [القسم](memory#memory-efficient-attention).

> وفقًا لهذا [المشكلة](https://github.com/huggingface/diffusers/issues/2234#issuecomment-1416931212)، لا يمكن استخدام xFormers `v0.0.16` للتدريب (التنغيم الدقيق أو DreamBooth) في بعض وحدات معالجة الرسوميات (GPU). إذا لاحظت هذه المشكلة، فقم بتثبيت إصدار التطوير كما هو موضح في تعليقات المشكلة.