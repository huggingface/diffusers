# PEFT

يدعم Diffusers تحميل المحولات مثل [LoRA](../../using-diffusers/loading_adapters) مع مكتبة [PEFT](https://huggingface.co/docs/peft/index) باستخدام فئة [`~loaders.peft.PeftAdapterMixin`]. يسمح هذا للنماذج في Diffusers مثل [`UNet2DConditionModel`] بتحميل محول.

<Tip>
راجع البرنامج التعليمي [الاستنتاج باستخدام PEFT](../../tutorials/using_peft_for_inference.md) للحصول على نظرة عامة حول كيفية استخدام PEFT في Diffusers للاستنتاج.
</Tip>

## PeftAdapterMixin

[[autodoc]] loaders.peft.PeftAdapterMixin