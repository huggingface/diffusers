# فك تشفير الاتساق

يمكن استخدام فك تشفير الاتساق لفك تشفير المخفيات من denoising UNet في [`StableDiffusionPipeline`]. تم تقديم هذا الفك في [التقرير الفني DALL-E 3](https://openai.com/dall-e-3).

يمكن العثور على الكود الأصلي في [openai/consistencydecoder](https://github.com/openai/consistencydecoder).

<Tip warning={true}>
يتم دعم الاستنتاج حاليًا لمرتين فقط.
</Tip>

لم يكن من الممكن المساهمة في خط الأنابيب بدون مساعدة [madebyollin](https://github.com/madebyollin) و [mrsteyk](https://github.com/mrsteyk) من [هذه القضية](https://github.com/openai/consistencydecoder/issues/1).

## ConsistencyDecoderVAE

[[autodoc]] ConsistencyDecoderVAE

- all
- Decoder