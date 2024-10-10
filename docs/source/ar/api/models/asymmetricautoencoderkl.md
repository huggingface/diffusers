# AsymmetricAutoencoderKL

تم تحسين نموذج VAE الأكبر (النموذج التلقائي التبايني) مع فقدان KL لمهمة الإكمال: [تصميم VQGAN غير متماثل أفضل لـ StableDiffusion] (https://arxiv.org/abs/2306.04632) بواسطة Zixin Zhu، Xuelu Feng، Dongdong Chen، Jianmin Bao، Le Wang، Yinpeng Chen، Lu Yuan، Gang Hua.

المستخلص من الورقة هو:

*StableDiffusion هو مولد نص إلى صورة ثوري يثير ضجة في عالم إنشاء الصور وتحريرها. على عكس الطرق التقليدية التي تتعلم نموذج الانتشار في مساحة البكسل، يتعلم StableDiffusion نموذج الانتشار في مساحة الكمون عبر VQGAN، مما يضمن الكفاءة والجودة. فهو لا يدعم مهام إنشاء الصور فحسب، بل يمكّن أيضًا من تحرير الصور الفعلية، مثل إكمال الصور والتحرير المحلي. ومع ذلك، لاحظنا أن VQGAN الفانيليا المستخدم في StableDiffusion يؤدي إلى فقدان كبير في المعلومات، مما يتسبب في تشوهات حتى في مناطق الصورة غير المعدلة. ولهذه الغاية، نقترح VQGAN غير متماثل جديد بتصميمين بسيطين. أولاً، بالإضافة إلى الإدخال من الترميز، يحتوي فك الترميز على فرع شرطي يتضمن معلومات من القواعد السابقة المحددة للمهمة، مثل منطقة الصورة غير المُقنعة في الإكمال. ثانيًا، فك التشفير أثقل بكثير من الترميز، مما يسمح باسترداد أكثر تفصيلاً مع زيادة تكلفة الاستدلال الإجمالية زيادة طفيفة فقط. وتكلفة تدريب VQGAN غير المتماثل لدينا رخيصة، ونحن بحاجة فقط إلى إعادة تدريب فك تشفير غير متماثل جديد مع الحفاظ على ترميز VQGAN الفانيليا و StableDiffusion دون تغيير. يمكن استخدام VQGAN غير المتماثل لدينا على نطاق واسع في طرق الإكمال والتحرير المحلية المستندة إلى StableDiffusion. تُظهر التجارب المستفيضة أنه يمكنها تحسين أداء الإكمال والتحرير بشكل كبير، مع الحفاظ على القدرة الأصلية على إنشاء نص إلى صورة. الكود متاح في https://github.com/buxiangzhiren/Asymmetric_VQGAN*

يمكن العثور على نتائج التقييم في القسم 4.1 من الورقة الأصلية.

## نقاط التفتيش المتاحة

* [https://huggingface.co/cross-attention/asymmetric-autoencoder-kl-x-1-5](https://huggingface.co/cross-attention/asymmetric-autoencoder-kl-x-1-5)
* [https://huggingface.co/cross-attention/asymmetric-autoencoder-kl-x-2](https://huggingface.co/cross-attention/asymmetric-autoencoder-kl-x-2)

## مثال على الاستخدام

```python
from diffusers import AsymmetricAutoencoderKL, StableDiffusionInpaintPipeline
from diffusers.utils import load_image, make_image_grid

prompt = "a photo of a person with beard"
img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/celeba_hq_256.png"
mask_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/mask_256.png"

original_image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
pipe.vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
pipe.to("cuda")

image = pipe(prompt=prompt, image=original_image, mask_image=mask_image).images[0]
make_image_grid([original_image, mask_image, image], rows=1, cols=3)
```

## AsymmetricAutoencoderKL

[[autodoc]] models.autoencoders.autoencoder_asym_kl.AsymmetricAutoencoderKL

## AutoencoderKLOutput

[[autodoc]] models.autoencoders.autoencoder_kl.AutoencoderKLOutput

## DecoderOutput

[[autodoc]] models.autoencoders.vae.DecoderOutput
