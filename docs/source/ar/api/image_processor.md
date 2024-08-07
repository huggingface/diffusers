# معالج صور VAE

يوفر [`VaeImageProcessor`] واجهة برمجة تطبيقات موحدة لـ [`StableDiffusionPipeline`] لإعداد إدخالات الصور لتشفير VAE ومعالجة الإخراج بعد فك تشفيرها. ويشمل ذلك تحويلات مثل تغيير الحجم والتحجيم والتحويل بين صور PIL ومصفوفات PyTorch و NumPy.

تقبل جميع الأنابيب التي تحتوي على [`VaeImageProcessor`] صور PIL أو مصفوفات PyTorch أو NumPy كإدخالات للصور وتعيد الإخراج بناءً على وسيطة `output_type` التي يحددها المستخدم. يمكنك تمرير الصور المشفرة مباشرة إلى الأنبوب وإعادة الإخراج من الأنبوب كإخراج محدد باستخدام وسيطة `output_type` (على سبيل المثال `output_type="latent"`). يسمح لك ذلك بأخذ الصور المولدة من خط أنابيب واحد وتمريرها كإدخال إلى خط أنابيب آخر دون الخروج من مساحة الإدخال. كما يسهل استخدام خطوط أنابيب متعددة معًا عن طريق تمرير مصفوفات PyTorch مباشرة بين خطوط الأنابيب المختلفة.

## VaeImageProcessor

[[autodoc]] image_processor.VaeImageProcessor

## VaeImageProcessorLDM3D

يقبل [`VaeImageProcessorLDM3D`] إدخالات RGB و depth ويعيد إخرج إدخالات RGB و depth.

[[autodoc]] image_processor.VaeImageProcessorLDM3D

## PixArtImageProcessor

[[autodoc]] image_processor.PixArtImageProcessor

## IPAdapterMaskProcessor

[[autodoc]] image_processor.IPAdapterMaskProcessor