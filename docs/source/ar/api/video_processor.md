# معالج الفيديو

توفّر [`VideoProcessor`] واجهة برمجة تطبيقات موحدة لأنابيب الفيديو لإعداد المدخلات لتشفير VAE ومعالجة المخرجات بعد فك تشفيرها. يرث الصنف [`VaeImageProcessor`]، لذلك يتضمن تحويلات مثل تغيير الحجم والتحجيم والتحويل بين صور PIL Image و PyTorch و NumPy arrays.

## VideoProcessor

[[autodoc]] video_processor.VideoProcessor.preprocess_video

[[autodoc]] video_processor.VideoProcessor.postprocess_video