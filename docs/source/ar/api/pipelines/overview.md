# Pipelines
توفر خطوط الأنابيب طريقة بسيطة لتشغيل أحدث نماذج الانتشار في الاستنتاج عن طريق تجميع جميع المكونات اللازمة (نماذج متعددة مدربة بشكل مستقل، وجداول زمنية، ومعالجات) في فئة واحدة شاملة. تتمتع خطوط الأنابيب بالمرونة ويمكن تكييفها لاستخدام جداول زمنية مختلفة أو حتى مكونات نموذجية.

جميع خطوط الأنابيب مبنية على فئة [`DiffusionPipeline`] الأساسية التي توفر الوظائف الأساسية لتحميل جميع المكونات وتنزيلها وحفظها. يتم الكشف تلقائيًا عن أنواع خطوط الأنابيب المحددة (على سبيل المثال [`StableDiffusionPipeline`]) المحملة بـ [`~DiffusionPipeline.from_pretrained`] ويتم تحميل مكونات خط الأنابيب ونقلها إلى وظيفة `__init__` لخط الأنابيب.

<Tip warning={true}>
لا يجب استخدام فئة [`DiffusionPipeline`] للتدريب. عادة ما يتم تدريب مكونات خطوط أنابيب الانتشار (على سبيل المثال، [`UNet2DModel`] و [`UNet2DConditionModel`]) بشكل فردي، لذلك نقترح العمل مباشرة معها بدلاً من ذلك.
<br>
لا توفر خطوط الأنابيب أي وظائف تدريب. ستلاحظ أن PyTorch autograd معطل من خلال تزيين طريقة [`~DiffusionPipeline.__call__`] باستخدام مزين [`torch.no_grad`] لأن خطوط الأنابيب لا يجب استخدامها للتدريب. إذا كنت مهتمًا بالتدريب، فراجع أدلة [التدريب] (https://github.com/huggingface/diffusers/tree/6b47c2e3a9a9a3f6b6e22490391280e73686440d/training/overview) بدلاً من ذلك!
</Tip>
يسرد الجدول أدناه جميع خطوط الأنابيب المتاحة حاليًا في 🤗 Diffusers والمهام التي تدعمها. انقر فوق خط أنابيب لعرض ملخصه وورقته المنشورة.

| Pipeline | المهام |
|---|---|
| [AltDiffusion] (alt_diffusion) | image2image |
| [AnimateDiff] (animatediff) | text2video |
| [Attend-and-Excite] (attend_and_excite) | text2image |
| [Audio Diffusion] (audio_diffusion) | image2audio |
| [AudioLDM] (audioldm) | text2audio |
| [AudioLDM2] (audioldm2) | text2audio |
| [BLIP Diffusion] (blip_diffusion) | text2image |
| [Consistency Models] (consistency_models) | unconditional image generation |
| [ControlNet] (controlnet) | text2image، image2image، inpainting |
| [ControlNet with Stable Diffusion XL] (controlnet_sdxl) | text2image |
| [ControlNet-XS] (controlnetxs) | text2image |
| [ControlNet-XS with Stable Diffusion XL] (controlnetxs_sdxl) | text2image |
| [Cycle Diffusion] (cycle_diffusion) | image2image |
| [Dance Diffusion] (dance_diffusion) | unconditional audio generation |
| [DDIM] (ddim) | unconditional image generation |
| [DDPM] (ddpm) | unconditional image generation |
| [DeepFloyd IF] (deepfloyd_if) | text2image، image2image، inpainting، super-resolution |
| [DiffEdit] (diffedit) | inpainting |
| [DiT] (dit) | text2image |
| [GLIGEN] (stable_diffusion/gligen) | text2image |
| [InstructPix2Pix] (pix2pix) | تحرير الصور |
| [Kandinsky 2.1] (kandinsky) | text2image، image2image، inpainting، interpolation |
| [Kandinsky 2.2] (kandinsky_v22) | text2image، image2image، inpainting |
| [Kandinsky 3] (kandinsky3) | text2image، image2image |
| [Latent Consistency Models] (latent_consistency_models) | text2image |
| [Latent Diffusion] (latent_diffusion) | text2image، super-resolution |
| [LDM3D] (stable_diffusion/ldm3d_diffusion) | text2image، text-to-3D، text-to-pano، upscaling |
| [LEDITS++] (ledits_pp) | تحرير الصور |
| [MultiDiffusion] (panorama) | text2image |
| [MusicLDM] (musicldm) | text2audio |
| [Paint by Example] (paint_by_example) | inpainting |
| [ParaDiGMS] (paradigms) | text2image |
| [Pix2Pix Zero] (pix2pix_zero) | تحرير الصور |
| [PixArt-α] (pixart) | text2image |
| [PNDM] (pndm) | unconditional image generation |
| [RePaint] (repaint) | inpainting |
| [Score SDE VE] (score_sde_ve) | unconditional image generation |
| [Self-Attention Guidance] (self_attention_guidance) | text2image |
| [Semantic Guidance] (semantic_stable_diffusion) | text2image |
| [Shap-E] (shap_e) | text-to-3D، image-to-3D |
| [Spectrogram Diffusion] (spectrogram_diffusion) | |
| [Stable Diffusion] (stable_diffusion/overview) | text2image، image2image، depth2image، inpainting، image variation، latent upscaler، super-resolution |
| [Stable Diffusion Model Editing] (model_editing) | تحرير النماذج |
| [Stable Diffusion XL] (stable_diffusion/stable_diffusion_xl) | text2image، image2image، inpainting |
| [Stable Diffusion XL Turbo] (stable_diffusion/sdxl_turbo) | text2image، image2image، inpainting |
| [Stable unCLIP] (stable_unclip) | text2image، image variation |
| [Stochastic Karras VE] (stochastic_karras_ve) | unconditional image generation |
| [T2I-Adapter] (stable_diffusion/adapter) | text2image |
| [Text2Video] (text_to_video) | text2video، video2video |
| [Text2Video-Zero] (text_to_video_zero) | text2video |
| [unCLIP] (unclip) | text2image، image variation |
| [Unconditional Latent Diffusion] (latent_diffusion_uncond) | unconditional image generation |
| [UniDiffuser] (unidiffuser) | text2image، image2text، image variation، text variation، unconditional image generation، unconditional audio generation |
| [Value-guided planning] (value_guided_sampling) | value guided sampling |
| [Versatile Diffusion] (versatile_diffusion) | text2image، image variation |
| [VQ Diffusion] (vq_diffusion) | text2image |
| [Wuerstchen] (wuerstchen) | text2image |
## DiffusionPipeline
[[autodoc]] DiffusionPipeline
- all
- __call__
- device 
- to
- components
[[autodoc]] pipelines.StableDiffusionMixin.enable_freeu
[[autodoc]] pipelines.StableDiffusionMixin.disable_freeu
## FlaxDiffusionPipeline
[[autodoc]] pipelines.pipeline_flax_utils.FlaxDiffusionPipeline
## PushToHubMixin
[[autodoc]] utils.PushToHubMixin