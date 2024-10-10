# `DPMSolverMultistepInverse`
`DPMSolverMultistepInverse` هو المُجدول العكسي من [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://huggingface.co/papers/2206.00927) و [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://huggingface.co/papers/2211.01095) بواسطة Cheng Lu، Yuhao Zhou، Fan Bao، Jianfei Chen، Chongxuan Li، و Jun Zhu.

يستند التنفيذ بشكل أساسي إلى تعريف DDIM inversion من [Null-text Inversion for Editing Real Images using Guided Diffusion Models](https://huggingface.co/papers/2211.09794) وتنفيذ دفتر الملاحظات لـ [`DiffEdit`] latent inversion من [Xiang-cd/DiffEdit-stable-diffusion](https://github.com/Xiang-cd/DiffEdit-stable-diffusion/blob/main/diffedit.ipynb).

## نصائح
يتم دعم العتبات الديناميكية من [Imagen](https://huggingface.co/papers/2205.11487)، وبالنسبة لنماذج الانتشار على مستوى البكسل، يمكنك تعيين كل من `algorithm_type="dpmsolver++"` و `thresholding=True` لاستخدام العتبة الديناميكية. طريقة العتبة هذه غير مناسبة لنماذج الانتشار في الفراغ مثل Stable Diffusion.

## `DPMSolverMultistepInverseScheduler`
[[autodoc]] DPMSolverMultistepInverseScheduler

## SchedulerOutput
[[autodoc]] schedulers.scheduling_utils.SchedulerOutput