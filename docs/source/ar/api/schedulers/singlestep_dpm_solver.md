# DPMSolverSinglestepScheduler

`DPMSolverSinglestepScheduler` هو مخطط أحادي الخطوة من [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://huggingface.co/papers/2206.00927) و [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://huggingface.co/papers/2211.01095) بواسطة Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, و Jun Zhu.

DPMSolver (والنسخة المحسنة DPMSolver++) هو محول مخصص وعالي الكفاءة لمعادلات الانتشار التفاضلية مع ضمان ترتيب التقارب. ومن الناحية التجريبية، يمكن لنمذجة DPMSolver باستخدام 20 خطوة فقط أن تولد عينات عالية الجودة، ويمكنها توليد عينات جيدة جدًا حتى في 10 خطوات.

يمكن العثور على التنفيذ الأصلي في [LuChengTHU/dpm-solver](https://github.com/LuChengTHU/dpm-solver).

## نصائح

من المستحسن تعيين `solver_order` إلى 2 لعملية النمذجة الموجهة، و `solver_order=3` للنمذجة غير المشروطة.

يتم دعم العتبات الديناميكية من [Imagen](https://huggingface.co/papers/2205.11487)، وبالنسبة لنماذج الانتشار في مساحة البكسل، يمكنك تعيين كلا الخيارين `algorithm_type="dpmsolver++"` و `thresholding=True` لاستخدام العتبات الديناميكية. طريقة العتبة هذه غير مناسبة لنماذج الانتشار في الفراغ مثل Stable Diffusion.

## DPMSolverSinglestepScheduler

[[autodoc]] DPMSolverSinglestepScheduler

## SchedulerOutput

[[autodoc]] schedulers.scheduling_utils.SchedulerOutput