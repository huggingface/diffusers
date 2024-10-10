# DPMSolverMultistepScheduler

`DPMSolverMultistepScheduler` هو جدول زمني متعدد الخطوات من [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://huggingface.co/papers/2206.00927) و [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://huggingface.co/papers/2211.01095) بقلم تشينغ لو، ويوهاو تشو، وفان باو، وجيانفي تشين، وتشونغكسوان لي، وجون تشو.

DPMSolver (والنسخة المحسنة DPMSolver++) هو محول مخصص وعالي الكفاءة لمعادلات ODE الانتشارية مع ضمان ترتيب التقارب. ومن الناحية التجريبية، يمكن لبرنامج DPMSolver sampling الذي يستخدم 20 خطوة فقط أن ينتج عينات عالية الجودة، ويمكنه إنتاج عينات جيدة جدًا حتى في 10 خطوات.

## نصائح

من المستحسن تعيين `solver_order` إلى 2 لعملية التوجيه sampling، و`solver_order=3` لعملية sampling غير المشروطة.

يتم دعم العتبات الديناميكية من [Imagen](https://huggingface.co/papers/2205.11487)، وبالنسبة لنماذج الانتشار في مساحة البكسل، يمكنك تعيين كلاً من `algorithm_type="dpmsolver++"` و`thresholding=True` لاستخدام العتبات الديناميكية. لا تناسب طريقة العتبة هذه نماذج الانتشار في مساحة المخفية مثل Stable Diffusion.

يتم أيضًا دعم متغير SDE من DPMSolver وDPM-Solver++، ولكن فقط لمحللات الطلب الأولى والثانية. هذه هي محول SDE سريع لمعادلة SDE الانتشار العكسي. يوصى باستخدام محول SDE من الدرجة الثانية "sde-dpmsolver++".

## DPMSolverMultistepScheduler

[[autodoc]] DPMSolverMultistepScheduler

## SchedulerOutput

[[autodoc]] schedulers.scheduling_utils.SchedulerOutput