# EDMDPMSolverMultistepScheduler

`EDMDPMSolverMultistepScheduler` هو صيغة [Karras](https://huggingface.co/papers/2206.00364) من `DPMSolverMultistepScheduler`، وهو جدول زمني متعدد الخطوات من [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://huggingface.co/papers/2206.00927) و [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://huggingface.co/papers/2211.01095) بقلم Cheng Lu، Yuhao Zhou، Fan Bao، Jianfei Chen، Chongxuan Li، و Jun Zhu.

DPMSolver (والنسخة المحسنة DPMSolver++) هو مُحَلِّل مخصص وعالي الكفاءة لمعادلات الانتشار التفاضلية مع ضمان ترتيب التقارب. ومن الناحية التجريبية، يمكن لعملية أخذ العينات DPMSolver باستخدام 20 خطوة فقط أن تولد عينات عالية الجودة، ويمكنها توليد عينات جيدة جدًا حتى في 10 خطوات.

## EDMDPMSolverMultistepScheduler

[[autodoc]] EDMDPMSolverMultistepScheduler

## SchedulerOutput

[[autodoc]] schedulers.scheduling_utils.SchedulerOutput