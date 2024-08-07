# جدول المهام متعدد الخطوات للنموذج الاتساقي الكامن LCMScheduler

## نظرة عامة
تم تقديم جدول المهام متعدد الخطوات وذو الخطوة الواحدة (الخوارزمية 3) إلى جانب النماذج الاتساقية الكامنة في الورقة البحثية [Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378) بواسطة Simian Luo و Yiqin Tan و Longbo Huang و Jian Li و Hang Zhao.

يجب أن يكون هذا الجدول قادرًا على توليد عينات جيدة من [`LatentConsistencyModelPipeline`] في 1-8 خطوات.

## LCMScheduler

[[autodoc]] LCMScheduler