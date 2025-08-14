# Guiders

Guiders are components in Modular Diffusers that control how the diffusion process is guided during generation. They implement various guidance techniques to improve generation quality and control.

## BaseGuidance

[[autodoc]] diffusers.guiders.guider_utils.BaseGuidance

## ClassifierFreeGuidance

[[autodoc]] diffusers.guiders.classifier_free_guidance.ClassifierFreeGuidance

## ClassifierFreeZeroStarGuidance

[[autodoc]] diffusers.guiders.classifier_free_zero_star_guidance.ClassifierFreeZeroStarGuidance

## SkipLayerGuidance

[[autodoc]] diffusers.guiders.skip_layer_guidance.SkipLayerGuidance

## SmoothedEnergyGuidance

[[autodoc]] diffusers.guiders.smoothed_energy_guidance.SmoothedEnergyGuidance

## PerturbedAttentionGuidance

[[autodoc]] diffusers.guiders.perturbed_attention_guidance.PerturbedAttentionGuidance

## AdaptiveProjectedGuidance

[[autodoc]] diffusers.guiders.adaptive_projected_guidance.AdaptiveProjectedGuidance

## AutoGuidance

[[autodoc]] diffusers.guiders.auto_guidance.AutoGuidance

## TangentialClassifierFreeGuidance

[[autodoc]] diffusers.guiders.tangential_classifier_free_guidance.TangentialClassifierFreeGuidance
