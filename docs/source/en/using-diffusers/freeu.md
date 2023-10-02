# FreeU to improve generation quality

[[open-in-colab]]

Usually a UNet is responsible for denoising during the reverse diffusion process. The features inside the UNet can be boradly classified distinctively: 

1. Backbone features
2. Skip features

In [FreeU: Free Lunch in Diffusion U-Net](https://arxiv.org/abs/2309.11497), Si et al. investigate the contributions of these features in the context of diffusion. They found out that backbone features primarily contribute to the denoising process while the skip features mainly introduce high-frequency features into the decoder module. Furthermore, the skip features can make the network overlook the semantics baked in the backbone features. 

To mitigate these issues, the authors introduce the **FreeU mechanism** where they simply reweigh the contributions sourced from the UNet’s skip connections and backbone feature maps, to leverage the strengths of both components. 

FreeU is an inference-time mechanism meaning that it does not require any additional training. It is completely technique that works with different tasks such as text-to-image, text-to-video, and image-to-image.

In this guide, we will discuss how to apply FreeU for different pipelines like [`StableDiffusionPipeline`], [`StableDiffusionXLPipeline`], and [`TextToVideoSDPipeline`].