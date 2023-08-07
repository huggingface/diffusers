# Würstchen

[Wuerstchen: Efficient Pretraining of Text-to-Image Models](https://huggingface.co/papers/2306.00637) is by Pablo Pernias, Dominic Rampas, and Marc Aubreville.

The abstract from the paper is:

*We introduce Wuerstchen, a novel technique for text-to-image synthesis that unites competitive performance with unprecedented cost-effectiveness and ease of training on constrained hardware. Building on recent advancements in machine learning, our approach, which utilizes latent diffusion strategies at strong latent image compression rates, significantly reduces the computational burden, typically associated with state-of-the-art models, while preserving, if not enhancing, the quality of generated images. Wuerstchen achieves notable speed improvements at inference time, thereby rendering real-time applications more viable. One of the key advantages of our method lies in its modest training requirements of only 9,200 GPU hours, slashing the usual costs significantly without compromising the end performance. In a comparison against the state-of-the-art, we found the approach to yield strong competitiveness. This paper opens the door to a new line of research that prioritizes both performance and computational accessibility, hence democratizing the use of sophisticated AI technologies. Through Wuerstchen, we demonstrate a compelling stride forward in the realm of text-to-image synthesis, offering an innovative path to explore in future research.*

The original codebase can be found at [dome272/Wuerstchen](https://github.com/dome272/Wuerstchen).

## VQDiffusionPipeline
[[autodoc]] VQDiffusionPipeline
	- all
	- __call__

## ImagePipelineOutput
[[autodoc]] pipelines.ImagePipelineOutput

## WuerstchenGeneratorPipeline
[[autodoc]] WuerstchenGeneratorPipeline
	- all
	- __call__

## WuerstchenGeneratorPipelineOutput
[[autodoc]] pipelines.WuerstchenGeneratorPipelineOutput
