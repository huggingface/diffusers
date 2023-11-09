# Consistency Decoder

Consistency decoder can be used to decode the latents from the denoising UNet in the [`StableDiffusionPipeline`]. This decoder was introduced in the [DALL-E 3 technical report](https://openai.com/dall-e-3). 

The original codebase can be found at [openai/consistencydecoder](https://github.com/openai/consistencydecoder).

<Tip warning={true}>

Inference is only supported for 2 iterations as of now.

</Tip>


## ConsistencyDecoderVae
[[autodoc]] ConsistencyDecoderVae
    - all
    - decode