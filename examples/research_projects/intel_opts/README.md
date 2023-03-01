## Diffusers examples with Intel optimizations

**This research project is not actively maintained by the diffusers team. For any questions or comments, please make sure to tag @hshen14 .**

This aims to provide diffusers examples with Intel optimizations such as Bfloat16 for training/fine-tuning acceleration and 8-bit integer (INT8) for inference acceleration on Intel platforms.

## Accelerating the fine-tuning for textual inversion

We accelereate the fine-tuning for textual inversion with Intel Extension for PyTorch. The [examples](textual_inversion) enable both single node and multi-node distributed training with Bfloat16 support on Intel Xeon Scalable Processor.

## Accelerating the inference for Stable Diffusion using Bfloat16

We start the inference acceleration with Bfloat16 using Intel Extension for PyTorch. The [script](inference_bf16.py) is generally designed to support standard Stable Diffusion models with Bfloat16 support.

## Accelerating the inference for Stable Diffusion using INT8

Coming soon ...
