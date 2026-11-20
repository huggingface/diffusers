import torch

try:
    from .profiling_utils import PipelineProfilingConfig
except ImportError:
    from profiling_utils import PipelineProfilingConfig


PROMPT = "A cat holding a sign that says hello world"


def build_registry():
    """Build the pipeline config registry. Imports are deferred to avoid loading all pipelines upfront."""
    from diffusers import Flux2KleinPipeline, FluxPipeline, LTX2Pipeline, QwenImagePipeline, WanPipeline

    return {
        "flux": PipelineProfilingConfig(
            name="flux",
            pipeline_cls=FluxPipeline,
            pipeline_init_kwargs={
                "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
                "torch_dtype": torch.bfloat16,
            },
            pipeline_call_kwargs={
                "prompt": PROMPT,
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 4,
                "guidance_scale": 3.5,
                "output_type": "latent",
            },
        ),
        "flux2": PipelineProfilingConfig(
            name="flux2",
            pipeline_cls=Flux2KleinPipeline,
            pipeline_init_kwargs={
                "pretrained_model_name_or_path": "black-forest-labs/FLUX.2-klein-base-9B",
                "torch_dtype": torch.bfloat16,
            },
            pipeline_call_kwargs={
                "prompt": PROMPT,
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 4,
                "guidance_scale": 3.5,
                "output_type": "latent",
            },
        ),
        "wan": PipelineProfilingConfig(
            name="wan",
            pipeline_cls=WanPipeline,
            pipeline_init_kwargs={
                "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                "torch_dtype": torch.bfloat16,
            },
            pipeline_call_kwargs={
                "prompt": PROMPT,
                "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                "height": 480,
                "width": 832,
                "num_frames": 81,
                "num_inference_steps": 4,
                "output_type": "latent",
            },
        ),
        "ltx2": PipelineProfilingConfig(
            name="ltx2",
            pipeline_cls=LTX2Pipeline,
            pipeline_init_kwargs={
                "pretrained_model_name_or_path": "Lightricks/LTX-2",
                "torch_dtype": torch.bfloat16,
            },
            pipeline_call_kwargs={
                "prompt": PROMPT,
                "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
                "height": 512,
                "width": 768,
                "num_frames": 121,
                "num_inference_steps": 4,
                "guidance_scale": 4.0,
                "output_type": "latent",
            },
        ),
        "qwenimage": PipelineProfilingConfig(
            name="qwenimage",
            pipeline_cls=QwenImagePipeline,
            pipeline_init_kwargs={
                "pretrained_model_name_or_path": "Qwen/Qwen-Image",
                "torch_dtype": torch.bfloat16,
            },
            pipeline_call_kwargs={
                "prompt": PROMPT,
                "negative_prompt": " ",
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 4,
                "true_cfg_scale": 4.0,
                "output_type": "latent",
            },
        ),
    }
