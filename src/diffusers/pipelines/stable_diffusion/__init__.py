# flake8: noqa
from dataclasses import dataclass

from ...utils import ModelOutput, is_transformers_available


@dataclass
class StableDiffusionOutput(ModelOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`bool`)
            Flag stating whether generating images likely represent "not-safe-for-work" (nsfw) content.
    """

    images: Union[List[PIL.Images], np.ndarray] = None
    nsfw_content_detected: bool = None
    sample: Union[List[PIL.Images], np.ndarray] = None


if is_transformers_available():
    from .pipeline_stable_diffusion import StableDiffusionPipeline
    from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
    from .pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
    from .safety_checker import StableDiffusionSafetyChecker
