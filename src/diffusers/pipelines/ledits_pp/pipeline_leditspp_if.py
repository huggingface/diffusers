import re
from typing import Optional

from transformers import CLIPImageProcessor, T5EncoderModel, T5Tokenizer

from ...loaders import LoraLoaderMixin
from ...models import UNet2DConditionModel
from ...schedulers import DDPMScheduler
from ...utils import (
    is_bs4_available,
    is_ftfy_available,
    logging,
)
from ..deepfloyd_if.safety_checker import IFSafetyChecker
from ..deepfloyd_if.watermark import IFWatermarker
from ..pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_bs4_available():
    pass

if is_ftfy_available():
    pass


class LEditsPPPipelineIF(DiffusionPipeline, LoraLoaderMixin):
    """
    Pipeline for textual image editing using LEDits++ with DeepfloydIF.

    This model inherits from [`DiffusionPipeline`] and builds on the [`IFPipeline`]. Check the superclass
    documentation for the generic methods implemented for all pipelines (downloading, saving, running on a particular
    device, etc.).
    """

    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel

    unet: UNet2DConditionModel
    scheduler: DDPMScheduler

    feature_extractor: Optional[CLIPImageProcessor]
    safety_checker: Optional[IFSafetyChecker]

    watermarker: Optional[IFWatermarker]

    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + "\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa

    _optional_components = ["tokenizer", "text_encoder", "safety_checker", "feature_extractor", "watermarker"]
    model_cpu_offload_seq = "text_encoder->unet"

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        safety_checker: Optional[IFSafetyChecker],
        feature_extractor: Optional[CLIPImageProcessor],
        watermarker: Optional[IFWatermarker],
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the IF license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            watermarker=watermarker,
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def __call__(self):
        """
        Function invoked when calling the pipeline for generation.
        """
        pass

    def invert(self):
        """
        Inverts a real image according to Algorihm 1 in https://arxiv.org/pdf/2304.06140.pdf,
        based on the code in https://github.com/inbarhub/DDPM_inversion
        """
        pass
