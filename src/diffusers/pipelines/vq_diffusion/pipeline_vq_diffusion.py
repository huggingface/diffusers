from diffusers import VQDiffusionTransformer, VQModel
from transformers import CLIPTextModel, CLIPTokenizer

from ...pipeline_utils import DiffusionPipeline


# This class is a placeholder and does not have the full VQ-diffusion pipeline built out yet
#
# NOTE: In VQ-Diffusion, the VQVAE trained on the ITHQ dataset uses an EMA variant of the vector quantizer
# in diffusers. The EMA variant uses EMA's to update the codebook during training but acts the same as the
# usual vector quantizer during inference. The VQDiffusion pipeline uses the non-ema vector quantizer during
# inference. If diffusers is to support training, the EMA vector quantizer could be implemented. For more
# information on EMA Vector quantizers, see https://arxiv.org/abs/1711.00937.
class VQDiffusionPipeline(DiffusionPipeline):
    vqvae: VQModel
    transformer: VQDiffusionTransformer

    def __init__(
        self,
        vqvae: VQModel,
        transformer: VQDiffusionTransformer,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        super().__init__()
        self.register_modules(
            vqvae=vqvae,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
