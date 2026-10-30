from ...utils import is_torch_available


if is_torch_available():
    from .latent_upscaler_kandinsky6_sr import Kandinsky6SRLatentUpscalerBank
    from .latent_upscaler_ltx import LTXLatentUpsamplerModel
    from .latent_upscaler_ltx2 import LTX2LatentUpsamplerModel
