from ...utils import is_torch_available


if is_torch_available():
    from .latent_upscaler_kandinsky6_sr import Kandinsky6SRLatentUpscalerBank
