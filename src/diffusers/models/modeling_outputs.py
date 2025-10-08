from dataclasses import dataclass

from ..utils import BaseOutput, deprecate


@dataclass
class AutoencoderKLOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: "DiagonalGaussianDistribution"  # noqa: F821


class Transformer2DModelOutput:
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    def __new__(cls, *args, **kwargs):
        deprecate(
            "Transformer2DModelOutput",
            "1.0.0",
            "Importing `Transformer2DModelOutput` from `diffusers.models.modeling_outputs` is deprecated. Please use `from diffusers.models.transformers.modeling_common import Transformer2DModelOutput` instead.",
            standard_warn=False,
        )
        from .transformers.modeling_common import Transformer2DModelOutput

        return Transformer2DModelOutput(*args, **kwargs)
