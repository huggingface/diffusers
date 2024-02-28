from ..utils import deprecate
from .transformers.prior_transformer import PriorTransformer, PriorTransformerOutput


class PriorTransformerOutput(PriorTransformerOutput):
    deprecation_message = "Importing `PriorTransformerOutput` from `diffusers.models.prior_transformer` is deprecated and this will be removed in a future version. Please use `from diffusers.models.transformers.prior_transformer import PriorTransformerOutput`, instead."
    deprecate("PriorTransformerOutput", "0.29", deprecation_message)


class PriorTransformer(PriorTransformer):
    deprecation_message = "Importing `PriorTransformer` from `diffusers.models.prior_transformer` is deprecated and this will be removed in a future version. Please use `from diffusers.models.transformers.prior_transformer import PriorTransformer`, instead."
    deprecate("PriorTransformer", "0.29", deprecation_message)
