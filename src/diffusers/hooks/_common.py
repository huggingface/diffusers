from ..models.attention_processor import Attention, MochiAttention


_ATTENTION_CLASSES = (Attention, MochiAttention)

_SPATIAL_ATTENTION_BLOCK_IDENTIFIERS = ("blocks", "transformer_blocks", "single_transformer_blocks")
_TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS = ("temporal_transformer_blocks",)
_CROSS_ATTENTION_BLOCK_IDENTIFIERS = ("blocks", "transformer_blocks")
