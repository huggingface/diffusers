# RAE (Representation Autoencoder) Integration Plan

## Overview

This document outlines the plan to integrate RAE (Representation Autoencoder) into diffusers. RAE is a novel autoencoder architecture that uses frozen pretrained encoders (DINOv2, SigLIP2, MAE) with trainable ViT decoders for high-fidelity image reconstruction.

**Key Reference**: [RAE Paper](https://arxiv.org/abs/2510.11690) | [Official Repository](https://github.com/bytetriper/RAE)

## Architecture Summary

RAE consists of three main components:

1. **Frozen Encoder** (pretrained, not trainable)
   - DINOv2-with-Registers (facebook/dinov2-with-registers-base)
   - SigLIP2 (google/siglip2-base-patch16)
   - MAE (facebook/vit-mae-base)

2. **Trainable Decoder** (ViT MAE-style)
   - Based on ViTMAE architecture
   - Takes encoder outputs and reconstructs images
   - Configurable depth, hidden size, attention heads

3. **Optional Latent Normalization**
   - Statistics computed from encoder outputs
   - Batchnorm-like normalization for stable latent space

## Implementation Strategy

### Phase 1: Core Autoencoder Model

#### 1.1 Create `AutoencoderRAE` class

**Location**: `src/diffusers/models/autoencoders/autoencoder_rae.py`

**Key Design Decisions**:

1. **Follow diffusers conventions**:
   - Inherit from `ModelMixin`, `ConfigMixin`, `FromOriginalModelMixin`
   - Implement `encode()` and `decode()` methods
   - Return `DecoderOutput` for decode

2. **Support `from_pretrained`**:
   - Load full RAE model (encoder + decoder + stats) from HF Hub
   - Use `nyu-visionx/RAE-collections` as the model hub

3. **Encoder Types**:
   - Map RAE encoder names to HF model IDs
   - DINOv2, SigLIP2, MAE via transformers

4. **Decoder Configuration**:
   - Reuse/adapt ViTMAEDecoder from transformers or implement custom
   - Support configurable decoder sizes (ViTB, ViTL, ViTXL)

**Proposed API**:

```python
from diffusers import AutoencoderRAE

# Load from HF Hub
autoencoder = AutoencoderRAE.from_pretrained("nyu-visionx/RAE-collections")

# Encode images to latents
latents = autoencoder.encode(images)  # B, C, H, W

# Decode latents to images
reconstructed = autodecoder.decode(latents)  # B, 3, H, W

# Full forward pass
output = autoencoder(images)
```

#### 1.2 Configuration Class

**Location**: `src/diffusers/models/autoencoders/autoencoder_rae.py` (or separate config)

**Parameters**:
- `encoder_type`: str = "dinov2"  # or "siglip2", "mae"
- `encoder_name_or_path`: str = "facebook/dinov2-with-registers-base"
- `decoder_hidden_size`: int = 768
- `decoder_num_hidden_layers`: int = 12
- `decoder_num_attention_heads`: int = 12
- `decoder_intermediate_size`: int = 3072
- `decoder_patch_size`: int = 16
- `patch_size`: int = 16  # encoder patch size
- `encoder_input_size`: int = 224
- `image_size`: int = 256
- `num_channels`: int = 3
- `latent_mean`: Optional[torch.Tensor] = None
- `latent_var`: Optional[torch.Tensor] = None
- `noise_tau`: float = 0.0
- `reshape_to_2d`: bool = True

### Phase 2: Encoder Wrappers

**Location**: `src/diffusers/models/encoders/` (new directory)

Create lightweight wrappers for each encoder type:

1. **`encoder_dinov2.py`**: Wraps `Dinov2WithRegistersModel`
   - Strips CLS + register tokens (5 tokens)
   - Applies LayerNorm without affine parameters

2. **`encoder_siglip2.py`**: Wraps `SiglipModel`
   - Uses vision transformer output
   - Applies post-LayerNorm without affine

3. **`encoder_mae.py`**: Wraps `ViTMAEForPreTraining.vit`
   - No masking (mask_ratio=0)
   - Returns patch tokens only

### Phase 3: Decoder Implementation

**Location**: `src/diffusers/models/autoencoders/vae.py` or new file

Reuse existing patterns from `GeneralDecoder` in RAE repo:

- `ViTMAEEmbeddings`: Patch embedding + positional encoding
- `ViTMAELayer`: Transformer block (pre-norm)
- `GeneralDecoder`: Full decoder with configurable depth

Key features to implement:
- Sin-cos 2D positional embeddings
- Trainable CLS token
- Latent interpolation for variable sequence lengths
- Unpatchify for image reconstruction

### Phase 4: Integration Points

1. **`src/diffusers/models/autoencoders/__init__.py`**:
   ```python
   from .autoencoder_rae import AutoencoderRAE
   ```

2. **`src/diffusers/__init__.py`**:
   ```python
   from .models import AutoencoderRAE
   ```

3. **`src/diffusers/models/__init__.py`**:
   - Export `AutoencoderRAE`

### Phase 5: Optional Pipeline (Future)

**Location**: `src/diffusers/pipelines/rae/` (new directory)

Only if needed for complete integration:
- `RAEPipeline`: End-to-end encoding/decoding
- Integration with existing diffusion pipelines via latents

## Weight Loading Strategy

The pretrained weights in `nyu-visionx/RAE-collections` follow this structure:
- `models/decoders/<encoder>/<config>/model.pt` - decoder weights
- `models/stats/<encoder>/<dataset>/stat.pt` - normalization stats

**Loading Approach**:
1. Download full collection or specific models
2. The model is saved as a full `RAE` module
3. Can be loaded directly with `torch.load()` for conversion

## Code Reuse from RAE Repository

| Component | RAE Location | Diffusers Approach |
|-----------|--------------|-------------------|
| RAE Model | `src/stage1/rae.py` | Create new `AutoencoderRAE` class |
| Encoders | `src/stage1/encoders/*.py` | Create lightweight wrappers |
| Decoder | `src/stage1/decoders/decoder.py` | Adapt `GeneralDecoder` |
| Config | Custom `ViTMAEConfig` | Use or adapt transformers ViTMAEConfig |

## Testing Plan

1. **Unit Tests**:
   - `tests/models/test_autoencoder_rae.py`
   - Test `from_pretrained` loading
   - Test encode/decode roundtrip
   - Test with different encoder types

2. **Integration Tests**:
   - Compare outputs with original RAE implementation
   - Verify latent dimensions match expected shapes

3. **Slow Tests** (optional):
   - Full reconstruction quality test
   - FID comparison with pretrained weights

## Files to Create

```
src/diffusers/models/autoencoders/
└── autoencoder_rae.py          # Main AutoencoderRAE class

src/diffusers/models/encoders/   # New directory
├── __init__.py
├── encoder_dinov2.py           # DINOv2 wrapper
├── encoder_siglip2.py          # SigLIP2 wrapper
└── encoder_mae.py              # MAE wrapper

src/diffusers/pipelines/rae/     # Optional, future
├── __init__.py
└── rae_pipeline.py
```

## Files to Modify

1. `src/diffusers/models/autoencoders/__init__.py`
2. `src/diffusers/models/__init__.py`
3. `src/diffusers/__init__.py`
4. `tests/models/test_autoencoder_rae.py` (new file)

## Notes

- **Focus**: Primary goal is the autoencoder (encode/decode), not the full diffusion pipeline
- **Encoder Compatibility**: RAE already uses HF transformers for encoders, making integration straightforward
- **Latent Space**: The latent is 2D feature map (B, C, H, W), similar to other VAEs in diffusers
- **Scaling**: No scaling factor needed - latents are raw encoder outputs
- **Normalization Stats**: Optional latent normalization using pre-computed mean/var

## References

- [Original RAE Repository](https://github.com/bytetriper/RAE)
- [DINOv2 with Registers](https://huggingface.co/facebook/dinov2-with-registers-base)
- [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-256)
- [ViTMAE](https://huggingface.co/facebook/vit-mae-base)
- [RAE Collections on HF](https://huggingface.co/nyu-visionx/RAE-collections)
