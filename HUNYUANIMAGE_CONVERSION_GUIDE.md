# HunyuanImage-2.1 to Diffusers Conversion Guide

This document outlines the conversion of the HunyuanImage-2.1 model from the official Tencent repository to the Diffusers library style.

## Overview

HunyuanImage-2.1 is a 17B parameter text-to-image diffusion model capable of generating 2K (2048×2048) resolution images. The model has several unique features:

- **Dual-stream Architecture**: Uses both double-stream and single-stream transformer blocks (similar to FLUX)
- **Custom 32x VAE**: A specialized VAE with 32x spatial compression instead of the typical 8x
- **ByT5 Text Encoder**: Uses ByT5 for glyph-aware text rendering capabilities
- **MeanFlow Distillation**: Supports distilled models for faster inference
- **Guidance Embedding**: Has optional guidance embedding for CFG distillation

## Repository Structure

The official repository is located at: https://github.com/Tencent-Hunyuan/HunyuanImage-2.1

Key directories:
- `hyimage/models/hunyuan/modules/` - Core transformer modules
- `hyimage/models/vae/` - VAE implementation  
- `hyimage/diffusion/pipelines/` - Inference pipelines
- `hyimage/models/text_encoder/` - Text encoder wrappers

## What Has Been Completed

### 1. VAE Model ✅

**File**: `/workspace/src/diffusers/models/autoencoders/autoencoder_kl_hunyuanimage.py`

A complete implementation of the HunyuanImage VAE with:
- 32x spatial compression
- Custom encoder/decoder architecture
- Diagonal Gaussian distribution
- Support for gradient checkpointing
- Optional slicing for memory efficiency

**Key Features**:
- `in_channels`: 3 (RGB images)
- `latent_channels`: 64 (unlike SD's 4)
- `block_out_channels`: (512, 1024, 2048, 4096)
- `ffactor_spatial`: 32 (spatial downsampling factor)

**Usage**:
```python
from diffusers.models import AutoencoderKLHunyuanImage

vae = AutoencoderKLHunyuanImage(
    in_channels=3,
    out_channels=3,
    latent_channels=64,
    block_out_channels=(512, 1024, 2048, 4096),
    layers_per_block=2,
    ffactor_spatial=32,
)
```

## What Still Needs to Be Done

### 2. Transformer Model ⚠️ (Priority: HIGH)

**Target File**: `/workspace/src/diffusers/models/transformers/transformer_hunyuanimage_2d.py`

The main transformer model needs to be converted from:
- Source: `/tmp/hunyuanimage-2.1/hyimage/models/hunyuan/modules/hunyuanimage_dit.py`

**Key Components to Implement**:

#### a) `MMDoubleStreamBlock` 
- Dual-stream attention mechanism
- Separate processing for image and text tokens
- RoPE (Rotary Position Embeddings)
- QK normalization
- Modulation layers

#### b) `MMSingleStreamBlock`
- Single-stream processing after double blocks
- Parallel linear layers for Q, K, V, and MLP
- Concatenated image + text tokens

#### c) `HYImageDiffusionTransformer`
Main model class with:
- Patch embedding (2D or 3D)
- Text projection or token refiner
- Time embedding
- Optional guidance embedding (for distilled models)
- Optional MeanFlow support (timesteps_r parameter)
- Double blocks (20 layers)
- Single blocks (40 layers)
- Final layer with modulation

**Configuration**:
```python
# v2.1 non-distilled
in_channels=64
out_channels=64
mm_double_blocks_depth=20
mm_single_blocks_depth=40
rope_dim_list=[64, 64]
hidden_size=3584
heads_num=28
mlp_width_ratio=4
patch_size=[1, 1]
text_states_dim=3584
glyph_byT5_v2=True
guidance_embed=False

# v2.1 distilled
guidance_embed=True
use_meanflow=True
```

### 3. Pipeline ⚠️ (Priority: HIGH)

**Target File**: `/workspace/src/diffusers/pipelines/hunyuanimage/pipeline_hunyuanimage.py`

Convert from: `/tmp/hunyuanimage-2.1/hyimage/diffusion/pipelines/hunyuanimage_pipeline.py`

**Key Components**:

#### a) Text Encoding
- Support for multi-modal LLM text encoder
- Integration with ByT5 for glyph rendering
- Text mask handling

#### b) ByT5 Integration  
- Character-level encoding for text rendering
- Glyph extraction from prompts (quoted text)
- Token reordering mechanism

#### c) Sampling
- Custom timestep scheduling with shift parameter
- Euler sampler (simple first-order)
- Optional MeanFlow (two timesteps per step)
- CFG with guidance scale
- Optional APG (Adaptive Projected Guidance)

#### d) Configuration Management
- Model offloading strategies
- FP8 quantization support
- Memory optimization

**Pipeline Interface**:
```python
from diffusers import HunyuanImagePipeline

pipe = HunyuanImagePipeline.from_pretrained(
    "tencent/HunyuanImage-2.1",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

image = pipe(
    prompt="A cute penguin",
    width=2048,
    height=2048,
    num_inference_steps=50,
    guidance_scale=3.5,
).images[0]
```

### 4. Conversion Script ⚠️ (Priority: MEDIUM)

**Target File**: `/workspace/scripts/convert_hunyuanimage_to_diffusers.py`

Similar to existing conversion scripts, this should:
1. Load official checkpoint weights
2. Map weight keys from official format to diffusers format
3. Handle different configurations (base, distilled, refiner)
4. Save in diffusers format

**Key Weight Mappings**:

```python
# Transformer blocks
"double_blocks.{i}.attn_q" -> "double_blocks.{i}.img_attn_q"
"double_blocks.{i}.attn_k" -> "double_blocks.{i}.img_attn_k"
"double_blocks.{i}.attn_v" -> "double_blocks.{i}.img_attn_v"

# Single blocks
"single_blocks.{i}.linear1_q" -> "single_blocks.{i}.linear1_q"
"single_blocks.{i}.linear1_k" -> "single_blocks.{i}.linear1_k"  
"single_blocks.{i}.linear1_v" -> "single_blocks.{i}.linear1_v"
"single_blocks.{i}.linear1_mlp" -> "single_blocks.{i}.linear1_mlp"
"single_blocks.{i}.linear2.fc" -> "single_blocks.{i}.linear2.fc"

# Embeddings
"img_in" -> "pos_embed"
"txt_in" -> "text_embedder" or "txt_in"
"time_in" -> "time_embedder"
"time_r_in" -> "time_r_embedder" (for distilled models)
"guidance_in" -> "guidance_embedder" (for distilled models)
```

### 5. Tests ⚠️ (Priority: MEDIUM)

**Target Files**: 
- `/workspace/tests/models/transformers/test_models_transformer_hunyuanimage.py`
- `/workspace/tests/models/autoencoders/test_models_autoencoder_hunyuanimage.py`
- `/workspace/tests/pipelines/hunyuanimage/test_hunyuanimage.py`

Tests should cover:
- Model loading and saving
- Forward pass shapes
- Gradient checkpointing
- Different configurations
- Pipeline end-to-end generation

### 6. Documentation ⚠️ (Priority: LOW)

**Target Files**:
- `/workspace/docs/source/en/api/pipelines/hunyuanimage.md`
- `/workspace/docs/source/en/api/models/hunyuanimage_transformer2d.md`

Documentation should include:
- Model overview and features
- Usage examples
- Parameter explanations
- Known limitations

## Key Technical Challenges

### 1. Flash Attention Implementation
HunyuanImage uses custom flash attention (`flash_attn_no_pad`):
- Source: `/tmp/hunyuanimage-2.1/hyimage/models/hunyuan/modules/flash_attn_no_pad.py`
- Handles variable-length sequences with masks
- Need to adapt or use diffusers' flash attention

### 2. ByT5 Integration
The glyph-aware text encoding requires:
- ByT5 tokenizer and model
- Custom prompt parsing (extracting quoted text)
- Token reordering logic
- May need separate component or helper class

### 3. Text Encoder Handling
Multiple text encoder configurations:
- Linear projection (simpler)
- Single token refiner (more complex, default for v2.1)
- Need flexible interface

### 4. RoPE (Rotary Position Embeddings)
Custom n-dimensional RoPE:
- Source: `/tmp/hunyuanimage-2.1/hyimage/models/hunyuan/modules/posemb_layers.py`
- Supports 2D and 3D position encoding
- Different dimensions for different axes

### 5. Model Variants
Need to support multiple variants:
- HunyuanImage-2.1 (base, 50 steps)
- HunyuanImage-2.1-distilled (8 steps)
- HunyuanImage-refiner (optional refinement stage)

## Recommended Implementation Order

1. **Phase 1**: Core Transformer Model
   - Start with simplified version (no ByT5, basic text projection)
   - Implement double and single stream blocks
   - Test with dummy inputs

2. **Phase 2**: VAE Integration ✅ (DONE)
   - Already completed
   - Test encoding/decoding

3. **Phase 3**: Basic Pipeline
   - Simple pipeline without ByT5
   - Use basic text encoder (e.g., T5)
   - Get end-to-end generation working

4. **Phase 4**: Advanced Features
   - Add ByT5 support
   - Add token refiner
   - Add distilled model support

5. **Phase 5**: Polish
   - Add tests
   - Add documentation
   - Optimize performance

## Code References

### Original Files to Convert

1. **Transformer Model**:
   - `/tmp/hunyuanimage-2.1/hyimage/models/hunyuan/modules/hunyuanimage_dit.py`
   - `/tmp/hunyuanimage-2.1/hyimage/models/hunyuan/modules/models.py`
   - `/tmp/hunyuanimage-2.1/hyimage/models/hunyuan/modules/embed_layers.py`
   - `/tmp/hunyuanimage-2.1/hyimage/models/hunyuan/modules/mlp_layers.py`
   - `/tmp/hunyuanimage-2.1/hyimage/models/hunyuan/modules/norm_layers.py`
   - `/tmp/hunyuanimage-2.1/hyimage/models/hunyuan/modules/modulate_layers.py`
   - `/tmp/hunyuanimage-2.1/hyimage/models/hunyuan/modules/posemb_layers.py`
   - `/tmp/hunyuanimage-2.1/hyimage/models/hunyuan/modules/token_refiner.py`
   - `/tmp/hunyuanimage-2.1/hyimage/models/hunyuan/modules/activation_layers.py`

2. **Pipeline**:
   - `/tmp/hunyuanimage-2.1/hyimage/diffusion/pipelines/hunyuanimage_pipeline.py`

3. **VAE** ✅:
   - `/tmp/hunyuanimage-2.1/hyimage/models/vae/hunyuanimage_vae.py` (DONE)

### Existing Diffusers Files for Reference

1. **Similar Transformer Models**:
   - `/workspace/src/diffusers/models/transformers/hunyuan_transformer_2d.py` (HunyuanDiT)
   - `/workspace/src/diffusers/models/transformers/transformer_flux.py` (FLUX, similar dual-stream)
   - `/workspace/src/diffusers/models/transformers/transformer_sd3.py` (SD3, MMDiT blocks)

2. **Similar Pipelines**:
   - `/workspace/src/diffusers/pipelines/hunyuandit/pipeline_hunyuandit.py`
   - `/workspace/src/diffusers/pipelines/flux/pipeline_flux.py`

3. **Conversion Scripts**:
   - `/workspace/scripts/convert_hunyuandit_to_diffusers.py`
   - `/workspace/scripts/convert_flux_to_diffusers.py`

## Model Weights

The official weights are available on HuggingFace:
- https://huggingface.co/tencent/HunyuanImage-2.1

Models available:
- `hunyuanimage-v2.1` - Base model (50 steps)
- `hunyuanimage-v2.1-distilled` - Distilled model (8 steps)  
- `hunyuanimage-refiner` - Optional refiner model
- FP8 quantized versions

## Additional Resources

- **Official Repository**: https://github.com/Tencent-Hunyuan/HunyuanImage-2.1
- **Model Card**: https://huggingface.co/tencent/HunyuanImage-2.1
- **Paper** (if available): Check the repository README

## Next Steps

The priority tasks are:

1. **Implement Transformer Model** - This is the core of the model
2. **Create Basic Pipeline** - Get end-to-end generation working
3. **Conversion Script** - Enable loading official weights
4. **Tests** - Ensure correctness
5. **Documentation** - Help users understand the model

## Notes

- The VAE has been completed and is ready to use
- The transformer model is the most complex part and will require the most work
- Consider starting with a simplified version that works with existing text encoders
- ByT5 integration can be added later as an enhancement
- The official repository has working code that can be tested for reference

## Contact

For questions or issues with this conversion, please refer to:
- Diffusers repository: https://github.com/huggingface/diffusers
- HunyuanImage repository: https://github.com/Tencent-Hunyuan/HunyuanImage-2.1
