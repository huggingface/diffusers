# Latent Perceptual Loss (LPL) for Stable Diffusion XL

This directory contains an implementation of Latent Perceptual Loss (LPL) for training Stable Diffusion XL models, based on the paper: [Boosting Latent Diffusion with Perceptual Objectives](https://huggingface.co/papers/2411.04873) (Berrada et al., 2025). LPL is a perceptual loss that operates in the latent space of a VAE, helping to improve the quality and consistency of generated images by bridging the disconnect between the diffusion model and the autoencoder decoder. The implementation is based on the reference implementation provided by Tariq Berrada.

## Overview

LPL addresses a key limitation in latent diffusion models (LDMs): the disconnect between the diffusion model training and the autoencoder decoder. While LDMs train in the latent space, they don't receive direct feedback about how well their outputs decode into high-quality images. This can lead to:

- Loss of fine details in generated images
- Inconsistent image quality
- Structural artifacts
- Reduced sharpness and realism

LPL works by comparing intermediate features from the VAE decoder between the predicted and target latents. This helps the model learn better perceptual features and can lead to:

- Improved image quality and consistency (6-20% FID improvement)
- Better preservation of fine details
- More stable training, especially at high noise levels
- Better handling of structural information
- Sharper and more realistic textures

## Implementation Details

The LPL implementation follows the paper's methodology and includes several key features:

1. **Feature Extraction**: Extracts intermediate features from the VAE decoder, including:
   - Middle block features
   - Up block features (configurable number of blocks)
   - Proper gradient checkpointing for memory efficiency
   - Features are extracted only for timesteps below the threshold (high SNR)

2. **Feature Normalization**: Multiple normalization options as validated in the paper:
   - `default`: Normalize each feature map independently
   - `shared`: Cross-normalize features using target statistics (recommended)
   - `batch`: Batch-wise normalization

3. **Outlier Handling**: Optional removal of outliers in feature maps using:
   - Quantile-based filtering (2% quantiles)
   - Morphological operations (opening/closing)
   - Adaptive thresholding based on standard deviation

4. **Loss Types**:
   - MSE loss (default)
   - L1 loss
   - Optional power law weighting (2^(-i) for layer i)

## Usage

To use LPL in your training, add the following arguments to your training command:

```bash
python examples/research_projects/lpl/train_sdxl_lpl.py \
    --use_lpl \
    --lpl_weight 1.0 \                    # Weight for LPL loss (1.0-2.0 recommended)
    --lpl_t_threshold 200 \              # Apply LPL only for timesteps < threshold (high SNR)
    --lpl_loss_type mse \                # Loss type: "mse" or "l1"
    --lpl_norm_type shared \             # Normalization type: "default", "shared" (recommended), or "batch"
    --lpl_pow_law \                      # Use power law weighting for layers
    --lpl_num_blocks 4 \                 # Number of up blocks to use (1-4)
    --lpl_remove_outliers \              # Remove outliers in feature maps
    --lpl_scale \                        # Scale LPL loss by noise level weights
    --lpl_start 0 \                      # Step to start applying LPL
    # ... other training arguments ...
```

### Key Parameters

- `lpl_weight`: Controls the strength of the LPL loss relative to the main diffusion loss. Higher values (1.0-2.0) improve quality but may slow training.
- `lpl_t_threshold`: LPL is only applied for timesteps below this threshold (high SNR). Lower values (100-200) focus on more important timesteps.
- `lpl_loss_type`: Choose between MSE (default) and L1 loss. MSE is recommended for most cases.
- `lpl_norm_type`: Feature normalization strategy. "shared" is recommended as it showed best results in the paper.
- `lpl_pow_law`: Whether to use power law weighting (2^(-i) for layer i). Recommended for better feature balance.
- `lpl_num_blocks`: Number of up blocks to use for feature extraction (1-4). More blocks capture more features but use more memory.
- `lpl_remove_outliers`: Whether to remove outliers in feature maps. Recommended for stable training.
- `lpl_scale`: Whether to scale LPL loss by noise level weights. Helps focus on more important timesteps.
- `lpl_start`: Training step to start applying LPL. Can be used to warm up training.

## Recommendations

1. **Starting Point** (based on paper results):
   ```bash
   --use_lpl \
   --lpl_weight 1.0 \
   --lpl_t_threshold 200 \
   --lpl_loss_type mse \
   --lpl_norm_type shared \
   --lpl_pow_law \
   --lpl_num_blocks 4 \
   --lpl_remove_outliers \
   --lpl_scale
   ```

2. **Memory Efficiency**:
   - Use `--gradient_checkpointing` for memory efficiency (enabled by default)
   - Reduce `lpl_num_blocks` if memory is constrained (2-3 blocks still give good results)
   - Consider using `--lpl_scale` to focus on more important timesteps
   - Features are extracted only for timesteps below threshold to save memory

3. **Quality vs Speed**:
   - Higher `lpl_weight` (1.0-2.0) for better quality
   - Lower `lpl_t_threshold` (100-200) for faster training
   - Use `lpl_remove_outliers` for more stable training
   - `lpl_norm_type shared` provides best quality/speed trade-off

## Technical Details

### Feature Extraction

The LPL implementation extracts features from the VAE decoder in the following order:
1. Middle block output
2. Up block outputs (configurable number of blocks)

Each feature map is processed with:
1. Optional outlier removal (2% quantiles, morphological operations)
2. Feature normalization (shared statistics recommended)
3. Loss calculation (MSE or L1)
4. Optional power law weighting (2^(-i) for layer i)

### Loss Calculation

For each feature map:
1. Features are normalized according to the chosen strategy
2. Loss is calculated between normalized features
3. Outliers are masked out (if enabled)
4. Loss is weighted by layer depth (if power law enabled)
5. Final loss is averaged across all layers

### Memory Considerations

- Gradient checkpointing is used by default
- Features are extracted only for timesteps below the threshold
- Outlier removal is done in-place to save memory
- Feature normalization is done efficiently using vectorized operations
- Memory usage scales linearly with number of blocks used

## Results

Based on the paper's findings, LPL provides:
- 6-20% improvement in FID scores
- Better preservation of fine details
- More realistic textures and structures
- Improved consistency across different resolutions
- Better performance on both small and large datasets

## Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{berrada2025boosting,
    title={Boosting Latent Diffusion with Perceptual Objectives},
    author={Tariq Berrada and Pietro Astolfi and Melissa Hall and Marton Havasi and Yohann Benchetrit and Adriana Romero-Soriano and Karteek Alahari and Michal Drozdzal and Jakob Verbeek},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=y4DtzADzd1}
}
```
