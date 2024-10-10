# VAE

`vae_roundtrip.py` Demonstrates the use of a VAE by roundtripping an image through the encoder and decoder. Original and reconstructed images are displayed side by side.

```
cd examples/research_projects/vae
python vae_roundtrip.py \
    --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --subfolder="vae" \
    --input_image="/path/to/your/input.png"
```
