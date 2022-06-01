# Diffusers

## Library structure:

```
├── models
│   ├── dalle2
│   │   ├── modeling_dalle2.py
│   │   ├── README.md
│   │   └── run_dalle2.py
│   ├── ddpm
│   │   ├── modeling_ddpm.py
│   │   ├── README.md
│   │   └── run_ddpm.py
│   ├── glide
│   │   ├── modeling_glide.py
│   │   ├── README.md
│   │   └── run_dalle2.py
│   ├── imagen
│   │   ├── modeling_dalle2.py
│   │   ├── README.md
│   │   └── run_dalle2.py
│   └── latent_diffusion
│       ├── modeling_latent_diffusion.py
│       ├── README.md
│       └── run_latent_diffusion.py
├── src
│   └── diffusers
│       ├── configuration_utils.py
│       ├── __init__.py
│       ├── modeling_utils.py
│       ├── models
│       │   └── unet.py
│       ├── processors
│       └── samplers
│           ├── gaussian.py
├── tests
│   └── test_modeling_utils.py
```
