from diffusers import DCAE, DCAE_HF
# from diffusers import SanaPipeline


# vae = DCAE()
dc_ae = DCAE_HF.from_pretrained(f"mit-han-lab/dc-ae-f32c32-sana-1.0")
print(dc_ae)

# pipe = Sana()





