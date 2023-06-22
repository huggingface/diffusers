import os

import torch
from modules import Paella
from vqgan import VQModel

from diffusers import PaellaVQModel


model_path = "models/"
device = "cpu"

paella_vqmodel = VQModel()
state_dict = torch.load(os.path.join(model_path, "vqgan_f4_v1_500k.pt"), map_location=device)["state_dict"]
paella_vqmodel.load_state_dict(state_dict)

state_dict["vquantizer.embedding.weight"] = state_dict["vquantizer.codebook.weight"]
state_dict.pop("vquantizer.codebook.weight")
vqmodel = PaellaVQModel(
    codebook_size=paella_vqmodel.codebook_size,
    c_latent=paella_vqmodel.c_latent,
)

vqmodel.load_state_dict(state_dict)

# test vqmodel outputs match paella_vqmodel outputs


state_dict = torch.load(os.path.join(model_path, "paella_v3.pt"), map_location=device)
paella_model = Paella(byt5_embd=2560).to(device)
paella_model.load_state_dict(state_dict)
