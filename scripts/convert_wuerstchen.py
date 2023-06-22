import argparse
import inspect
import os

import numpy as np
import torch
import torch.nn as nn

from diffusers import PaellaVQModel
from transformers import CLIPTextModel, AutoTokenizer

from vqgan import VQModel
from modules import Paella, Prior

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
# TODO: test vqmodel outputs match paella_vqmodel outputs

# Clip Text encoder and tokenizer
text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
clip_tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

# EfficientNet

# Paella
state_dict = torch.load(os.path.join(model_path, "model_stage_b.pt"), map_location=device)t['state_dict']
paella_model = Paella(byt5_embd=2560).to(device)
paella_model.load_state_dict(state_dict)

# Prior
prior_model = Prior(c_in=16, c=1536, c_cond=1024, c_r=64, depth=32, nhead=24).to(device)