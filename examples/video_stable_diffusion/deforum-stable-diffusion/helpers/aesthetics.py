import os
import torch
from .simulacra_fit_linear_model import AestheticMeanPredictionLinearModel
import requests

def wget(url, outputdir):
    filename = url.split("/")[-1]

    ckpt_request = requests.get(url)
    request_status = ckpt_request.status_code

    # inform user of errors
    if request_status == 403:
        raise ConnectionRefusedError("You have not accepted the license for this model.")
    elif request_status == 404:
        raise ConnectionError("Could not make contact with server")
    elif request_status != 200:
        raise ConnectionError(f"Some other error has ocurred - response code: {request_status}")

    # write to model path
    with open(os.path.join(outputdir, filename), 'wb') as model_file:
        model_file.write(ckpt_request.content)


def load_aesthetics_model(args,root):

    clip_size = {
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
        "ViT-L/14@336px": 768,
    }

    model_name = {
        "ViT-B/32": "sac_public_2022_06_29_vit_b_32_linear.pth",
        "ViT-B/16": "sac_public_2022_06_29_vit_b_16_linear.pth",
        "ViT-L/14": "sac_public_2022_06_29_vit_l_14_linear.pth",
    }
    
    if not os.path.exists(os.path.join(root.models_path,model_name[args.clip_name])):
    	print("Downloading aesthetics model...")
    	os.makedirs(root.models_path, exist_ok=True)
    	wget("https://github.com/crowsonkb/simulacra-aesthetic-models/raw/master/models/"+model_name[args.clip_name], root.models_path)
    
    aesthetics_model = AestheticMeanPredictionLinearModel(clip_size[args.clip_name])
    aesthetics_model.load_state_dict(torch.load(os.path.join(root.models_path,model_name[args.clip_name])))

    return aesthetics_model.to(root.device)
