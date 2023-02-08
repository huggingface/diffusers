import os
import torch

# Decodes the image without passing through the upscaler. The resulting image will be the same size as the latent
# Thanks to Kevin Turner (https://github.com/keturn) we have a shortcut to look at the decoded image!
def make_linear_decode(model_version, device='cuda:0'):
    v1_4_rgb_latent_factors = [
        #   R       G       B
        [ 0.298,  0.207,  0.208],  # L1
        [ 0.187,  0.286,  0.173],  # L2
        [-0.158,  0.189,  0.264],  # L3
        [-0.184, -0.271, -0.473],  # L4
    ]

    if model_version[:5] == "sd-v1":
        rgb_latent_factors = torch.Tensor(v1_4_rgb_latent_factors).to(device)
    else:
        raise Exception(f"Model name {model_version} not recognized.")

    def linear_decode(latent):
        latent_image = latent.permute(0, 2, 3, 1) @ rgb_latent_factors
        latent_image = latent_image.permute(0, 3, 1, 2)
        return latent_image

    return linear_decode

def load_model(root, load_on_run_all=True, check_sha256=True):

    import requests
    import torch
    from ldm.util import instantiate_from_config
    from omegaconf import OmegaConf
    from transformers import logging
    logging.set_verbosity_error()

    try:
        ipy = get_ipython()
    except:
        ipy = 'could not get_ipython'

    if 'google.colab' in str(ipy):
        path_extend = "deforum-stable-diffusion"
    else:
        path_extend = ""

    model_map = {
        "512-base-ema.ckpt": {
            'sha256': 'd635794c1fedfdfa261e065370bea59c651fc9bfa65dc6d67ad29e11869a1824',
            'url': 'https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/512-base-ema.ckpt',
            'requires_login': True,
            },
        "v1-5-pruned.ckpt": {
            'sha256': 'e1441589a6f3c5a53f5f54d0975a18a7feb7cdf0b0dee276dfc3331ae376a053',
            'url': 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt',
            'requires_login': True,
            },
        "v1-5-pruned-emaonly.ckpt": {
            'sha256': 'cc6cb27103417325ff94f52b7a5d2dde45a7515b25c255d8e396c90014281516',
            'url': 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt',
            'requires_login': True,
            },
        "sd-v1-4-full-ema.ckpt": {
            'sha256': '14749efc0ae8ef0329391ad4436feb781b402f4fece4883c7ad8d10556d8a36a',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-2-original/blob/main/sd-v1-4-full-ema.ckpt',
            'requires_login': True,
            },
        "sd-v1-4.ckpt": {
            'sha256': 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt',
            'requires_login': True,
            },
        "sd-v1-3-full-ema.ckpt": {
            'sha256': '54632c6e8a36eecae65e36cb0595fab314e1a1545a65209f24fde221a8d4b2ca',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-3-original/blob/main/sd-v1-3-full-ema.ckpt',
            'requires_login': True,
            },
        "sd-v1-3.ckpt": {
            'sha256': '2cff93af4dcc07c3e03110205988ff98481e86539c51a8098d4f2236e41f7f2f',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-3-original/resolve/main/sd-v1-3.ckpt',
            'requires_login': True,
            },
        "sd-v1-2-full-ema.ckpt": {
            'sha256': 'bc5086a904d7b9d13d2a7bccf38f089824755be7261c7399d92e555e1e9ac69a',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-2-original/blob/main/sd-v1-2-full-ema.ckpt',
            'requires_login': True,
            },
        "sd-v1-2.ckpt": {
            'sha256': '3b87d30facd5bafca1cbed71cfb86648aad75d1c264663c0cc78c7aea8daec0d',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-2-original/resolve/main/sd-v1-2.ckpt',
            'requires_login': True,
            },
        "sd-v1-1-full-ema.ckpt": {
            'sha256': 'efdeb5dc418a025d9a8cc0a8617e106c69044bc2925abecc8a254b2910d69829',
            'url':'https://huggingface.co/CompVis/stable-diffusion-v-1-1-original/resolve/main/sd-v1-1-full-ema.ckpt',
            'requires_login': True,
            },
        "sd-v1-1.ckpt": {
            'sha256': '86cd1d3ccb044d7ba8db743d717c9bac603c4043508ad2571383f954390f3cea',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-1-original/resolve/main/sd-v1-1.ckpt',
            'requires_login': True,
            },
        "robo-diffusion-v1.ckpt": {
            'sha256': '244dbe0dcb55c761bde9c2ac0e9b46cc9705ebfe5f1f3a7cc46251573ea14e16',
            'url': 'https://huggingface.co/nousr/robo-diffusion/resolve/main/models/robo-diffusion-v1.ckpt',
            'requires_login': False,
            },
        "wd-v1-3-float16.ckpt": {
            'sha256': '4afab9126057859b34d13d6207d90221d0b017b7580469ea70cee37757a29edd',
            'url': 'https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-float16.ckpt',
            'requires_login': False,
            },
    }

    # config path
    ckpt_config_path = root.custom_config_path if root.model_config == "custom" else os.path.join(root.configs_path, root.model_config)

    if os.path.exists(ckpt_config_path):
        print(f"{ckpt_config_path} exists")
    else:
        print(f"Warning: {ckpt_config_path} does not exist.")
        ckpt_config_path = os.path.join(path_extend,"configs",root.model_config)
        print(f"Using {ckpt_config_path} instead.")
        
    ckpt_config_path = os.path.abspath(ckpt_config_path)

    # checkpoint path or download
    ckpt_path = root.custom_checkpoint_path if root.model_checkpoint == "custom" else os.path.join(root.models_path, root.model_checkpoint)
    ckpt_valid = True

    if os.path.exists(ckpt_path):
        pass
    elif 'url' in model_map[root.model_checkpoint]:
        url = model_map[root.model_checkpoint]['url']

        # CLI dialogue to authenticate download
        if model_map[root.model_checkpoint]['requires_login']:
            print("This model requires an authentication token")
            print("Please ensure you have accepted the terms of service before continuing.")

            username = input("[What is your huggingface username?]: ")
            token = input("[What is your huggingface token?]: ")

            _, path = url.split("https://")

            url = f"https://{username}:{token}@{path}"

        # contact server for model
        print(f"..attempting to download {root.model_checkpoint}...this may take a while")
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
        with open(os.path.join(root.models_path, root.model_checkpoint), 'wb') as model_file:
            model_file.write(ckpt_request.content)
    else:
        print(f"Please download model checkpoint and place in {os.path.join(root.models_path, root.model_checkpoint)}")
        ckpt_valid = False
        
    print(f"config_path: {ckpt_config_path}")
    print(f"ckpt_path: {ckpt_path}")

    if check_sha256 and root.model_checkpoint != "custom" and ckpt_valid:
        try:
            import hashlib
            print("..checking sha256")
            with open(ckpt_path, "rb") as f:
                bytes = f.read() 
                hash = hashlib.sha256(bytes).hexdigest()
                del bytes
            if model_map[root.model_checkpoint]["sha256"] == hash:
                print("..hash is correct")
            else:
                print("..hash in not correct")
                ckpt_valid = False
        except:
            print("..could not verify model integrity")

    def load_model_from_config(config, ckpt, verbose=False, device='cuda', half_precision=True,print_flag=False):
        map_location = "cuda" # ["cpu", "cuda"]
        print(f"..loading model")
        pl_sd = torch.load(ckpt, map_location=map_location)
        if "global_step" in pl_sd:
            if print_flag:
                print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if print_flag:
            if len(m) > 0 and verbose:
                print("missing keys:")
                print(m)
            if len(u) > 0 and verbose:
                print("unexpected keys:")
                print(u)

        if half_precision:
            model = model.half().to(device)
        else:
            model = model.to(device)
        model.eval()
        return model

    if load_on_run_all and ckpt_valid:
        local_config = OmegaConf.load(f"{ckpt_config_path}")
        model = load_model_from_config(local_config, f"{ckpt_path}", half_precision=root.half_precision)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

    autoencoder_version = "sd-v1" #TODO this will be different for different models
    model.linear_decode = make_linear_decode(autoencoder_version, device)

    return model, device


def get_model_output_paths(root):

    models_path = root.models_path
    output_path = root.output_path

    #@markdown **Google Drive Path Variables (Optional)**
    
    force_remount = False

    try:
        ipy = get_ipython()
    except:
        ipy = 'could not get_ipython'

    if 'google.colab' in str(ipy):
        if root.mount_google_drive:
            from google.colab import drive # type: ignore
            try:
                drive_path = "/content/drive"
                drive.mount(drive_path,force_remount=force_remount)
                models_path = root.models_path_gdrive
                output_path = root.output_path_gdrive
            except:
                print("..error mounting drive or with drive path variables")
                print("..reverting to default path variables")

    models_path = os.path.abspath(models_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    print(f"models_path: {models_path}")
    print(f"output_path: {output_path}")

    return models_path, output_path
