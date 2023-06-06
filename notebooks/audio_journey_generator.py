import json
import torch
from tqdm import tqdm
import torchaudio
from json import JSONEncoder
import numpy as np
import threading
import random
import os 
from diffusers.utils import logging
import scipy
from diffusers import DDIMScheduler,PNDMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from transformers import AutoTokenizer, T5EncoderModel,T5TokenizerFast
from encodec import EncodecModel
import csv
import torchvision.transforms as transforms
from PIL import Image

print("Modules Imported")


def get_pipe_aud(device):
    pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm-l-full", torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.unet.do_reshape = False
    return pipe

def generate_audio_aud(chunk, prompts, device, folder="/u/li19/data_folder/audioset_60k", batch_size = 1):
    print(f'Starting audio method on device {device}')
    pipe = get_pipe_aud(device)
    
    encodec = EncodecModel.encodec_model_24khz()
    encodec.set_target_bandwidth(6.0)
    encodec = encodec.to(device)
    print(f'{device}: encodec loaded')
    # print(pipe.progress_bar())
    pipe.set_progress_bar_config(disable=True)
    
    logging.disable_progress_bar()
    
    batches = torch.split(chunk, batch_size)
    pbar = tqdm(range(len(batches)))    
    for i in pbar:
        pbar.set_description(f'Processing Slice: {device}')
        chunk_idx = batches[i]
        batch_prompts = [prompts[idx].replace("/", " ") for idx in chunk_idx]        
        gens = pipe(prompt=batch_prompts, num_inference_steps=200, audio_length_in_s=10)
                
        for j in range(batch_size):
            prompt = batch_prompts[j]
            out_name = prompt+".png"
            out_name = out_name.replace("/", "")
            out_path = os.path.join(folder, out_name)
            # gen = rescale(gens[j], device)
            audio = torch.from_numpy(gens[j][None, :, :]).to(device, dtype=torch.float32)
            codes = encodec.encode(audio)[0][0]
            codes = codes.transpose(0, 1)
            gen = encodec.quantizer.decode(codes)
            # plt.imshow(gen.detach().cpu().numpy()[0])
            # plt.show()
            
            # Convert the Torch tensor to a PIL image
            tensor_to_pil = transforms.ToPILImage()
            image = tensor_to_pil(gen)
            image.save(out_path)


if __name__ == "__main__":
    target_file = "/u/li19/data_folder/eval_sets/aud_ldm_eval_gen"
    csv_path = "/u/li19/diffusers_with_dataloader/notebooks/notebook_data/audio_caps_test.csv"
    n_generate = None
    captions = []

    # skip = 1773

    with open(csv_path, 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)
        # Skip the header row
        header = next(reader)
        # Iterate over each row in the CSV file
        for row in reader:
            # Access the data in each row
            captions.append(row[-1])


    # captions = captions[skip:]

    if (n_generate is None):
        n_generate = len(captions)

    captions = captions[:n_generate]

    ids = torch.arange(n_generate)

    n_gpu = torch.cuda.device_count()

    generate_audio_aud(ids, captions, "cuda", folder=target_file)

    print("Done!")
            