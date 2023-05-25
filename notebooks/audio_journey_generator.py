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


channel_means = [  2.2741,  11.2872,  -3.3938,  -1.5556,  -0.0302,   7.6089,  -5.5797,
          0.2140,  -0.3536,   6.0188,   1.8582,  -0.1103,   2.2026,  -7.0081,
         -0.0721,  -8.7742,  -2.4182,   4.4447,  -0.2184,  -0.5209, -11.9494,
         -4.0776,  -1.4555,  -1.6505,   6.4522,   0.0997,  10.4067,  -3.9268,
         -7.0161,  -3.1253,  -8.5145,   3.1156,   2.2279,  -5.2728,   2.8541,
         -3.3980,  -1.1775,  -9.7662,   0.3048,   3.8765,   4.5021,   2.6239,
         14.1057,   3.2852,   1.9702,  -1.6345,  -4.3733,   3.8198,   1.1421,
         -4.4388,  -5.3498,  -6.6044,  -0.4426,   2.8000,  -7.0858,   2.4989,
         -1.4915,  -6.1275,  -3.0896,   1.1227,  -8.7984,  -4.9831,  -0.3888,
         -3.1017,  -7.5745,  -2.4760,   1.0540,  -2.5350,   0.0999,   0.6126,
         -1.2301,  -5.8328,  -0.7275,  -1.2316,  -2.2532, -11.5017,   0.9166,
         -2.2268,  -2.8496,  -0.5093,  -0.3037,  -6.3689,  -9.5225,   4.5965,
          3.1329,  -1.8315,   5.3135,  -3.8361,   1.6335,  -0.1705,  11.0513,
          5.3907,  -0.2660,   4.6109,  -8.9019,   6.5515,   0.8596,  16.6196,
         -0.7732,   4.1237,   2.9267,   9.9652,   4.6615,   1.4660,  -9.7225,
         -1.5841,  -0.5714,  -4.3343,  -0.1914,   2.8624, -11.2139,  -2.5840,
         -6.7120,   0.2601,  -5.4195,   0.3554,   3.0438,  -1.0295,   1.3360,
         -4.1767,   0.6468,   1.8145,   1.7140,   3.0185,   0.4881,   0.5796,
         -2.4755,   2.6202]
channel_stds = [1.7524, 1.2040, 1.1098, 1.1021, 1.3688, 1.1374, 1.8660, 0.9791, 1.4331,
        1.7740, 1.2690, 1.0297, 0.9953, 1.5363, 1.2166, 1.6564, 1.4858, 1.2349,
        1.5086, 1.0814, 1.4421, 0.9258, 0.9343, 1.2007, 1.3848, 1.2732, 1.7759,
        1.3544, 1.4707, 1.2685, 1.7004, 1.2947, 1.2967, 1.8925, 0.9231, 0.7637,
        1.3777, 1.6680, 0.9658, 0.9257, 0.5259, 0.9949, 1.7375, 1.0734, 1.2916,
        0.8570, 0.6263, 0.9911, 0.9574, 0.9979, 1.5969, 1.1886, 1.1147, 1.2280,
        2.0169, 1.1813, 1.2589, 1.1162, 1.3689, 1.2516, 1.2139, 1.0343, 1.1895,
        1.1726, 1.1923, 1.2714, 1.0043, 0.6465, 1.3860, 1.4449, 0.9567, 1.0218,
        0.9560, 1.4757, 1.0544, 0.8112, 1.4364, 1.0843, 1.2569, 1.0138, 1.1886,
        0.8627, 1.1016, 1.4231, 1.3607, 1.1215, 1.9759, 1.5381, 0.9219, 0.8572,
        0.6288, 0.8029, 1.1699, 1.1962, 1.5783, 0.9037, 1.2214, 2.0878, 1.3015,
        1.2254, 1.2898, 1.5421, 1.2834, 1.7237, 1.3471, 0.8689, 1.2807, 1.2174,
        1.2048, 0.6644, 1.5379, 1.4997, 0.7932, 0.7638, 0.8680, 1.3108, 1.8261,
        1.3964, 1.2147, 1.1391, 1.0011, 1.5988, 1.5721, 1.0963, 1.4303, 1.3737,
        1.5043, 1.3079]

def rescale(image, device):
    image = torch.Tensor.view(image, [128, 24, 21]).clone()
    mean_v = torch.tensor(channel_means).to(device)
    std_v =torch.tensor(channel_stds).to(device)
    image = image * std_v.view(-1,1,1) + mean_v.view(-1,1,1)
    
    # for row, mean, std in zip(image, channel_means, channel_stds):
    #     row *= std 
    #     row += mean
    
    image = torch.Tensor.view(image, [1, 128, 504])
    return image


def get_clap(device):
    model_path = "/u/li19/data_folder/model_cache/audio_journey_clap"
    pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    device_map=None, 
    safety_checker=None,
    low_cpu_mem_usage=False)

    pipe.to(device)
    pipe.vae_scale_factor = 1
    
    tokenizer = AutoTokenizer.from_pretrained("cvssp/audioldm-m-full", model_max_length=512,  subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained("cvssp/audioldm-m-full", subfolder="text_encoder")

    # === Fix for using masking on text encoder ===

    text_encoder.config.use_attention_mask = True
    
    pipe.tokenizer = tokenizer
    pipe.text_encoder = text_encoder.to(device)
    
    conf = "/u/li19/data_folder/testing/audio_journey_clap_attn_mask_restart/checkpoint-30000/unet/"

    new_unet = UNet2DConditionModel.from_pretrained(conf)
    new_unet.to(device)
    pipe.unet = new_unet
    
    noise_scheduler = DDIMScheduler.from_pretrained("/u/li19/data_folder/model_cache/audio_journey_128_ddim_2", subfolder="scheduler")
    pipe.scheduler = noise_scheduler

    return pipe
    

def get_pipe(device):
    model_path = "/u/li19/data_folder/model_cache/audio_journey_128"
    ckpt_path = "/u/li19/data_folder/testing/audio_journey_128_channel_norm/checkpoint-34000/unet"
    tokenizer = AutoTokenizer.from_pretrained("t5-large", model_max_length=512)
    text_encoder = T5EncoderModel.from_pretrained("t5-large")
    noise_scheduler = PNDMScheduler.from_pretrained("/u/li19/data_folder/model_cache/audio_journey_128", subfolder="scheduler")
    new_unet = UNet2DConditionModel.from_pretrained(ckpt_path)
    print(f'{device}: components loaded')
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        device_map=None, 
        safety_checker=None,
        low_cpu_mem_usage=False)
    
    print(f'{device}: pipe loaded')
    pipe.vae_scale_factor = 1
    pipe.to(device)
    pipe.unet = new_unet.to(device)
    pipe.scheduler = noise_scheduler
    pipe.tokenizer = tokenizer
    pipe.text_encoder = text_encoder.to(device)
    
    return pipe

def generate_audio(chunk, prompts, device, folder="/u/li19/data_folder/audioset_60k", batch_size = 8, steps=65):
    print(f'Starting audio method on device {device}')
    pipe = get_pipe(device)
    # pipe = get_clap(device)
    
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
        gens = pipe(prompt=batch_prompts, width=504, height=128, num_inference_steps=steps, guidance_scale=6.5, output_type="latent", unet_mask=False, debug=False).images
                
        for j in range(batch_size):
            prompt = batch_prompts[j]
            out_name = prompt+".npy"
            out_name = out_name.replace("/", "")
            out_path = os.path.join(folder, out_name)
            # gen = rescale(gens[j], device)
            gen = gens[j]
            
            with open(out_path, 'wb') as f:
                np.save(f, gen.detach().cpu().numpy())


if __name__ == "__main__":
    target_file = "/u/li19/data_folder/ablation/t5_with_mask"
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

    # generate_audio(ids, captions, 1, folder="/u/li19/data_folder/aud_journ_eval_encodec_45", steps=45)
    generate_audio(ids, captions, 3, folder=target_file, steps=65)

#     slices = ids.chunk(n_gpu)
        
#     print("Creating Threads")
#     threads = [threading.Thread(target=generate_audio, args=(slices[i], captions, i, target_file)) for i in range(n_gpu)]
#     print("Starting Threads")
#     for thread in threads:
#         thread.start()
        
#     for thread in threads:
#         thread.join()
        
    print("Done!")
    