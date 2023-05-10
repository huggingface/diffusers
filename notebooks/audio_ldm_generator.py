import json
import torch
from tqdm import tqdm
import torchaudio
from json import JSONEncoder
import numpy
import threading
import pandas as pd
import random
import os 
from diffusers import AudioLDMPipeline
from diffusers.utils import logging
import scipy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

    
def generate_audio(start, stop, prompts, classes, device, folder="/u/li19/data_folder/audioset_60k"):
    pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm-l-full")
    pipe = pipe.to(device)
    pipe.unet.do_reshape = False
    
    # print(pipe.progress_bar())
    pipe.set_progress_bar_config(disable=True)
    
    logging.disable_progress_bar()
    
    pbar = tqdm(range(start, stop))    
    for i in pbar:
        pbar.set_description(f'Processing Slice: {start} - {stop}')
        out_name = "-".join(str(j) for j in sorted(classes[i])) + ".wav"
        out_path = os.path.join(folder, out_name)
        k = 1
        while os.path.exists(out_path):
            out_name = "-".join(str(j) for j in classes[i]) + f'_{k}.wav'
            out_path = os.path.join(folder, out_name)
            k += 1
        prompt = prompts[i]
        
        audio = pipe(prompt,num_inference_steps=50, audio_length_in_s=10)[0][0]
        scipy.io.wavfile.write(out_path, rate=16000, data=audio)

def gen_caption(classes):
    # Generate a caption
    caption = "generate sound of a"

    for i, c in enumerate(classes):
        if i == 0:
            caption += f" {c}"
        elif i == len(classes) - 1:
            caption += f", and a {c}."
        else:
            caption += f", a {c}"
    return caption
    
if __name__ == "__main__":    
    
    # pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm-l-full")

        
        
    logging.disable_progress_bar()
    
    target_file = "/u/li19/data_folder/audioSetAudio/audioset_100k"
    # os.mkdir(target_file)
    print("Reading Data")
    class_file_path = "/u/li19/data_folder/AudioTaggingDoneRight/egs/audioset/data/class_labels_indices.csv"
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(class_file_path)

    classes = df["display_name"]
    
    
    n_classes = len(classes)
    
    
    n_generations = 20_000 - 19987
    
    
    print("generating sample sizes")
    
    sample_class_number = [random.randint(1,3) for _ in range(n_generations)]
    
    
    sample_class_ids = [[random.randint(0,n_classes - 1) for _ in range(k)] for k in sample_class_number]

    
    print("generating sample classes")
    
    sample_classes = [[classes[i] for i in k] for k in sample_class_ids]
    

    print("generating prompts")
                   
    prompts = [gen_caption(classes) for classes in sample_classes]
    offset = 0
    
    n_gpu = torch.cuda.device_count()
    
    # n_gpu = 1
    
    stride = n_generations // n_gpu
    slices = [[offset + (stride * i), offset + (stride * (i + 1))] for i in range(n_gpu)]
    
    slices[-1][-1] = n_generations
    
    
    print("Creating Threads")
    threads = [threading.Thread(target=generate_audio, args=(slices[i][0], slices[i][1], prompts, sample_class_ids, i, target_file)) for i in range(n_gpu)]
    print("Starting Threads")
    for thread in threads:
        thread.start()
        
    for thread in threads:
        thread.join()
        
    print("Done!")
    