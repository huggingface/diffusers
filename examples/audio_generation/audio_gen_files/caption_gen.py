import os
from tqdm import tqdm
import csv
import random
import llm_client
from AudiosetDataset import AudiosetDataset
import torch
import json
import threading
import time

def gen_caption(classes, client):
    human_sample = ", ".join(classes[:-2] + [", and ".join(classes[-2:])])
    prompt = ""
    
    preface = "describe a situation with all of these sounds together:"
    
    prompt += "For each of the 3 line summarize the sounds into a single sentence: \n"
    
    prompt += preface + " (alarm, burp, inside, small room)." + "\n > burping while an alarm plays inside a small room. \n\n"
    
    prompt += preface + " (dog, bark, howl, speech)." + "\n > a dog barking and howling with a person speaking aswell. \n\n"
    
    prompt += preface + " (Music, jazz, piano, singing, speaking)." + "\n > a person plays jazz piano with a singer while people talk. \n\n"
    
    prompt += preface + " (engine, vehicle, wind, music, speech)." + "\n > people talking inside a car while driving and listening to music. \n\n"
    
    prompt += preface + " (water, garggle, inside, small room)." + "\n >air is passing through the water in their mouth in a small room with water. \n\n"
        
    prompt += preface + " (scratch, hammer, metal)." + "\n > hammer striking a metal surface and scratching sounds can be heard \n\n"
        
    prompt += preface + " (thunder, wind, bark, small room)." + "\n > a dog is barking in a small room during a thunderstorm with audible wind. \n\n"
        
    prompt += preface + " (gunshot, vehicle engine, siren, crash)." + "\n > a car chase with gunfire and sirens where a vehicle crashes. \n\n"
    
    prompt += preface + " (waterfall, wind, sizzle, crackle)." + "\n > a fire is cracking with something sizzling near a waterfall with wind. \n\n"
    
    prompt += preface + " (stream, cough, cat, Purr)." + "\n > a cat purrs near a coughing person while a stream can be heard. \n\n"
    
    
    
    prompt += "imagine a scene with all of these sounds existing together: " + f"({human_sample}). \n >"
    outputs = client.prompt(prompt)
    outputs = outputs[0].text[len(prompt) + 1:]
    final_output = outputs.split("\n")[0]
    return final_output


def process(start, stop, client, batch_size = 1):
    t_0 = time.time()
    of = open(f"logs/{start}_{stop}_cache_unbal.txt", "a")
    total = stop - start
    log_step = total // 100
    for i in range(start, stop, batch_size):
        prog = i - start
        if (prog%log_step == 0 and prog != 0):
            t_1 = time.time()
            percent = prog / total
            print(f"PROGRESS ON {start}_{stop}: {percent * 100 :.2f}%")
            d_t = t_1 - t_0
            print(f"ETA: {d_t / percent}")
            
            
        datum = data["data"][i]
        labels = datum['labels'].split(",")
        labels = [label_dict[label] for label in labels]

        cap = gen_caption(labels, client)

        datum["caption"] = cap
        datum["classes"] = labels
        of.write(json.dumps(datum) + "\n")
        of.flush()

if __name__ =="__main__":
    local = "/home/junchenl/AudioTaggingDoneRight/egs/audioset/data"
    label_csv=os.path.join(local, 'class_labels_indices.csv')
    data_path = os.path.join(local,'datafiles/audioset_bal_unbal_train_data.json')

    ports = [4343, 4342, 4341, 4340]
    addresses = ["tir-1-18", "tir-1-18", "tir-1-18", "tir-1-18"]
    clients = [llm_client.Client(address=ip, port=ports) for ip, port in zip(addresses, ports)]

    data_file = open(data_path)

    data = json.load(data_file)

    label_dict = {}
    with open(label_csv,'r') as data_file:
        for line in csv.reader(data_file):
            label_dict[line[1]] = line[2]
        
    total = len(data["data"])
    n_slices = len(ports)
    stride = total // n_slices
    parts = [[i * stride, ((i + 1) * stride) - 1] for i in range(n_slices)]
    parts [-1][-1] = total - 1
    
    threads = [threading.Thread(target=process, args=(parts[i][0], parts[i][1], clients[i])) for i in range(n_slices)]
    
    
    for thread in threads:
        thread.start()
        
    for thread in threads:
        thread.join()
        
    output_path = os.path.join(local,'datafiles/multi_alpaca_audioset_bal_train_data.json')

    out_file = open(output_path, "w")
    json.dump({"data": data["data"]}, out_file)