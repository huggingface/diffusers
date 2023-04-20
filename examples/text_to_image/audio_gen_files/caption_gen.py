import os
from tqdm import tqdm
import csv
import random
import llm_client
from AudiosetDataset import AudiosetDataset
import torch
import json



def gen_caption(classes, client):
    human_sample = ", ".join(classes[:-2] + [", and ".join(classes[-2:])])
    prompt = ""
    prompt += "For each of the 3 line summarize the sounds into a single sentence: \n"
    prompt += "imagine a scene with all of these sounds existing together: (inside, small room)." + "\n > This is happening inside a small room. \n\n"
    prompt += "imagine a scene with all of these sounds existing together: (dog, bark, howl, speech)." + "\n > a dog barking and howling with a person speaking aswell. \n\n"
    prompt += "imagine a scene with all of these sounds existing together: " + f"({human_sample}). \n >"
    outputs = client.prompt(prompt)
    outputs = outputs[0].text[len(prompt) + 1:]
    final_output = outputs.split("\n")[0]
    return final_output


local = "/home/junchenl/AudioTaggingDoneRight/egs/audioset/data"
label_csv=os.path.join(local, 'class_labels_indices.csv')
data_path = os.path.join(local,'datafiles/audioset_bal_train_data.json')

alpaca_client = llm_client.Client(address="tir-1-7")


data_file = open(data_path)

data = json.load(data_file)

label_dict = {}
with open(label_csv,'r') as data_file:
    for line in csv.reader(data_file):
        label_dict[line[1]] = line[2]
        

        
of = open("cache.txt", "a")
    
for i in tqdm(range(len(data["data"]))):
    datum = data["data"][i]
    labels = datum['labels'].split(",")
    labels = [label_dict[label] for label in labels]
    
    cap = gen_caption(labels, alpaca_client)
    
    datum["caption"] = cap
    datum["classes"] = labels
    of.write(json.dumps(datum) + "\n")
    of.flush()
print("Done")

output_path = os.path.join(local,'datafiles/alpaca_audioset_bal_train_data.json')

out_file = open(output_path, "w")
json.dump({"data": data["data"]}, out_file)