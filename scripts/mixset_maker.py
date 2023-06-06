import json
import torchaudio
import IPython.display as ipd
import os
from tqdm import tqdm
import random



def get_id(path):
    p_end = path.split("/")[-1]
    yid = p_end.split(".")[0]
    return yid
    
def mix_datum(d1, d2, wav_folder="/u/li19/data_folder/audioSetAudio/mixup_wav"):
    new_id = get_id(d1['wav']) + "_" + get_id(d2['wav'])
    
    new_path = os.path.join(wav_folder, new_id+".wav")
    
    mixed_wav, sr = mix_path(d1['wav'], d2['wav'])
    torchaudio.save(new_path, mixed_wav, sr)
    mixed_caption = d1['caption'] + " and " + d2['caption']
    mixed_labels = d1['labels'] + "," + d2['labels']
    mixed_classes = d1['classes'] + d2['classes']
    mixed_datum = {
        "wav": new_path,
        "labels": mixed_labels,
        "caption": mixed_caption,
        "classes": mixed_classes
    }
    return mixed_datum

def mix_path(path1, path2, beta=0.5):
    wav1, sr1 = torchaudio.load(path1)
    wav2, sr2 = torchaudio.load(path2)
    
    if (sr1 != sr2):
        print("CANNOT MIX WITHOUT SAME SAMPLE RATE")
    mixed = mix(wav1, wav2, beta)
    return mixed, sr1
    
def mix(waveform1, waveform2, beta=0.5):
    return beta * waveform1 + (1.0 - beta) * waveform2

reference_csv = "/u/li19/data_folder/AudioTaggingDoneRight/egs/audioset/data/datafiles/alpaca_bal_unbal.json"
mixup_folder = "/u/li19/data_folder/audioSetAudio/mixup_wav"

if __name__ == "__main__":
    print("reading csv")
    with open(reference_csv, 'r') as f:
        data = json.load(f)["data"]
        
    n_gen = 1_000_000
    # n_gen = 10
    new_samples = {"data": []}
    print("starting mixup")
    for i in tqdm(range(n_gen)):
        datum1 = random.choice(data)
        datum2 = random.choice(data)
        
        while(datum1['wav'] == datum2['wav']):
            datum1 = random.choice(data)
            datum2 = random.choice(data)
            
        new_datum = mix_datum(datum1, datum2)
        new_samples["data"].append(new_datum)
        
        
        
    with open("mixup_samples.json", 'w') as f:
        json.dump({'data': new_samples['data']}, f)