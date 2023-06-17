import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
from pathlib import Path
import os
from torchvision import transforms

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudiosetDataset(Dataset):
    def __init__(self, audio_conf, tokenizer=None, device=None, dtype=None, latent_folder=None, label_csv=None, wav_folder=None, channels=4, logger=None, mixup_ratio=0, mixup_type = 'random', n_mixup = 2):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        :param mixup_ratio: portion of the data which we perform mixup
        :param n_mixup: when doing temporal mixup, how many audio samples do we mixup
        """
        self.datapath = audio_conf["dataset_json_file"]
        self.latent_folder = audio_conf["latent_folder_path"]
        self.wav_folder = audio_conf["wav_folder"]
        if ('formant_folder' in audio_conf):
            self.formant_folder = audio_conf['formant_folder']
        else:
            self.formant_folder = None
            
        if ('strong_label_dir' in audio_conf):
            self.strong_label_dir = audio_conf['strong_label_dir']
            self.make_data_lookup()
            
            tsv_file = open(self.strong_label_dir, 'r')
            reader = csv.DictReader(myFile, delimiter="\t")
            self.strong_label_data = list()
            for dictionary in reader:
                dictionary['start_time_seconds'] = float(dictionary['start_time_seconds'])
                dictionary['end_time_seconds'] = float(dictionary['end_time_seconds'])
                dictionary['length'] = dictionary['end_time_seconds'] - dictionary['start_time_seconds']
                self.strong_label_data.append(dictionary)
                
            len_max = 5.0
            len_min = 1.5
            self.strong_label_data = [el for el in self.strong_label_data if el['length'] > len_min and el['length'] < len_max]
        else:
            self.strong_label_dir = None
        
        
        if ("norm_mean" in audio_conf.keys() and "norm_std" in audio_conf.keys()):
            self.norm_mean = audio_conf["norm_mean"]
            self.norm_std = audio_conf["norm_std"]
        else:
            self.norm_mean = None
            self.norm_std = None
        
        
        with open(self.datapath, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.dataset = self.datapath
        self.tokenizer = tokenizer
        self.device = device
        self.channels = channels
        self.logger = logger
        self.mixup_ratio = mixup_ratio
        self.mixup_type = mixup_type
        self.n_mixup = n_mixup
        if (dtype is None):
            self.dtype = torch.float16
        else:
            self.dtype = dtype

            
    def mixup_tensor_set(self, tensors, ids=None):
        total_size = tensors[0].shape[2]
        n_tensors = len(tensors)
        slice_size = total_size // n_tensors
        shakeup_size = slice_size // 4

        final = torch.clone(tensors[0])
        
        if (ids is None):
            ids = list(range(n_tensors))
        prev_spot = 0
        for i in range(n_tensors):
            if (i != n_tensors - 1):
                end = (i + 1) * slice_size
                end += random.random()*shakeup_size - (shakeup_size/2)
                end = int(end)
            else:
                end = total_size
            final[:, :, prev_spot:end] = tensors[ids[i]][:, :, prev_spot:end]
            prev_spot = end

        return final
    
    def mixup_label_set(self, labels, ids):
        final = [labels[i] for i in ids]
        return " and then ".join(final)

    def mixup_tensor(self, t1, t2, beta=0.5):
        split_idx = int(t1.shape[2] * beta)
        
        t1[:, :, split_idx:] = t2[:, :, split_idx:]
                
        return t1
    
    def make_data_lookup(self):
        self.data_lookup = {datum['wav'].split("/")[-1].split(".")[0]: datum for datum in self.data}
        
    def make_mid_to_name(self, path):
        tsv_file = open(path, 'r')
        reader = csv.Reader(myFile, delimiter="\t")
        self.mid_to_name = {}
        for line in reader:
            self.mid_to_name[line[0]] = line[1]
            self.strong_label_data.append(dictionary)
    
    def mixup_strong(self, idx):
        pass
    
    def get_strong_datum(self, idx):
        datum = self.strong_label_data[idx]
        seg_id = datum['segment_id'][:11]
        ref_datum = self.data_lookup[seg_id]
        waveform, label, wav, sr, latent, formant = gather_data(ref_datum)
        
        total_len = latent.shape[2]
        latent_t0 = int(datum['start_time_seconds'] * total_len)
        latent_t1 = int(datum['end_time_seconds'] * total_len)
        strong_latent = latent[latent_t0 : latent_t1]
        
        if (formant):
            formant_total_len = formant.shape[2]
            formant_t0 = int(datum['start_time_seconds'] * formant_total_len)
            formant_t1 = int(datum['end_time_seconds'] * formant_total_len)
            strong_formant = formant[formant_t0 : formant_t1]
        
        label = datum['label']
        return waveform, label, wav, sr, latent, formant
    
    def gather_data(self, datum):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """        
        if (self.latent_folder is None):
            if (self.wav_folder):
                path = datum['wav'].split("/")[-1]
                path = os.path.join(self.wav_folder, path)
                waveform, sr = torchaudio.load(path)
            else:
                waveform, sr = torchaudio.load(datum['wav'])
            waveform = waveform - waveform.mean()
            m = torch.nn.ConstantPad1d((0, 1_000), 0)    
            waveform = m(waveform)
        else:
            waveform = torch.Tensor([0])
            sr = 1
            
        if "av_caption" in datum.keys():
            label = datum["av_caption"]
        else:
            label = datum["caption"]
        
            
        latent = torch.Tensor([0])
        formant = torch.Tensor([0])
        
        if (self.latent_folder):
            file_name = Path(datum["wav"]).name.split(".")[0]
            latent = np.load(f'{os.path.join(self.latent_folder,file_name)}.npy', allow_pickle=True)
            latent = torch.from_numpy(latent)[None, :, :]
            if (self.channels != 1):
                latent = latent.repeat(self.channels, 1, 1)
                
        if (self.formant_folder):
            file_name = Path(datum["wav"]).name.split(".")[0]
            formant = np.load(f'{os.path.join(self.formant_folder,file_name)}.npy', allow_pickle=True)
        return waveform, label, datum['wav'], sr, latent, formant
    
    def idx_to_datum(self, index):
        
        waveform, label, path, sr, latent, formant = self.gather_data(self.data[index])
            
        do_mixup = random.random() <= self.mixup_ratio
        
        if (do_mixup):
            if (self.mixup_type == 'random'):
                wavs = [waveform]
                labels = [label]
                latents = [latent]
                formants = [formant]

                for i in range(self.n_mixup - 1):
                    ids = list(range(self.n_mixup))

                    if (self.n_mixup > 2):
                        random.shuffle(ids)

                    mix_index = random.randrange(len(self.data))
                    mix_wav, mix_label, mix_path, mix_st, mix_latent, mix_formant = self.gather_data(self.data[mix_index])

                    wavs.append(mix_wav)
                    labels.append(mix_label)
                    latents.append(mix_latent)
                    formants.append(mix_formant)

                if (self.latent_folder is None):
                    # waveform = self.mixup_tensor(waveform, waveform2, split)
                    pass
                else:
                    latent = self.mixup_tensor_set(latents, ids)

                if (self.formant_folder):
                    formant = self.mixup_tensor_set(formants, ids)

                label = self.mixup_label_set(labels, ids)
                
            elif (self.mixup_type == 'strong'):
                rand_mixup_idx = random.randrange(len(self.
                     
            else:
                print("no other mixup types are implemented yet")
        input_ids = []
        attn_mask = None
        if (self.tokenizer is not None):
            tokens = self.tokenizer(
                [label], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            input_ids = tokens["input_ids"][0]
            attn_mask = tokens["attention_mask"][0]
            
            
        if (self.device is not None):
            waveform = waveform.to(self.device, dtype=self.dtype)
            input_ids = input_ids.to(self.device, dtype=torch.int32)
            if (latent is not None):
                latent = latent.to(self.device, dtype=self.dtype)
        
        #TODO add attention mask as a key:
        batch = {"waveform": waveform, "input_ids": input_ids, "caption": label, 'filename': path, 'sample_rate': sr, "latent": latent, "attn_mask": attn_mask, 'formant': formant}
        
        return batch
    def __getitem__(self, index):
        return self.idx_to_datum(index)

    def __len__(self):
        return len(self.data)

    def construct_label(self, label_indices):
        #base string to add labels to
        label = "A mel spectrogram of the sound of "
        # pulling english label names from non-zero indicies of multi-hot encoded labels
        terms = [self.name_dict[str(int(idx))] for idx in (label_indices >= 0).nonzero(as_tuple=True)[0]]
        # converting list of labels to oxford comma single string
        label += ", ".join(terms[:-2] + [", and ".join(terms[-2:])])
        return label