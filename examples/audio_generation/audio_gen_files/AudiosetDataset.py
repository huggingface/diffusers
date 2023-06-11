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
    def __init__(self, audio_conf, tokenizer=None, device=None, dtype=None, latent_folder=None, label_csv=None, wav_folder=None, channels=4, logger=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = audio_conf["dataset_json_file"]
        self.latent_folder = audio_conf["latent_folder_path"]
        self.wav_folder = audio_conf["wav_folder"]
        
        
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
        if (dtype is None):
            self.dtype = torch.float16
        else:
            self.dtype = dtype

#     def _wav2fbank(self, filename, filename2=None):
#         # mixup
#         if filename2 == None:
#             waveform, sr = torchaudio.load(filename)
#             waveform = waveform - waveform.mean()
#         # mixup
#         else:
#             waveform1, sr = torchaudio.load(filename)
#             waveform2, _ = torchaudio.load(filename2)

#             waveform1 = waveform1 - waveform1.mean()
#             waveform2 = waveform2 - waveform2.mean()

#             if waveform1.shape[1] != waveform2.shape[1]:
#                 if waveform1.shape[1] > waveform2.shape[1]:
#                     # padding
#                     temp_wav = torch.zeros(1, waveform1.shape[1])
#                     temp_wav[0, 0:waveform2.shape[1]] = waveform2
#                     waveform2 = temp_wav
#                 else:
#                     # cutting
#                     waveform2 = waveform2[0, 0:waveform1.shape[1]]

#             # sample lambda from uniform distribution
#             #mix_lambda = random.random()
#             # sample lambda from beta distribtion
#             mix_lambda = np.random.beta(10, 10)

#             mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
#             waveform = mix_waveform - mix_waveform.mean()
#         if self.melbins > 100:
#             fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
#                                                   window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
#         else:
#             melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, center=False, pad_mode=None, power=2, norm=None, onesided=False, n_mels=self.melbins, mel_scale="htk")

#             spec = melspec(waveform).squeeze()

#             a2db = torchaudio.transforms.AmplitudeToDB(spec)
#             fbank = a2db(spec)
#             if (self.transpose_images):
#                 fbank = fbank.transpose(0,1)
                
                
#         target_length = self.audio_conf.get('target_length')
#         n_frames = fbank.shape[0]

#         p = target_length - n_frames

#         # cut and pad
#         if p > 0:
#             m = torch.nn.ZeroPad2d((0, 0, 0, p))
#             fbank = m(fbank)
#         elif p < 0:
#             fbank = fbank[0:target_length, :]


#         fbank = fbank[..., np.newaxis]


#         if (self.channels > 1):
#             out_shape = np.array(fbank.shape)
#             out_shape[-1] = self.channels
#             fbank2 = np.zeros(out_shape)
#             for i in range(self.channels):
#                 fbank2[:,:,i] = fbank[:,:,0]
#             fbank = torch.tensor(fbank2, requires_grad=True)


#         if filename2 == None:
#             return fbank, 0, sr
#         else:
#             return fbank, mix_lambda, sr

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        datum = self.data[index]
        
        # do mix-up for this sample (controlled by the given mixup rate)
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
        
        if (self.logger):
            self.logger.info(datum['wav'])
            self.logger.info(f'LABEL: {label}')
            
        input_ids = []
        attn_mask = None
        if (self.tokenizer is not None):
            tokens = self.tokenizer(
                [label], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            input_ids = tokens["input_ids"][0]
            attn_mask = tokens["attention_mask"][0]
            
            # print("TOKENS: ", input_ids.shape, " : ", attn_mask.shape)
            # input_ids, attn_mask = input_ids
            
        
        latent = None
        
        if (self.logger):
            self.logger.info(f'INPUT_IDS: {input_ids}')
        
        if (self.latent_folder):
            file_name = Path(datum["wav"]).name.split(".")[0]
            latent = np.load(f'{os.path.join(self.latent_folder,file_name)}.npy', allow_pickle=True)
            latent = torch.from_numpy(latent)[None, :, :]
            if (self.channels != 1):
                latent = latent.repeat(self.channels, 1, 1)
            
        if (self.device is not None):
            waveform = waveform.to(self.device, dtype=self.dtype)
            input_ids = input_ids.to(self.device, dtype=torch.int32)
            if (latent is not None):
                latent = latent.to(self.device, dtype=self.dtype)
        
        #TODO add attention mask as a key:
        batch = {"waveform": waveform, "input_ids": input_ids, "caption": label, 'filename': datum['wav'], 'sample_rate': sr, "latent": latent, "attn_mask": attn_mask}
        
        return batch

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