import os
import torch
import torchaudio
from torch.utils.data import Dataset

class GSCDataset(Dataset):
    def __init__(self, split="train", root="./data/gsc_data", target_length=16000):
        self.split = split
        self.root = root
        self.target_length = target_length 
        
        self.samples = []  
        self.labels = []   
        
        if os.path.exists(self.root):
            self.labels = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
            self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
            
            for label in self.labels:
                label_dir = os.path.join(self.root, label)
                for fname in os.listdir(label_dir):
                    if fname.endswith(".wav"):
                        fpath = os.path.join(label_dir, fname)
                        self.samples.append((fpath, self.label_to_idx[label]))
        
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=10,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
        )

    def __len__(self):
        return len(self.samples)  
        
    def _to_mfcc(self, waveform, sample_rate):
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        mfcc = self.mfcc_transform(waveform)
        return mfcc
        
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        waveform, sample_rate = torchaudio.load(file_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if waveform.shape[1] < self.target_length:
            pad_size = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        elif waveform.shape[1] > self.target_length:
            waveform = waveform[:, :self.target_length]
            
        mfcc = self._to_mfcc(waveform, sample_rate)
        
        return mfcc, label
