import torch
import torchaudio
from torch.utils.data import Dataset

class GSCDataset(Dataset):
    def __init__(self, split="train", root="./data/gsc_data"):
        self.split = split
        
    def __len__(self):
        return 0  
        
    def __to_mfcc(self, waveform, sample_rate):
        transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=10,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
        )
        mfcc = transform(waveform)
        return mfcc
        
    def __getitem__(self, idx):
        pass
