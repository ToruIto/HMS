from scipy.signal import butter, lfilter
import albumentations as A
import random
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader,Subset

import torch

def make_pl_exper():
    FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]
    expr_list =[]
    for i in range(4):
        for j in range(4):
            n1=FEATS[i][j]
            n2=FEATS[i][j+1]
            expr_list.append((pl.col(n1)-pl.col(n2)).alias(n1+"-"+n2))

    FEATS = [['Fp1','Fp2'],['F7','F8'],['F3','F4'],
            ['T3','T4'],['C3','C4'],['T5','T6'],['O1','O2']]
    for i in range(7):
            n1=FEATS[i][0]
            n2=FEATS[i][1]
            expr_list.append((pl.col(n1)-pl.col(n2)).alias(n1+"-"+n2))

    return expr_list




def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    # bins = np.linspace(-1, 1, classes)
    # quantized = np.digitize(mu_x, bins) - 1
    return mu_x#quantized

def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x

def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s

def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data



def cutmix(self,seq,n):
    if n%100 >= 100-self.cut_a:
        dl=int(len(seq)*np.random.randint(0,100)*0.01)
        seq=pl.concat([seq[dl:],seq[:dl]])
    return seq



class HMS_EEG_Dataset(Dataset):
    def __init__(self, label_data, duration, valid_mode = False):

        self.label_data=label_data
        self.duration=int(duration)
        self.valid=valid_mode
        self.exp=make_pl_exper()

    def __len__(self):
        return len(self.label_data)

    def aug(self,seq):
        if np.random.randint(1,100) > 50:
            seq=seq.with_columns(-pl.col(pl.Float32))
        if np.random.randint(1,100) > 50:
            seq=seq[::-1]
        return seq

    def __getitem__(self, index):
        path        = "/content/train_eegs/"
        eeg_id      = str(self.label_data.iloc[index].loc["eeg_id"])
        eeg_sub_id  = str(self.label_data.iloc[index].loc["eeg_sub_id"])
        off_set     = int(self.label_data.iloc[index].loc["eeg_label_offset_seconds"]*200)

        crop=np.random.randint(-1000,1000)
        if self.valid:
            crop=0
        seq_data = pl.scan_parquet(path+eeg_id+".parquet")

        init = int(off_set+5000-self.duration+crop)
        end  = int(off_set+5000+self.duration+crop)
        seq_data = seq_data[init:end].collect()
        
        seq_data = seq_data.with_columns(self.exp)

        if not self.valid:
            seq_data=self.aug(seq_data)

        seq_data = seq_data.with_columns(pl.col(pl.Float32)-pl.col(pl.Float32).min())
        seq_data = seq_data.with_columns(pl.col(pl.Float32)/(pl.col(pl.Float32).abs().max()+0.0001))

        seq_data = seq_data.with_columns(pl.col(pl.Float32).fill_null(strategy="zero"))
        seq_data = np.array(seq_data)

        seq_data = butter_lowpass_filter(seq_data)
        seq_data = quantize_data(seq_data,1)


        seq_data = torch.tensor(seq_data).float()
        
        TARGET=['seizure_vote','lpd_vote','gpd_vote', 'lrda_vote','grda_vote','other_vote']


        label=self.label_data.iloc[index].loc[TARGET].values.astype(np.float32)


        return seq_data,label,eeg_id,eeg_sub_id