import os
import pickle
import random
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,Subset
from torch.cuda import amp
import timm

def train_one_epoch(train_dataloader,epoch,cfg,model,criterion,optimizer,
                    scaler,lr_scheduler,val_loss,i):
    pbar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch}/"+str(cfg.epoch) )
    train_loss=0
    model.train()
    for seq_data,label,eeg_id,eeg_sub_id in pbar:
        seq_data = seq_data.to(cfg.device).float()
        label    = label.to(cfg.device).float()
        optimizer.zero_grad()
        with amp.autocast(True):
            y = model(seq_data)
            #y = nn.functional.softmax(y,  dim=1)
            loss = criterion(y, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()/len(train_dataloader)
        lr_scheduler.step()
        pbar.set_postfix(loss=train_loss,valid_loss=val_loss,fold= i )
    return train_loss

def valid_one_epoch(valid_dataloader,epoch,cfg,model,loss_func_val):
    model.eval()
    eeg_list=[]
    sub_list=[]
    pred_list=[]

    val_loss=0
    pred_val=pd.DataFrame()
    TARGET=["seizure_vote","lpd_vote","gpd_vote","lrda_vote","grda_vote","other_vote"]
    
    for seq_data,label,eeg_id,eeg_sub_id in valid_dataloader:
        tmp=pd.DataFrame(columns=["eeg_id","eeg_sub_id"]+TARGET)
        seq_data = seq_data.to(cfg.device).float()
        label    = label
        with torch.no_grad(), amp.autocast(True):
            y = model(seq_data)
            pred = nn.functional.softmax(y,  dim=1)
        y = y.detach().cpu().to(torch.float32)
        loss_func_val(y, label)
        tmp["eeg_id"]=eeg_id
        tmp["eeg_sub_id"]=eeg_sub_id
        tmp[TARGET]=pred.detach().cpu().numpy()
        pred_val=pd.concat([pred_val,tmp],axis=0)

    val_loss = loss_func_val.compute()

    return val_loss,pred_val