import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection
import torchvision.transforms as transforms
import torchvision.io 
import librosa
from PIL import Image
import albumentations as alb
import torch.multiprocessing as mp
import warnings

warnings.filterwarnings('ignore')
from pytorch_lightning.callbacks import ModelCheckpoint, BackboneFinetuning, EarlyStopping
import torch.nn as nn
from torch.nn.functional import cross_entropy
import torchmetrics
import timm
from pathlib import Path

from vggish_master.vggish_input import wavfile_to_examples
from tqdm import tqdm
from vggish_master.torchvggish.vggish import VGGish
from compute_vad import vad_main
from torch.utils.tensorboard import SummaryWriter
import random

CUDA_DEVICE = '3'



class Config:
    epoch_num = 50
    num_classes = 264
    batch_size = 128
    PRECISION = 16    
    seed = 114514
    model = "tf_efficientnet_b1_ns"
    pretrained = False
    use_mixup = False
    mixup_alpha = 0.2   
    DEVICE = torch.device(f'cuda:{CUDA_DEVICE}' if torch.cuda.is_available() else 'cpu')    

    data_root = "/home/chenjunhui/workspace/kaggle_competition/dataset/"

    train_path = "/home/chenjunhui/workspace/kaggle_competition/dataset/train_metadata.csv"
    
    test_path = '/home/chenjunhui/workspace/kaggle_competition/dataset/test_soundscapes/'
    
    SR = 32000
    DURATION = 5
    LR = 5e-4

    


def cut_to_2_frames(df_train, label_dict):
    print("cut audio to 2-frames pieces...")
    audio_tensor_list = []
    # test
    # for idx in tqdm(range(len(df_train) // 100)):
    for idx in tqdm(range(len(df_train))):
        audio_tensor = wavfile_to_examples(Config.data_root + 'train_audio/' + df_train.loc[idx, "filename"])
        pos = 0
        while pos < audio_tensor.shape[0]:
            if pos + 2 <= audio_tensor.shape[0]:
                audio_tensor_list.append( ( audio_tensor[pos:pos+2], label_dict[df_train.loc[idx, 'primary_label']] ) )
            else:
                break
            pos += 2

    print(audio_tensor_list[0][0].shape)
    return audio_tensor_list
    


class BirdDataset(Dataset):
    def __init__(self, audio_tensor_list):
        self.audio_tensor_list = audio_tensor_list

    def __len__(self):
        return len(self.audio_tensor_list)
    
    def __getitem__(self, idx):
        return self.audio_tensor_list[idx][0], self.audio_tensor_list[idx][1]


   
def set_dict(df_train):
    labels = []
    for idx in range(len(df_train)):
        labels.append(df_train.loc[idx, 'primary_label'])
    labels = list(set(labels))
    count = 1
    res_dict = {}
    for label in labels:
        res_dict[label] = count
        count += 1
    return res_dict




if __name__ == "__main__":

    # 添加tensorboard
    writer = SummaryWriter("./logs_train")


    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(Config.DEVICE)
    
    


    df_train = pd.read_csv(Config.train_path)

    label_dict = set_dict(df_train)
    print("len label_dict", len(label_dict))
    
    audio_tensor_list = cut_to_2_frames(df_train, label_dict)
    # shuffle
    random.shuffle(audio_tensor_list)

    train_dataset = BirdDataset(audio_tensor_list)

    print("len(audio_tensor_list)", len(audio_tensor_list))

    # 8:2
    valid_sample_size = len(audio_tensor_list) // 10 * 2
    indices = list(range(len(audio_tensor_list)))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[valid_sample_size:])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:valid_sample_size])



    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = Config.batch_size, 
        sampler = train_sampler,
        # shuffle = True, 
        drop_last = True, 
        num_workers = 5
    )

    valid_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = Config.batch_size, 
        sampler = valid_sampler,
        drop_last = True, 
        num_workers = 5
    )


    ## train ##
    total_train_step = 0
    total_test_step = 0


    model = VGGish(' ', torch.device(f'cuda:{CUDA_DEVICE}' if torch.cuda.is_available() else 'cpu'))
    model = model.to(Config.DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=Config.LR)

    for epoch in range(Config.epoch_num):
        # train
        model.train()
        print(f"epoch {epoch} / {Config.epoch_num}")
        for data in tqdm(train_dataloader):
            tensors, labels = data 
            tensors = tensors.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            print("tensors.shape:", tensors.shape)
            print("labels.shape:", labels.shape)

            output = model(tensors)

            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1

            if total_train_step % 100 == 0:
                writer.add_scalar("train_loss", loss.item(), total_train_step)



        # evaluate
        print("start evaluation")
        model.eval()
        total_eval_loss = 0
        total_acc = 0
        with torch.no_grad():
            for data in tqdm(valid_dataloader):
                tensors, labels = data
                tensors = tensors.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)

                output = model(tensors)
                loss = loss_fn(output, labels)
                total_eval_loss += loss.item()
                acc = (output.argmax(1)==labels).sum()
                total_acc += acc
        
        print("valid dataset total loss:", total_eval_loss)
        print("valid dataset total acc:", total_acc / valid_sample_size)
        writer.add_scalar("test_loss", total_eval_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_acc / valid_sample_size, total_test_step)
        total_train_step += 1

        # save
        torch.save(model, f"/home/chenjunhui/workspace/kaggle_competition/model/vggish_master/ckpts/Birdclef_vggish_epoch{epoch}.pth")
    
    writer.close()
    print("finish training!")




            
 


               
            




