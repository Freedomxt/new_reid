import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import scipy.io
import pickle

import torch
import torchvision
import argparse
import torchvision.transforms as T

from model import build_model
from data import soft_label_loader

parser = argparse.ArgumentParser(description='Generate soft label')
parser.add_argument('-d', '--dataset', type=str, default='market1501')
parser.add_argument('-s', '--image_size', type=list, default=[384, 128])
parser.add_argument('-p', '--flip_p', type=float, default=0.5)
parser.add_argument('-pd', '--input_pading', type=int, default=10)
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-m', '--model_name', type=str, default='resnet50')
parser.add_argument('-bl', '--base_learning_rate', type=float, default=0.00035)
parser.add_argument('-wd', '--weight_decay', type=float, default=0.0005)
parser.add_argument('-e', '--Epochs', type=int, default=120)
parser.add_argument('-samppler', '--sampler', type=str, default="softmax")
parser.add_argument('-n_save', '--n_save', type=int, default=10)
parser.add_argument('-mp', '--model_path', type=str, default='./models')
args = parser.parse_args()

#define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#get the dataloader
train_loader,num_classes,dataset_train = soft_label_loader(args)

#load the resnet50 model
model = build_model(args,num_classes)
model_params_path = './models/resnet50/net_120.pth'
model_params = torch.load(model_params_path)
model.load_state_dict(model_params['model'])
model.to(device)
model.eval()


#process the data
soft_labels = []
with torch.no_grad():
    for data in tqdm(train_loader):
        img, label = data
        img = img.to(device)
        score = model(img)
        score = score.cpu().numpy()
        soft_labels.extend(score)


assert len(dataset_train) == len(soft_labels), 'dataset size if not match with softlabels'

dataset_new = []
for index,item in enumerate(tqdm(dataset_train)):
    img_name,pid,camid = item
    tmp = (img_name.split('/')[-1],pid,camid,soft_labels[index])
    dataset_new.append(tmp)


res = open('./soft_label/soft_label_resnet50.txt','wb')
pickle.dump(dataset_new,res)
print('Generate soft_labels finished!!!')
















    
    






