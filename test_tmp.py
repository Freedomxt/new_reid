import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import sys
import torch
import torchvision
from tqdm import tqdm
import time
from Logger import Logger

from data import make_data_loader
from model import build_model
from solver import make_optimizer,WarmupMultiStepLR
from layer import make_loss
from evaluate import Evaluator


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out), ]
    return nn.Sequential(*blocks)


class Net(nn.Module):
    def __init__(self, num_classes=751, reid=False):
        super(Net, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        # 32 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        # 64 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        # 128 16 8
        self.layer4 = make_layers(256, 512, 2, True)
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        # 256 1 1
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x,x
        # classifier
        x = self.classifier(x)
        return x


def main(args):
    train_loader, val_loader, num_query, num_classes, train_size = make_data_loader(args)

    #load the parameters
    net = Net(reid=True)
    state_dict = torch.load('./ckpt.t7', map_location=lambda storage, loc: storage)['net_dict']
    net.load_state_dict(state_dict)

    evaluator = Evaluator(net,val_loader,num_query)
    cmc, mAP = evaluator.run()
    print('---------------------------')
    print("CMC Curve:")
    for r in [1, 5, 10]:
        print("Rank-{} : {:.1%}".format(r, cmc[r - 1]))
    print("mAP : {:.1%}".format(mAP))
    print('---------------------------')



if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Konwledge distillation')
    parser.add_argument('-d','--dataset',type=str,default='market1501')
    parser.add_argument('-s','--image_size',type=list,default=[128,64])
    parser.add_argument('-p','--flip_p',type=float,default=0.5)
    parser.add_argument('-pd','--input_pading',type=int,default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-m', '--model_name', type=str, default='resnet50')
    parser.add_argument('-mo', '--modify', type=str, default='base')
    parser.add_argument('-bl', '--base_learning_rate', type=float, default=0.00035)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0005)
    parser.add_argument('-e', '--Epochs', type=int, default=120)
    parser.add_argument('-sampler', '--sampler', type=str, default="softmax")
    parser.add_argument('-n_save', '--n_save', type=int, default=10)
    parser.add_argument('-mp', '--model_path', type=str, default='./models')

    parser.add_argument('-lp', '--log_path', type=str, default='./logs')
    parser.add_argument('-ld', '--log_description', type=str, default='stn_partial')

    parser.add_argument('-mr', '--margin_tri', type=float, default=0.02)
    main(parser.parse_args())










