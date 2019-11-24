import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import torchvision


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class MobileNet(nn.Module):

    def __init__(self,class_num):
        super(MobileNet, self).__init__()
        model_ori = models.mobilenet_v2(pretrained=True)
        self.features = model_ori.features
        classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, class_num),
        )
        classifier.apply(weights_init_classifier)
        self.classifier = classifier
        
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2,3])
        x = self.classifier(x)
        return x

