import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import torch.nn.functional as F


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



#define the STN network
class STN(nn.Module):
    def __init__(self,in_channels=2048):
        super(STN, self).__init__()

        #Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(2048,1024,kernel_size=3,padding=1),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True),
            nn.Conv2d(1024,512,kernel_size=3),
            #nn.MaxPool2d(2,stride=1),
            nn.ReLU(True),
            nn.Conv2d(512,256,kernel_size=1),
            #nn.MaxPool2d(2,stride=1),
            nn.ReLU(True)
        )
        #get the 3*2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            #nn.Linear(256*9*1,256),
            nn.ReLU(True),
            nn.Conv2d(256,6,kernel_size=1)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def stn(self,x):
        xs = self.localization(x)
        #xs = xs.view(-1,256*9*1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1,2,3)

        grid = F.affine_grid(theta,x.size())
        x = F.grid_sample(x,grid)
        return x

    def forward(self, x):
        x = self.stn(x)
        return x

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x


# Define the ResNet50-based Model
class ft_net(nn.Module):

    in_planes = 2048

    def __init__(self, class_num, stride=2):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft

        self.max_pooling = nn.AdaptiveMaxPool2d((1,1))
        self.down_conv = nn.Conv2d(4096,2048,kernel_size=1,stride=1,bias=False)
        init.kaiming_normal_(self.down_conv.weight,mode='fan_in')

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.classifier = nn.Linear(self.in_planes,class_num,bias=False)
        self.stn = STN(2048)


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        #x = self.stn.stn(x)
        x = self.model.avgpool(x)
        
        x = x.view(x.size(0), x.size(1))
        feat = x
        x = self.bottleneck(x)
        x = self.classifier(x)
        return x,feat




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
        feat = x.div(x.norm(p=2, dim=1, keepdim=True))
        #if self.reid:
            #return x
        # classifier
        x = self.classifier(x)
        return x,feat


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
    # Here I left a simple forward function.
    # Test the model, before you train it.
    net = ft_net(751, stride=1)
    net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 128))
    output = net(input)
    print('net output size:')
    print(output.shape)