import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, dropout, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        out = self.dropout(F.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ModifiedResnet(nn.Module):
    def __init__(self, block, dropout=0.0, num_classes=10):
        super(ModifiedResnet, self).__init__()
           
        self.in_planes = 64
        
        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, padding_mode= 'replicate' , bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.part1_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode= 'replicate', bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.res_block1 = block(in_planes = 128, planes = 128, dropout= dropout, stride=1)

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.part3_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, padding_mode= 'replicate' , bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.res_block3 = block(in_planes = 512, planes = 512, dropout= dropout, stride=1)
        
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes, bias=False)
        )

    def forward(self, x):
        out = self.prep_layer(x)
        out = self.part1_1(out)
        block_output = self.res_block1(out)
        out = out + block_output 
        out = self.layer2(out)
        out = self.part3_1(out)
        block_output = self.res_block3(out)
        out = out + block_output
        out = F.avg_pool2d(out, 4)
        out = self.output_layer(out)
        
        return F.log_softmax(out, dim=1)


def ResNet18(dropout):
    return ModifiedResnet(BasicBlock,  dropout)
