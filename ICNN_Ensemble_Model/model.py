import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset1 import Dataset_CNN1
from dataset2 import Dataset_CNN2
from dataset3 import Dataset_CNN3
from dataset4 import Dataset_CNN4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, 256)
        self.linear1 = nn.Linear(256,128)
        self.linear2 = nn.Linear(128,3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        x = self.maxpool(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        
        x = self.flatten(x)
        x = F.relu(self.linear(x))
        
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        
        return x
    
cnn1 = Model(3)
cnn2 = Model(1)
cnn3 = Model(1)
cnn4 = Model(1)