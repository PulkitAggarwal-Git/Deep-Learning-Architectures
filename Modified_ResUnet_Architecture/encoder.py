import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return F.relu(x + self.conv(x))
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.res_block1 = nn.Sequential(
            EncoderBlock(64),
            EncoderBlock(64),
            EncoderBlock(64)
        )

        self.conv_2 = nn.Conv2d(64, 128, kernel_size=1)
        
        self.res_block2 = nn.Sequential(
            EncoderBlock(128),
            EncoderBlock(128),
            EncoderBlock(128),
            EncoderBlock(128)
        )

        self.conv_3 = nn.Conv2d(128, 256, kernel_size=1)

        self.res_block3 = nn.Sequential(
            EncoderBlock(256),
            EncoderBlock(256),
            EncoderBlock(256),
            EncoderBlock(256),
            EncoderBlock(256),
            EncoderBlock(256)
        )

        self.conv_4 = nn.Conv2d(256, 512, kernel_size=1)

        self.res_block4 = nn.Sequential(
            EncoderBlock(512),
            EncoderBlock(512),
            EncoderBlock(512)
        )

    def forward(self, x):
        residue = x
        skip_connections = []
        
        x = self.conv_1(x)
        skip_connections.append(x)
        
        x = self.pool(x)
        x = self.res_block1(x)
        skip_connections.append(x)

        x = self.pool(x)
        x = self.conv_2(x)
        x = self.res_block2(x)
        skip_connections.append(x)

        x = self.pool(x)
        x = self.conv_3(x)
        x = self.res_block3(x)
        skip_connections.append(x)

        x = self.pool(x)
        x = self.conv_4(x)
        x = self.res_block4(x)

        return x, skip_connections