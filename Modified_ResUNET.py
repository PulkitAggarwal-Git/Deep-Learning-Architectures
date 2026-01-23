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
    
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.conv1x1 = nn.Conv2d(skip_ch, out_ch, kernel_size=1)
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(2 * out_ch)

    def forward(self, x, skip_conn):
        skip_conn = self.conv1x1(skip_conn)
        x = self.up(x)
        x = torch.cat([x, skip_conn], dim=1)
        x = self.bn(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = DecoderBlock(512, 256, 128)
        self.block2 = DecoderBlock(256, 128, 128)
        self.block3 = DecoderBlock(256, 64, 128)
        self.block4 = DecoderBlock(256, 64, 128)

        self.final = nn.ConvTranspose2d(256, 3, kernel_size=2, stride=2)

    def forward(self, enc_output, skips):
        output = self.block1(enc_output, skips.pop())
        output = self.block2(output, skips.pop())
        output = self.block3(output, skips.pop())
        output = self.block4(output, skips.pop())
        output = self.final(output)

        return output
    
class ResUNET(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)

    def forward(self, x):
        enc_output, skips = self.encoder(x)
        dec_output = self.decoder(enc_output, skips)

        return dec_output
    
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResUNET(device).to(device)

x = torch.randn((1, 3, 512, 512)).to(device)
output = model(x)

print(output.shape)