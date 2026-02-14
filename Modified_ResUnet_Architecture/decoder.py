import torch
import torch.nn as nn
import torch.nn.functional as F

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