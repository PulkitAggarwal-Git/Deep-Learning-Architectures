# Paper Followed: Robust optic disc and cup segmentation with deep learning for glaucoma detection. Computerized Medical Imaging
# and Graphics

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder

class Model(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)

    def forward(self, x):
        enc_output, skips = self.encoder(x)
        dec_output = self.decoder(enc_output, skips)

        return dec_output

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model(device).to(device)

x = torch.randn((1, 3, 512, 512)).to(device)
output = model(x)

print(output.shape)