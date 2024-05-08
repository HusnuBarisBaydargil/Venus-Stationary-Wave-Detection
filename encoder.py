import torch.nn as nn
from helper_func import GELU, NonLocalBlock, LayerNorm
import torch

class Encoder(nn.Module):
    def __init__(self, image_channels=1):
        super(Encoder, self).__init__()
        self.channels = [32, 32, 64, 128, 128]
        self.convs = nn.ModuleList([
            nn.Conv2d(image_channels, self.channels[0], 3, 1, 1),
            nn.Conv2d(self.channels[0], self.channels[1], 3, 2, 1),
            nn.Conv2d(self.channels[1], self.channels[2], 3, 2, 1),
            nn.Conv2d(self.channels[2], self.channels[3], 3, 2, 1),
            nn.Conv2d(self.channels[3], self.channels[4], 3, 2, 1)
        ])
        self.nonlocal_blocks = nn.ModuleList([
            NonLocalBlock(self.channels[i]) for i in range(1, 5)
        ])
        self.norms_gelus = nn.ModuleList([
            LayerNorm(ch, 128 // (2**i), 128 // (2**i)) for i, ch in enumerate(self.channels)
        ])

    def forward(self, x):
        skips = []
        all_blocks = [None] + list(self.nonlocal_blocks)
        for idx, (conv, norm, nonlocal_block) in enumerate(zip(self.convs, self.norms_gelus, all_blocks)):
            x = conv(x)
            x = norm(x)
            x = GELU()(x)
            if nonlocal_block is not None:
                x = nonlocal_block(x)
            skips.append(x)
        return x, skips

# encoder = Encoder()
# nonlocal_blocks_count = len(encoder.nonlocal_blocks)
# print("Number of NonLocalBlock layers:", nonlocal_blocks_count)
# total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
# print("Total number of trainable parameters:", total_params)
# image = torch.randn(1, 1, 128, 128)
# encoded, skips = encoder(image)
# print("Encoded output size:", encoded.size())