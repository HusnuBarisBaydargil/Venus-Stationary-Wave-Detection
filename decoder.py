import torch
import torch.nn as nn
from helper_func import LayerNorm, GELU, UpSampleBlock

class Decoder(nn.Module):
    def __init__(self, image_channels=1):
        super(Decoder, self).__init__()

        self.up_blocks = nn.ModuleList([
            UpSampleBlock(128, 128),  # First upsampling, size goes from (128, 8, 8) to (128, 16, 16)
            LayerNorm(128, 16, 16),
            GELU(),
            UpSampleBlock(128, 64),  # Second upsampling, size goes from (128, 16, 16) to (64, 32, 32)
            LayerNorm(64, 32, 32),
            GELU(),
            UpSampleBlock(64, 32),  # Third upsampling, size goes from (64, 32, 32) to (32, 64, 64)
            LayerNorm(32, 64, 64),
            GELU(),
            UpSampleBlock(32, 32),  # Fourth upsampling, size goes from (32, 64, 64) to (32, 128, 128)
            LayerNorm(32, 128, 128),
            GELU()
        ])

        self.final_conv = nn.Conv2d(32, image_channels, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, skips):
        # print(f"Decoder input size: {x.size()}")
        for i, block in enumerate(self.up_blocks):
            x = block(x)
            # print(f"Output size after {type(block).__name__} {i}: {x.size()}")
            if isinstance(block, UpSampleBlock) and (i // 3) < len(skips):
                skip_index = -(i // 3 + 1)
                if x.size(2) == skips[skip_index].size(2) and x.size(3) == skips[skip_index].size(3):
                    x += skips[skip_index]
                    # print(f"Output size after adding skip from layer {len(skips) - (i // 3) - 1}: {x.size()}")

        x = self.final_conv(x)
        x = self.tanh(x)  
        # print(f"Output size after final Conv2d: {x.size()}")
        return x
