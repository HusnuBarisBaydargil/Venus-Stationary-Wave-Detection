import torch.nn as nn
from utils import ResidualBlock, NonLocalBlock, GroupNorm, Swish, UpSampleBlock

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.channels = [512, 256, 256, 128, 128]
        self.attn_resolutions = [16]
        self.num_res_blocks = 3
        self.resolution = 16

        self.model = nn.Sequential(*self._build_layers(args))

    def _initial_layers(self, args):
        in_channels = self.channels[0]
        return [
            nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1),
            ResidualBlock(in_channels, in_channels),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels)
        ]

    def _residual_and_upsample_layers(self):
        layers = []
        in_channels = self.channels[0]

        for i, out_channels in enumerate(self.channels):
            for _ in range(self.num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if self.resolution in self.attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                self.resolution *= 2

        return layers

    def _final_layers(self, in_channels, image_channels):
        return [
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, image_channels, 3, 1, 1)
        ]

    def _build_layers(self, args):
        layers = self._initial_layers(args)
        layers.extend(self._residual_and_upsample_layers())
        layers.extend(self._final_layers(self.channels[-1], args.image_channels))
        return layers

    def forward(self, x):
        return self.model(x)
