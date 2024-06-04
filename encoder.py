import torch.nn as nn
from utils import ResidualBlock, NonLocalBlock, GroupNorm, Swish, DownSampleBlock

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.channels = [128, 128, 128, 256, 256, 512]
        self.attn_resolutions = [16]
        self.num_res_blocks = 2
        self.resolution = 128

        self.model = nn.Sequential(*self._build_layers(args))

    def _initial_layers(self, args):
        return [nn.Conv2d(args.image_channels, self.channels[0], 3, 1, 1)]

    def _residual_and_downsample_layers(self):
        layers = []
        resolution = self.resolution

        for i in range(len(self.channels) - 1):
            in_channels = self.channels[i]
            out_channels = self.channels[i + 1]
            for _ in range(self.num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in self.attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != len(self.channels) - 2:
                layers.append(DownSampleBlock(out_channels))
                resolution //= 2

        return layers

    def _final_layers(self, args):
        in_channels = self.channels[-1]
        return [
            ResidualBlock(in_channels, in_channels),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels),
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, args.latent_dim, 3, 1, 1)
        ]

    def _build_layers(self, args):
        layers = self._initial_layers(args)
        layers.extend(self._residual_and_downsample_layers())
        layers.extend(self._final_layers())
        return layers

    def forward(self, x):
        return self.model(x)
